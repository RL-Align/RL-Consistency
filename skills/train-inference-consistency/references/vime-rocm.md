# 在 AMD ROCm 上应用 skill：vime / Megatron / vLLM

当目标是 AMD GPU、vime 训练任务和独立于 RL-Kernel 的训推一致性时，使用本操作说明。先读 [SKILL.md](../SKILL.md)，保留默认 `consistency-only` 模式。本文提供环境准备与 agent 任务入口；安装成功、原生训练成功均不等于 bitwise 验收通过。

目录：

- [职责与路径](#职责与路径)
- [在宿主机创建容器](#在宿主机创建容器)
- [在容器中准备代码与环境记录](#在容器中准备代码与环境记录)
- [先排除工作目录挂载遮蔽镜像源码](#先排除工作目录挂载遮蔽镜像源码)
- [由 agent 整理可修改的运行环境](#由-agent-整理可修改的运行环境)
- [交给 agent 的任务](#交给-agent-的任务)
- [运行与交付](#运行与交付)

## 职责与路径

| 项目 | 本次用途 | 代码改在哪里 |
|---|---|---|
| RL-Consistency | C0–C10 的 Markdown 操作指南、数值契约、验收规则 | 此仓库保持 Markdown；不安装 Python 包 |
| vime | 组织采样、训练、权重同步，接入独立采集与验收 | `/workspace/vime` |
| Megatron-LM | 真实 old-logprob 重算与带梯度 actor 的训练计算 | agent 准备的独立源码工作区 |
| vLLM | 真实 rollout、prefill/decode、KV cache 与输出采集 | agent 准备的独立源码工作区 |
| 公共算术实现 | 固定计算顺序的 GEMM、attention、norm、logprob、MoE 等 | 上述目标项目中的可版本化模块；三条路径分别执行 |

先复用现有 `rl-kernel:rocm-dev` 镜像提供的 ROCm 软件栈，在**新容器**里检查、移除 RL-Kernel 依赖。镜像名称不决定运行时依赖；卸载一个包也不能证明框架补丁和 worker 已脱离该依赖。若无法复原可靠的独立软件栈，改用 vime 的 ROCm 镜像或构建配方。

## 在宿主机创建容器

下面是单机、单容器的起点。先确认分配给本任务的 GPU 及可用显存。参数 `4,5` 只是示例；不要据此占用这两张卡，也不要假设两张卡一定足够容纳完整 MoE 训练。

```bash
mkdir -p rl-consistency-ct
cd rl-consistency-ct
cat > docker.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

CONSISTENCY_GPUS="${1:?Usage: bash docker.sh GPU_IDS, for example 4,5}"
if [[ ! "$CONSISTENCY_GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  printf 'GPU_IDS must be comma-separated integer device indices\n' >&2
  exit 2
fi
CONSISTENCY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONSISTENCY_IMAGE="${CONSISTENCY_IMAGE:-rl-kernel:rocm-dev}"
CONSISTENCY_NAME="${CONSISTENCY_NAME:-rl-consistency-ct}"
mkdir -p "$CONSISTENCY_ROOT/cache/huggingface" "$CONSISTENCY_ROOT/artifacts"
CONSISTENCY_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$CONSISTENCY_IMAGE")"
printf '%s\n' "$CONSISTENCY_IMAGE_ID" > "$CONSISTENCY_ROOT/artifacts/container-image.txt"

docker run -itd \
  --name "$CONSISTENCY_NAME" \
  --privileged \
  --group-add video \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  --network bridge \
  --ipc private \
  --shm-size=64g \
  --ulimit memlock=-1:-1 \
  --ulimit nofile=1048576:1048576 \
  -v /home/amd/models:/app/model:ro \
  -v "$CONSISTENCY_ROOT:/workspace" \
  -w /workspace \
  -e CONSISTENCY_GPU_IDS="$CONSISTENCY_GPUS" \
  -e HIP_VISIBLE_DEVICES="$CONSISTENCY_GPUS" \
  -e CUDA_VISIBLE_DEVICES="$CONSISTENCY_GPUS" \
  -e VISIBLE_GPUS="$CONSISTENCY_GPUS" \
  -e HF_HOME=/workspace/cache/huggingface \
  -e HF_HUB_CACHE=/workspace/cache/huggingface/hub \
  -e WANDB_MODE=offline \
  --entrypoint /bin/bash \
  "$CONSISTENCY_IMAGE_ID"
BASH

# 将 4,5 改为已经分配给本任务的 GPU。
bash docker.sh 4,5
docker exec -it rl-consistency-ct bash
```

使用独立网络、IPC 和默认私有 PID 空间，让本容器的 Ray 端口、共享内存与进程管理独立于旧任务。`64g` 是可按宿主机内存调整的共享内存上限，不是 GPU 显存。跨容器或多节点任务需要另行配置网络；本配方不声称覆盖该范围。保留原有 ROCm 必需环境变量，先检查再调整。

原模型目录只读挂载到 `/app/model`；新下载、转换权重、源码和运行结果写入 `/workspace`，随宿主机工作目录保留。GPU 可见性变量是软件选择配置，并非硬件隔离；后续须在 Ray/训练/推理 worker 中核对实际设备映射。不要添加 `--pid=host`，不要清理旧容器或宿主机进程。重入现有容器用 `docker exec`；脚本不会自动删除同名容器。

## 在容器中准备代码与环境记录

只在新建容器内执行。克隆的 vime 是待核对的源码；先不要对未知旧镜像直接升级依赖。

```bash
cd /workspace
git clone https://github.com/RL-Align/RL-Consistency.git
git clone https://github.com/vllm-project/vime.git
git -C /workspace/vime switch -c consistency/rocm

export CONSISTENCY_BOOTSTRAP="$(mktemp -d /workspace/artifacts/bootstrap.XXXXXX)"
git -C /workspace/RL-Consistency rev-parse HEAD > "$CONSISTENCY_BOOTSTRAP/skill-commit.txt"
git -C /workspace/vime rev-parse HEAD > "$CONSISTENCY_BOOTSTRAP/vime-candidate-commit.txt"
python3 -m pip list --format=json > "$CONSISTENCY_BOOTSTRAP/packages.json"
set -o pipefail
python3 - <<'PY' | tee "$CONSISTENCY_BOOTSTRAP/probe.txt"
import importlib
import os
import torch

print("torch:", torch.__version__, "HIP:", torch.version.hip)
for key in ("CONSISTENCY_GPU_IDS", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES", "VISIBLE_GPUS"):
    print(key, "=", os.environ.get(key))
assert torch.version.hip, "The loaded PyTorch is not a ROCm build"
assert torch.cuda.is_available(), "No usable GPU in this container"
requested = os.environ["CONSISTENCY_GPU_IDS"].split(",")
assert len(set(requested)) == len(requested), "Duplicate GPU indices"
assert torch.cuda.device_count() == len(requested), "Inspect visibility/remapping"
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print("logical device", i, p.name, getattr(p, "gcnArchName", "unknown"), p.total_memory)
    x = torch.ones((16, 16), device=f"cuda:{i}")
    assert torch.isfinite(x @ x).all().item()
for name in ("vime", "vllm", "megatron.core", "ray", "transformers"):
    try:
        module = importlib.import_module(name)
        print(name, getattr(module, "__version__", "unknown"), getattr(module, "__file__", None))
    except Exception as exc:
        print(name, "IMPORT_FAILED", type(exc).__name__, str(exc))
PY
```

这是环境探测，不是 C3/C5 验收。框架出现 `IMPORT_FAILED` 时，先记录并修复环境，不能直接启动完整训练。PyTorch 在 ROCm 上仍使用 `torch.cuda` API。探测到的 `gcnArchName` 决定需要编译时的架构；不能因为某个脚本面向其他 AMD 卡，就把实际 `gfx942` 改成那个架构。

现在可以让 agent 进入该容器，以 `/workspace` 为工作目录，读取后面的任务。agent 安装在宿主机时，也可通过 `docker exec rl-consistency-ct ...` 执行容器内命令；最终验收必须发生在记录的容器与设备上。

## 先排除工作目录挂载遮蔽镜像源码

容器中的 `import vllm` 失败不一定表示镜像没有安装 vLLM。
上面的 `-v "$CONSISTENCY_ROOT:/workspace"` 会遮蔽镜像原有的整个
`/workspace`；镜像中的 editable 安装或 `easy-install.pth` 可能仍指向
被遮蔽的源码。先检查 `.pth`、`.egg-link` 和 editable finder 的目标是否存在，
再决定是否安装或编译。仅从 `pip list` 或 `find_spec` 不能区分缺包与失效绑定。

在宿主机记录目标容器的精确 image ID，并用不挂 GPU、不挂工作目录、
不联网的临时容器查看原始目录。例如，以下检查不运行模型：

```bash
CONSISTENCY_IMAGE_ID="$(docker inspect --format '{{.Image}}' rl-consistency-ct)"
docker run --rm --network none --entrypoint /bin/bash "$CONSISTENCY_IMAGE_ID" \
  -c 'ls -la /workspace'
```

若已确认所需源码就在镜像中，可创建一个**本任务独有的停止状态容器**，
用 `docker cp` 把确认过的源码目录及编译扩展恢复到独立工作区。目标目录
必须尚不存在，避免覆盖已有修改。保留 Git 提交、原有补丁和扩展文件；
核对当前 vime 所需 API 后，在目标运行容器中重新验证加载路径。
复制源码不等于已经证明 API/ABI 兼容。复制后的所有者可能与容器运行用户不同；
只对已确认属于本任务的路径处理 Git ownership 检查，不能全局关闭检查。
完成后仅删除本任务创建的辅助容器。

从包安装目录恢复 Megatron 源码时，记录原始 dist-info 和逐文件内容身份。
上游 SHA 不可取得时明确记为未知，不把版本号当作提交。确认
`megatron.core` **和** `megatron.training` 都从独立目录加载；
只追加一个 namespace 路径可能仍优先选中旧的 site-packages 实现。

环境探测要以新进程或脚本文件方式运行，而不只在仓库根目录执行
`python -c`：后者的当前目录可能让未安装的 vime 看似可以导入。
按下节的 editable 安装绑定源码后，重跑独立探测，并在真实 worker 中再次核对。
缺失 `vllm_router` 等纯上层依赖时，先按当前 vime 的 requirements 确认包名，
使用明确版本且不替换现有 ROCm 包的安装方式；保留原失败记录与安装日志。

## 由 agent 整理可修改的运行环境

1. **冻结实际环境。** 检查镜像原有 vime、Megatron、vLLM 的加载路径、源码提交、未提交修改、已编译扩展和必要 AMD 补丁。记录 ROCm/PyTorch/Transformer Engine/RCCL/Ray 版本、设备映射及相关环境变量。只记录与任务有关的配置，避免把凭据混入日志。检查模型快照、配置、tokenizer 和权重分片是否齐全，解析缓存符号链接，记录内容身份。
2. **对齐源码与依赖。** 优先以镜像中能正常运行的软件栈及其对应源码为起点。让 `/workspace/vime` 使用相匹配的提交，保留有用的 AMD 补丁及其来源；不要将镜像中的未提交补丁直接丢弃。Megatron/vLLM 需要改动时，准备独立工作区并按该版本的 ROCm 开发方式重建或绑定。不要拿最新 vime 源码配未知旧 vLLM API 后宣称环境已就绪。
3. **清除 RL-Kernel 接入。** 在新容器中卸载 `rl-kernel`；检查实际源码、启动入口、环境注入、`.pth`/`sitecustomize`、import hook、共享库与框架 monkey patch。移除或替换依赖它的 dispatch。保留原有必需的 ROCm/Torch Memory Saver 等修复，不能笼统清空 `LD_PRELOAD` 或 `PYTHONPATH`。在新启动的真实 worker 中核验已加载模块、算子提供方及源码路径；旧 worker 不可复用为卸载后的证据。
4. **绑定独立工作区。** 以下命令只在版本与构建依赖核对后执行。`--no-deps` 避免自动换掉 ROCm 依赖，但它不会自动修复不兼容版本。绑定后重新检查真实 worker 的导入路径；镜像中的 `.pth`、Ray `runtime_env` 或硬编码 `/root/Megatron-LM` 可能仍指向旧源码。

   ```bash
   python3 -m pip uninstall -y rl-kernel
   python3 -m pip install --no-deps --no-build-isolation -e /workspace/vime
   ```

5. **准备输入与任务入口。** 优先使用 `/app/model` 中已有的 Qwen3-8B/Qwen3-30B-A3B 原始权重；不要默认它们是普通目录或需要重新下载。按已选 vime 提交的转换工具建立 Megatron 权重，输出到 `/workspace/artifacts` 的独立目录，验证转换后的逻辑张量映射和数值身份。数据缺失时按该提交的训练示例准备实际目标函数所需数据；记录 tokenizer、chat template、响应 mask 和 loss 配置。新建启动脚本，保留分配的 GPU、唯一输出目录和本任务拥有的进程；不要直接执行上游示例中的通用 `pkill` 或清空旧 checkpoint。
6. **检查最小可运行链路。** 先确认模型加载、一次 rollout、训练 forward/backward、一次权重同步可运行。普通运行成功只解除环境阻塞；它不是原生性能基线，也不替代任何 bitwise/精度关卡。随后按 C0–C10 实现与验收。

如果镜像缺少兼容的软件栈或无法剥离已有集成，读取所选 vime 提交中的 [AMD 指南](https://docs.vllm.ai/projects/vime/en/latest/platform_support/amd_tutorial.html)、[ROCm Dockerfile](https://github.com/vllm-project/vime/blob/main/docker/Dockerfile.rocm) 或 [ROCm 7.14 Dockerfile](https://github.com/vllm-project/vime/blob/main/docker/Dockerfile.rocm7.14)，选择与硬件/宿主驱动兼容的配方。在新镜像中准备环境，记录配方提交、实际 base image ID、依赖提交和补丁；不要把浮动 `latest` 当作可复现标识。构建/导入成功仍不是一致性证据。

## 交给 agent 的任务

将下面任务原样交给能访问该容器和 `/workspace` 的 agent：

```text
读取 /workspace/RL-Consistency/skills/train-inference-consistency/SKILL.md，
以及 references/vime-rocm.md、references/correctness-first.md 和模型指南。

目标：在本 AMD ROCm 容器内，以 vime 为上层框架、Megatron 为训练路径、
vLLM 为 rollout 路径，先实现并验证 Qwen3-8B，再实现并验证 Qwen3-30B-A3B
的训推 bitwise 一致性。不考虑性能，不依赖 RL-Kernel。

先检查实际环境、模型和分配的 GPU，整理匹配的源码/依赖与独立工作区。
源模型在 /app/model；缓存、转换权重、运行结果写入 /workspace。
使用 /workspace/vime，并按需要修改独立的 Megatron/vLLM 源码工作区；
实现和测试写入目标项目，RL-Consistency 保持 Markdown。

不要只给计划：完成已授权的环境整理、代码修改、测试、诊断和复跑。
缺少硬件、权重或必要资源时精确记录 blocker，继续不依赖它的工作。
不要求性能优化，C1/C4 为 not-required。允许逐 token 计算、固定顺序
规约、显式 attention、顺序专家计算及公共算子代码。
保留三个实际执行入口，三条路径必须独立重新计算。

在运行前声明并记录模型/权重版本、精度、拓扑、外部 batch/序列长度范围、
prefill/decode、KV cache 和目标函数。保留真实 vime 训练入口与带梯度 actor；
不要用单独 HF/CPU 对比替代。需要缓存的案例必须实测缓存路径。
初始开发可以使用代表性减层模型，最终必须验证完整原始模型。

按 C0→C2→C3→C5→C6→C7→C8→C9→C10 推进：
独立 rollout、old-logprob 重算、actor forward；保留原始字节和中间边界；
首个不一致处做同输入 replay，修复实际提供方后重跑。
验证独立数学精度、真实梯度、两次有效更新，在 θ0/θ1/θ2 分别做同版本比较；
完成完整模型和 checkpoint 恢复后的首个 forward 与下一次更新。
MoE 额外独立比较 router logits、expert IDs/weights、dispatch 和 combine；
不得重放 rollout 的路由决定来伪造独立一致性。

把实际可执行的安装、启动、测试、复现命令及版本/证据保存到任务记录，
逐关更新 ledger/handoff。最终给两个模型分别列验收表；
未执行、不完整或失败的案例不能标为通过。不要以生成文本相同、KL 接近零、
allclose、统一随机种子或一次成功训练代替原始字节验收。
```

针对已核对的 vime 源码，重点检查以下位置；不同提交以实际调用关系为准：

| 位置 | 必须确认的行为 |
|---|---|
| `vime/backends/megatron_utils/actor.py` | 是否真正单独执行 old-logprob 重算；检查 `can_reuse_log_probs_in_loss` 等复用分支 |
| `vime/backends/megatron_utils/loss.py` | 是否将 rollout scores 或当前 actor 的 `detach()` 当作 old scores；验收采集必须独立执行 |
| `vime/rollout/vllm_rollout.py` | 在真实 rollout worker 记录输入、权重版本和未改写的数值输出 |
| `vime/backends/megatron_utils/update_weight/` | 训练到推理的张量映射、同步完成证据、版本与缓存失效 |
| 实际 Megatron/vLLM 模型及算子源码 | 公共数值提供方的绑定、fallback 拒绝和梯度完整性 |

仅设置 `use_rollout_logprobs=False` 不一定触发独立 old forward：某些路径会复用当前 actor scores。对验收用例显式执行、绑定并采集独立 old 分支；保留真实 RL 目标函数的 old-policy 冻结语义。θ1/θ2 使用新同步用例比较同版本三条路径，不能把旧轨迹的 old-policy 分数改成新策略分数。先核对当前提交提供的配置与接口，不要虚构“一键一致性”开关。

## 运行与交付

按“环境就绪 → Qwen3-8B 代表性层 → Qwen3-8B 完整模型 → Qwen3-30B-A3B 代表性层 → Qwen3-30B-A3B 完整模型”推进。复用已验证且数值契约兼容的公共实现，但 MoE 的独立路由、专家、combine、梯度和完整模型必须重新验收。内存不足时可以继续代理模型开发；完整模型保持 blocked，不能缩专家数量后冒充原模型通过。

要求 agent 在目标工作区交付：源码提交/补丁、可重复安装步骤、实际启动脚本、输入与权重身份、逐关记录、独立捕获的原始证据、首差异诊断及完整模型/恢复测试报告。容器内临时手改的 site-packages 必须转为可重建的源码修改；记录编译扩展与实际加载路径，避免重启后消失。

| 需要区分的结果 | 可以说明什么 |
|---|---|
| 环境探测与原生最小链路成功 | 软件栈能运行；尚未证明 bitwise 一致性 |
| 代理模型通过 | 相应算子/层路径得到验证；完整模型仍待验证 |
| 某模型全部必需关卡通过 | 该模型在记录的权重、设备、精度、拓扑和输入范围内完成 consistency-only 验收 |
| 两个模型通过 | 两个独立验收范围成立；不能外推为所有模型、所有 AMD 卡或所有配置 |

用 [release-report.md](../templates/release-report.md) 分别报告两个模型。C1/C4 为 `not-required`，性能为 `not evaluated`。此指南能为 agent 提供具体实施路线；需要开发多少算子、需要多久，以及能否在当前资源上完成，必须由实际运行确定。

# ROCm 实机诊断：Qwen3-8B

这条入口用于你已有的 **8×MI300X / gfx942 + vime + Megatron + vLLM + RL-Kernel**
环境。运行后直接查看逐 token 原始位差异、有效元素数、首个失败位置和 HTML 报告。
本仓库没有进行 GPU 实测，以下是待你在实机执行的入口。

## 1. 准备现有环境

保留已安装的 ROCm PyTorch、vLLM、vime、Megatron 和对应 companion patches。
在同一个 Python 环境里安装本项目；本项目不会安装或替换这些依赖：

```bash
cd /path/to/RL-Consistency
git pull --ff-only
/opt/venv/bin/python -m pip install -e .
```

将 `/opt/venv/bin/python` 换成你的真实运行环境。命令行进程与 profile 指定的
Python 必须属于同一个环境，以便启动和 `.pt` 数据读取使用同一套 PyTorch。

启动器固定参考 RL-Kernel `76a127c206b7ead6fb837b46264ba43f9ef20e82`。
若现有主仓库已经更新，可单独创建 worktree，避免回退主工作目录：

```bash
git -C /path/to/RL-Kernel worktree add /workspace/RL-Kernel-consistency \
  76a127c206b7ead6fb837b46264ba43f9ef20e82
```

需要已准备好的 Qwen3-8B HF 权重、匹配的 Megatron torch-dist checkpoint、
DAPO JSONL 数据，以及与 pinned RL-Kernel 兼容的框架版本/补丁。
这条命令不会下载模型、转换 checkpoint 或修补框架；缺失依赖会报告错误。
源码目录须可读取 Git 身份。RL-Kernel worktree 必须干净；框架 companion patches
允许保留，诊断会记录其状态，后续正式验收仍需冻结和审核完整来源。

## 2. 配置路径和真实 workload

优先复制你已经使用的 RL-Kernel machine profile，补上 `paths.rl_kernel_root`。
也可以从示例开始：

```bash
cp configs/rocm-mi300x.example.json local.rocm.json
```

编辑 `local.rocm.json` 中的路径：`workspace`、`rl_kernel_root`、`vime_root`、
`megatron_root`、`vllm_root`、`python`、`model_root`、`ref_load`、`prompt_data`。
全部使用绝对路径。示例是 8 个样本；若你的真实 global batch 为 128，保持
`global-batch-size=128`、`rollout-batch-size=16`、`samples-per-prompt=8`。
温度、top-p、长度、microbatch 和显存比例也应沿用你要验证的配置。

先查看解析后的完整命令与设置，不占用 GPU：

```bash
/opt/venv/bin/python -m rl_consistency rocm-plan \
  --profile local.rocm.json --output runs/rocm-check-001
```

诊断明确覆盖为：**1 轮，lr=0，weight-decay=0**；保留 profile 的数据、批量、
长度、采样、并行、KL 和算子路径。它可能执行 backward 和 optimizer machinery，
因此不等同于纯推理，也不构成有效更新验收。避免首轮 mismatch 后继续改变策略。
当前 bridge 仅接受训练 TP×CP=8、rollout TP1/2/4/8 + CP1、top-k=-1。

## 3. 实际运行

确认所分配的八张 GPU 可用于该实验，再执行：

```bash
/opt/venv/bin/python -m rl_consistency rocm-check \
  --profile local.rocm.json --output runs/rocm-check-001
```

`rocm-check` 会启动真实 GPU 作业。它依次检查路径、源码版本、运行环境和模型
配置，调用 pinned RL-Kernel 入口，等待结束，再读取所有 rank 的原始 sidecar。
上游启动器会拒绝复用/停止已存在的 Ray 集群；不会由本工具清理其他作业。
输出目录必须是新的，避免混用历史产物。

另一个终端可以查看实际引擎日志：

```bash
tail -f runs/rocm-check-001/upstream/arms/r-r/launcher.log
```

结束后查看：

| 文件 | 内容 |
|---|---|
| `report.html` | 诊断结论、比较数量、失败原因、未覆盖项目 |
| `result.json` | `pass` / `fail` / `inconclusive`，供脚本读取 |
| `diagnostic.json` | 原始值、dtype、逐 token 位置、来源文件哈希、位差异 |
| `plan.json` / `profile.json` | 输入 profile 身份和实际解析配置 |
| `preflight.json` | 主进程设备、框架来源和数据文件身份 |
| `upstream/arms/r-r/` | 引擎日志、worker readbacks、原始 `.pt`、上游验证结果 |

若仅需重新生成报告，不重新跑 GPU：

```bash
/opt/venv/bin/python -m rl_consistency rocm-report --run-dir runs/rocm-check-001
```

## 4. 如何判断结果

- `pass` / 退出码 0：本次原始有效 logprob 位差异为 0，rank/sample 覆盖完整，
  上游验证通过且作业正常退出。结论限于这次已记录的 selected-token 数据。
- `fail` / 退出码 1：观察到位差异，保留首个失败位置和原始值，继续定位。
- `inconclusive` / 退出码 2：环境失败、缺 rank/样本、空比较、非有限值、缺产物、
  非正常退出等，不能当作一致性通过。先看 `errors` 和 launcher 日志。

所有结果均保持 `certified: false`。现有 sidecar 缺少完整的已加载权重身份、
native old-logprob 与 grad-enabled actor 的双路径执行回执，不能自动通过 C5。
这次也没有验证 BI/PD 全矩阵、独立梯度、两次有效更新后的 θ2、checkpoint 恢复、
质量或性能，HTML 中 C0–C10 均显示 `not run`，不会把诊断结果包装成完整验收。

想观察原生差异，可另用新输出目录运行 `--mode native`。两次独立采样不能当作
受控性能 A/B，也不能用两个模式的不同 rollout 轨迹直接比数值；每次诊断只比较
该次同一批 token 的 rollout 与 training。

完整工作流使用 [v2 gate 定义](../skills/train-inference-consistency/references/gates.md)
和 [证据格式](evidence-format.md)。诊断与正式验收的范围见
[工作流说明](../skills/train-inference-consistency/references/workflow-scope.md)。

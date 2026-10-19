# Qwen3.8-27B：第二模型接入

For the independent shared backend, start with [independent-runtime.md](independent-runtime.md). The engine integration/preparation described below is the optional legacy route; its status does not describe the independent executor.

本版本已实现 **Qwen3.8-27B 的 skill 接入和 CPU 准备工具**：真实 config 检查、
Hybrid 算子清单、proxy 配置、CUDA/ROCm 候选配方、模型专用验收约束，以及
独立 FP64 GDN recurrence 精度参考。

**尚未实现完整 Hybrid strict runtime 或进行 GPU 一致性验收。**
`prepare-model` 不启动 GPU、不生成权重，也不通过任何 C0–C10 gate。
Qwen3-8B 的 `rocm-check`/`run` 入口不能用于这个模型。

## 1. 在现有环境安装并准备

```bash
git pull --ff-only
python -m pip install -e .
python -m rl_consistency list

python -m rl_consistency prepare-model \
  --model Qwen/Qwen3.8-27B \
  --config /path/to/Qwen3.8-27B/config.json \
  --backend rocm \
  --output runs/qwen38-27b-prepare-001

python -m rl_consistency check \
  --recipe runs/qwen38-27b-prepare-001/recipe.json
```

将 config 路径换成你实际下载的模型。CUDA 使用 `--backend cuda`，两端单独
验证。输出目录必须是新目录。无需 GPU 或导入 vime/Megatron/vLLM，即可执行
这个准备步骤。模型名是用户声明的身份，配置兼容不能证明权重属于该模型。

只检查配置、不写文件时使用：

```bash
python -m rl_consistency inspect-model \
  --model Qwen/Qwen3.8-27B \
  --config /path/to/Qwen3.8-27B/config.json
```

## 2. 准备产物

| 文件 | 用途 |
|---|---|
| `recipe.json` | 候选验收配置；已绑定本次 config 的原始和规范化哈希，其余真实身份待填 |
| `inventory.json` | 64 层结构、48 层 GDN/16 层全注意力、算子和状态边界 |
| `proxy/config.json` | 默认 8 层、维度不变的 proxy 配置；不含权重 |
| `proxy-plan.json` | 原始层映射和语义描述；`checkpoint_materialized=false` |
| `vime-model-args.json` | 完整模型和 proxy 的模型参数数组；不是完整启动命令 |
| `source-config.json` | 输入配置的规范化副本；其字节哈希可能与原文件不同 |
| `preparation.json` | 文件哈希、待补项、所有 GPU gate 的 `not-run` 状态 |

两份候选配方分别为 `qwen3-8-27b-vime-rocm` 和 `qwen3-8-27b-vime-cuda`。
默认规划为 8 GPU、训练 TP2/PP4/CP1、rollout TP4。**这些是待验证设置，
不是上游已测配置，也不保证显存可容纳。** 准备命令不会占用 GPU 或改变工作量。
使用现有集群时，按实际资源和 workload 修改候选配方，再开始冻结验收。

## 3. 让 skill 继续适配

向加载本仓库 skill 的 agent 提供模型和三套框架的实际路径，例如：

> 使用 train-inference-consistency 接入 Qwen/Qwen3.8-27B，执行 BF16 纯文本
> RL 的 ROCm 训推一致性适配。读取本次 prepare-model 输出，保留原始维度和
> L/L/L/F 顺序，优先复用已存在的算子，从 8 层 proxy 开始定位。按真实结果
> 推进 C0–C10，记录未覆盖的模块，不把准备成功视为数值验收通过。

具体模块、来源、训练反向和状态验证任务见
[Qwen3.8 skill 指南](../skills/train-inference-consistency/references/qwen38-27b.md)。
Agent 需要继续接入 GDN/短卷积等严格算子、实际 worker capture、proxy 权重和
全模型启动流程，并在你的硬件上完成测试。未完成之前，不能承诺模型开箱即用
的 bitwise 结果。

官方 checkpoint 带视觉模块。首个范围是 **BF16 text-only**，不包含图像/视频、
FP8、MTP 训练和投机解码。这些扩展需要单独的算子、采集和验收范围。

## 4. 独立精度参考

`rl_consistency.models.gdn_reference.gated_delta_reference` 使用可求导的 FP64
数学递推，支持 packed 序列、独立初始状态和 Q/K 到 value heads 的分组扩展。
其 CPU 测试包含手算结果、状态恢复和有限差分梯度检查。可用于排查 GDN core
的误差，不替代完整算子的精度参考、GPU bitwise 验证或生产实现。

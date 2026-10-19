# RL-Consistency

项目定位：**独立的一致性执行后端 + 模型适配器 + 开箱运行的 skill**。默认入口不需要安装 RL-Kernel，也不要求 vLLM 或 Megatron。

公共后端负责控制数值执行、独立重算、逐位比较和失败处理；模型适配器负责识别结构、加载真实权重、声明状态与路由观察点。已有算子覆盖的模型复用公共后端，新增算子仍需要开发和验证。

## 模型支持与绝对一致性状态

| 模型 | 独立运行路径 | 已完成的自动化验证 | 完整模型 CUDA / ROCm 绝对一致性 |
|---|---|---|---|
| Qwen3-8B，Dense | 生成、独立 old-logprob 重算、actor forward/backward | 小型真实架构 FP32/BF16 CPU 逐位比较、权重加载、两次有效更新 | 待验收 / 待验收 |
| Qwen3.8-27B，Hybrid Dense 文本路径 | 同一后端，完整前缀 GDN；按实际 `qwen3_5` 配置接入 | 小型真实架构 FP32/BF16 比较、嵌套 checkpoint 文本权重加载、两次有效更新 | 待验收 / 待验收 |
| Qwen3-30B-A3B，MoE | 同一后端，额外比较独立路由 ID、权重与 logits | 小型真实架构 FP32/BF16 比较、权重加载、两次有效更新 | 待验收 / 待验收 |

小型架构测试使用 Transformers 的真实模型实现、缩小维度和随机权重，**不能当成上述完整模型的验收结果**。当前没有模型可宣称已经在本仓库完成完整 ROCm 绝对一致性验收。

## 直接运行

在隔离环境中先安装对应平台的 PyTorch；AMD 使用 ROCm 版。然后执行：

```bash
python -m pip install -e '.[runtime]'
rl-consistency runtime-models

rl-consistency runtime-check \
  --model Qwen/Qwen3-8B \
  --checkpoint /models/Qwen3-8B \
  --fixture examples/runtime-tokens.json \
  --device cuda:0 --dtype bfloat16 \
  --max-new-tokens 2 --output runs/qwen3-independent-001
```

ROCm 同样使用 `cuda:0`。换模型只需更换 `--model`、真实本地 checkpoint 和输出目录。也可将 `--fixture` 换成 `--prompt '解释一下浮点数运算。'`，使用 checkpoint 中的 tokenizer。

增加 `--updates 2` 会执行两次真实 SGD 校准更新，并分别检查 θ0/θ1/θ2。同一 θ 内比较；不会拿不同权重版本的输出比较。这是基于生成 token 的负对数似然校准，不是完整 RL 训练，也不保存训练后 checkpoint。

`result.json` 记录输入、权重与实现身份、比较范围和结果；每个 θ 保存生成时的原始观察值以及独立重算结果。退出码 0 表示当前用例观察值完全相等；1 表示不一致；2 表示错误或不支持。没有“忽略误差继续通过”的选项。

## 首版实现范围

`shared-prefix-v1` 固定单设备、物理 batch=1、eager attention/experts、不使用 KV cache，每个 token 都重新计算完整前缀。生成、重算、actor 都在 eval 模式下启用 autograd，只有 actor 执行反向传播。比较原始字节，包括 signed zero；不是仅比较生成文本。

这一版优先建立共同执行路径，吞吐较低，完整前缀重算、状态哈希和 CPU 采集都有开销。eager attention、权重和梯度需要足够显存，27B/30B 可能无法在单卡完成；当前不会自动切换成多卡或 offload。环境中不能安装 `kernels`，Hybrid 还排除 `fla`、`causal_conv1d`，防止自动选择另一套数值实现。

通过一次检查只说明记录的输入、权重、环境和路径相等。它不代表跨 GPU、跨 CUDA/ROCm、原生动态 batch、KV cache/PD、vLLM/Megatron 双引擎或优化器轨迹一致，也不会自动通过 C0–C10。完整模型和目标环境仍要保存实机结果；不能把“安装 skill”当成验证证据。

详见 [独立后端契约和模型扩展](independent-runtime.md)。已有的 [首个分歧定位器](diagnostics.md) 和 [C0–C10](../skills/train-inference-consistency/references/gates.md) 继续可用；原来的 RL-Kernel 启动桥接是可选兼容入口，不影响独立后端。

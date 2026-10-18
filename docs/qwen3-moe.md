# Qwen3-30B-A3B：第三模型接入

已实现模型配置检查、MoE 算子清单、proxy 准备、CUDA/ROCm 候选配方，以及
C3/C5/C8/C9 的路由证据检查。**当前是 skill 接入和验收工具，尚未实现完整
MoE strict runtime，也没有 GPU 验收结果。** `prepare-model` 不启动 GPU、
不生成权重；Qwen3-8B 的 `rocm-check`/`run` 不能用于该模型。

## 准备命令

```bash
git pull --ff-only
python -m pip install -e .
python -m rl_consistency prepare-model \
  --model Qwen/Qwen3-30B-A3B \
  --config /path/to/Qwen3-30B-A3B/config.json \
  --backend rocm \
  --output runs/qwen3-30b-a3b-prepare-001

python -m rl_consistency check \
  --recipe runs/qwen3-30b-a3b-prepare-001/recipe.json
```

替换实际 config 路径。CUDA 使用 `--backend cuda`；输出目录必须是新目录。
仅查看配置时使用 `inspect-model --model Qwen/Qwen3-30B-A3B --config PATH`。
准备阶段无需 GPU。模型名称是用户声明，配置相同不能证明 checkpoint 身份；
实际 revision、权重、tokenizer、输入和运行环境仍需绑定。

## 模型与 proxy

| 项目 | 完整模型 | 默认 proxy |
|---|---|---|
| 层 | 48 个全注意力 + MoE 层 | 前 4 个完整层 |
| hidden / attention heads / KV heads / head dim | 2048 / 32 / 4 / 128 | 全部保留 |
| 专家数 / 每 token 选择数 | 128 / 8 | 全部保留 |
| 每专家 FFN 宽度 | 768 | 保留 |
| 权重 / router 目标精度 | BF16 / FP32 | 保留 |

`--proxy-layers` 可指定 4–47 层连续前缀。仅修改 `num_hidden_layers`，其余
config 字段保留，包括原始 `router_aux_loss_coef`。实际训练配方显式选择
aux coefficient 为 0，与所检查的 vime 模型参数一致；不能混淆两者。

产物包括 `recipe.json`、`inventory.json`、`proxy-plan.json`、
`proxy/config.json`、`source-config.json`、`vime-model-args.json` 和
`preparation.json`。模型参数数组是上游接入参考，不是启动器或严格数值实现。
proxy 权重需要从实际 checkpoint 复制全部选定层、router、128 个专家及公共
embedding/norm/head，保留真实 tensor 名称、精度和哈希，不可随机初始化替代。

## MoE 验收增加了什么

- C3/C5/C8/C9 每个输出比较都需要对应 `routing_pairs`。专家 ID 按整数比较，
  top-k 槽位顺序保留，归一化后的 FP32 路由权重按原始字节比较。
- 每个 token 都要覆盖 proxy 的全部层；C9 要覆盖全部 48 层。
  ledger 的权重版本、输入、worker 角色和双路径 callsite 必须与输出一致。
- 检查 dispatch 收据没有丢失或重复槽位，unpermute 后 expert ID 对得上，
  combine 按约定槽位顺序执行。收据必须来自真实 worker；JSON 声明本身
  不证明 kernel 执行，需要审查绑定、源码、rank 覆盖和原始中间值。
- C7 增加 router、expert、combine 的独立梯度精度检查；C9 增加 expert
  parallel、同步后路由和 uninterrupted/resumed 路由比较。

本次范围为 BF16 文本、独立路由、无 token dropping、无投机解码。FP8/INT4、
R3、DeepEP 等上游选项不会自动成为该配方的已验证能力。相同专家输出或最终
logprob 可能掩盖路由漂移，所以路由不一致时即使输出一致也不能通过。

## 在 ROCm 上继续实现

向加载本仓库 skill 的 agent 提供模型、RL-Kernel、vime、vLLM 和 Megatron
实际路径，以及本次准备目录。按 [MoE skill 指南](../skills/train-inference-consistency/references/qwen3-30b-a3b.md)
接入严格算子、两条训练路径和路由采集器，再推进 C0–C10。

候选起点为 8 GPU：训练 TP4/CP1/PP1/EP8/ETP1（dense DP2、expert DP1），
rollout TP8/EP1/ETP8。Megatron 的 TP 与 EP 使用有重叠的进程组，不能简单
把 TP×EP 相乘当作 GPU 数。实际 rank 映射、资源复用方式、显存和 workload
必须先检查。上述设置不是已验证 ROCm 配方，也不保证显存可容纳。

完成真实 GPU 验收前，这个模型的状态保持 `candidate`，不能承诺开箱即用的
绝对一致性，也不能从 Qwen3-8B 的结果推导该模型通过。

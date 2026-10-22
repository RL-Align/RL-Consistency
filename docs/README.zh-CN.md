# RL-Consistency

这是一个 **Markdown 编写的通用 agent skill 项目**。把模型、训练框架、推理框架和硬件环境交给其他 agent，agent 按 C0–C10 在目标项目中开发代码、定位分歧、运行验证，最终交付有实测证据的训推一致性实现。

本仓库交付方法、开发步骤、数值契约、排错流程和验收模板。使用 skill 无需安装本仓库的 Python 包，也不依赖 RL-Kernel；实际模型的运行环境由目标项目决定。

## 怎么交给另一个 agent

将整个 `skills/train-inference-consistency/` 文件夹放到该 agent 的 skill 目录，或者让它直接读取 [SKILL.md](../skills/train-inference-consistency/SKILL.md)。保留相对路径引用的 Markdown 文件。

可以直接给它这段任务：

> 使用这个 train-inference-consistency skill，为【模型】在【训练仓库】与【推理仓库】、【硬件】上实现训推 bitwise 一致性。先检查真实 checkpoint、框架源码和当前执行路径，再按 C0–C10 推进。代码和测试写进目标项目；独立采集 rollout、old-logprob 重算与 actor forward；保留原始失败样本和每关证据。未实际执行的验证保持待完成，不要缩小生产执行范围后宣称原目标已经通过。

## C0–C10 做什么

| 阶段 | agent 要完成的事情 |
|---|---|
| C0 | 查清模型、算子、框架和硬件路径，定义比较范围与数值契约 |
| C1 | 把原生基线调好、测量并冻结 |
| C2 | 使用原始权重与原始层内尺寸，构建有代表性的减层模型 |
| C3 | 开发并验证推理重复、batch、padding、prefill/decode 等一致性 |
| C4 | 检查实际算子形状、调用次数、同步、workspace 和 Graph 的结构性性能问题 |
| C5 | 让 old-logprob 独立重算和带梯度 actor forward 分别对齐 rollout |
| C6 | 完成 backward 接入，检查保存状态、梯度语义和训练开销 |
| C7 | 使用独立数学参考验证前向、梯度和更新精度 |
| C8 | 真实更新两次，在 θ0、θ1、θ2 各自进行同权重版本的训推比较 |
| C9 | 完整模型、上下文与拓扑矩阵，以及 checkpoint 恢复后的前向和更新验证 |
| C10 | 交付实现、复现方法、验证边界和公平的性能结果 |

每一关都有前置条件、代码工作、验证方式、所需产物、通过标准和失败后的返回路径。遇到未知模型，可以先按语义依赖图定位首个观察到的不一致，再逐步补适配。

## 模型指南与绝对一致性状态

| 模型 | 已提供的开发指南 | 完整模型绝对一致性验收 |
|---|---|---|
| Qwen3-8B | Dense、attention、GEMM、norm、RoPE、logprob 与规约 | 本 skill 项目未记录完整验收；已有外部结果只适用于其原始证据范围 |
| Qwen3.8-27B | Hybrid 文本路径、卷积状态、递归状态与注意力层交互 | 待完成 |
| Qwen3-30B-A3B | 独立路由、dispatch、专家计算和 combine | 待完成 |
| 其他模型 | 通用模型扩展模板与排错流程 | 按目标环境实际开发和验收 |

新增模型时增加对应 Markdown 资料和测试要求，公共 C0–C10 流程复用。新增算子、量化方式或状态路径仍需要 agent 在目标项目里开发代码，不能仅增加模型名称就宣称支持。

完整目录和入口见 [主 README](../README.md)。之前的 Python 实验实现已保留在 [archive/runtime-v0.3](https://github.com/RL-Align/RL-Consistency/tree/archive/runtime-v0.3)，主分支交付内容全部为 Markdown。

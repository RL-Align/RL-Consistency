# RL-Consistency

这是一个 **Markdown 编写的通用 agent skill 项目**。把模型、训练框架、推理框架和硬件环境交给其他 agent，agent 按 C0–C10 在目标项目中开发代码、定位分歧、运行验证，最终交付有实测证据的训推一致性实现。**默认只要求一致性与数值正确性，不设性能目标；慢但正确的实现可以作为最终交付。**

本仓库交付方法、开发步骤、数值契约、排错流程和验收模板。使用 skill 无需安装本仓库的 Python 包，也不依赖 RL-Kernel；实际模型的运行环境由目标项目决定。

## 怎么交给另一个 agent

将整个 `skills/train-inference-consistency/` 文件夹放到该 agent 的 skill 目录，或者让它直接读取 [SKILL.md](../skills/train-inference-consistency/SKILL.md)。保留相对路径引用的 Markdown 文件。

如果准备在 AMD 卡上用 vime 验证，直接按 [ROCm/vime 环境与 agent 任务](../skills/train-inference-consistency/references/vime-rocm.md) 操作：从现有镜像创建独立容器，检查并剥离旧 RL-Kernel 接入，再让 agent 修改实际 vime/Megatron/vLLM 路径。文档含可复制的 Docker 命令、环境探测和完整任务提示。先做 Qwen3-8B，再做 Qwen3-30B-A3B；环境安装成功与模型一致性验收分别记录。

可以直接给它这段任务：

> 使用这个 train-inference-consistency skill，为【模型】在【训练仓库】与【推理仓库】、【硬件】上实现训推 bitwise 一致性。先检查真实 checkpoint、框架源码和当前执行路径，再按 C0–C10 推进。代码和测试写进目标项目；独立采集 rollout、old-logprob 重算与 actor forward；保留原始失败样本和每关证据。不考虑性能，优先实现简单、可控制计算顺序的公共算术路径，允许逐 token 计算和串行规约。C1/C4 标记为 not-required；保留明确要求的模型、硬件和执行行为，未实际执行的验证保持待完成。

## C0–C10 做什么

| 阶段 | agent 要完成的事情 |
|---|---|
| C0 | 查清模型、算子、框架和硬件路径，定义比较范围与数值契约 |
| C1 | 可选：原生性能基线；默认 not-required，不阻塞一致性 |
| C2 | 使用原始权重与原始层内尺寸，构建有代表性的减层模型 |
| C3 | 开发并验证推理重复、batch、padding、prefill/decode 等一致性 |
| C4 | 可选：结构性性能优化；默认 not-required |
| C5 | 让 old-logprob 独立重算和带梯度 actor forward 分别对齐 rollout |
| C6 | 完成真实 backward，检查保存状态、梯度语义和覆盖 |
| C7 | 使用独立数学参考验证前向、梯度和更新精度 |
| C8 | 真实更新两次，在 θ0、θ1、θ2 各自进行同权重版本的训推比较 |
| C9 | 完整模型、上下文与拓扑矩阵，以及 checkpoint 恢复后的前向和更新验证 |
| C10 | 交付实现、复现方法、验证边界和一致性/精度证据 |

默认必经流程为 **C0→C2→C3→C5→C6→C7→C8→C9→C10**。C1/C4 保留原编号和职责，只有用户要求性能工作时才启用。没有原生测速、profile 或吞吐数据，不妨碍完成一致性验收。

参照[不考虑性能的实现步骤](../skills/train-inference-consistency/references/correctness-first.md)，agent 可以在目标项目中实现公共的逐 token 投影、固定顺序规约、显式 attention、顺序递推、逐专家计算和有序 combine，并分别接入 rollout、old-logprob 重算和带梯度 actor。三条路径可以共用代码，但必须各自重新计算。真实梯度、独立数值精度、两次有效更新和完整模型恢复验证仍然必需。

每一关都有前置条件、代码工作、验证方式、所需产物、通过标准和失败后的返回路径。遇到未知模型，可以先按语义依赖图定位首个观察到的不一致，再逐步补适配。

## 模型指南与绝对一致性状态

| 模型 | 已提供的开发指南 | 完整模型绝对一致性验收 |
|---|---|---|
| Qwen3-8B | Dense、attention、GEMM、norm、RoPE、logprob 与规约 | 已完成一次限定范围的 MI300X ROCm/vime/Megatron/vLLM 完整验收（2026-10-23），含有效更新与精确恢复；见 [8B 实测记录](../skills/train-inference-consistency/models/qwen3-8b.md#measured-rocm-run--2026-10-23)，其他环境与模型不外推 |
| Qwen3.8-27B | Hybrid 文本路径、卷积状态、递归状态与注意力层交互 | 待完成 |
| Qwen3-30B-A3B | 独立路由、dispatch、专家计算和 combine | 待完成 |
| 其他模型 | 通用模型扩展模板与排错流程 | 按目标环境实际开发和验收 |

新增模型时增加对应 Markdown 资料和测试要求，公共 C0–C10 流程复用。新增算子、量化方式或状态路径仍需要 agent 在目标项目里开发代码，不能仅增加模型名称就宣称支持。

完整目录和入口见 [主 README](../README.md)。之前的 Python 实验实现已保留在 [archive/runtime-v0.3](https://github.com/RL-Align/RL-Consistency/tree/archive/runtime-v0.3)，主分支交付内容全部为 Markdown。

# RL-Consistency

这是 RL-Kernel 上层的通用 Skill 和验收工具。Skill 规定开发顺序，工具检查证据，底层数值计算继续由 RL-Kernel 承担。

## 模型支持与绝对一致性状态

| 模型 | 架构 | CUDA 接入 | ROCm 接入 | 本仓库绝对一致性验证状态 |
|---|---|---|---|---|
| [Qwen3-8B](qwen3.md) | Dense | 启动器已接入 | [固定策略诊断可运行](rocm-quickstart.md) | 完整 C0–C10 验收待完成 |
| [Qwen3.8-27B](qwen38.md) | Hybrid Dense，文本路径 | 配置准备已接入 | 配置准备已接入 | 严格运行时与 GPU 验收待完成 |
| [Qwen3-30B-A3B](qwen3-moe.md) | MoE，128 专家 / top-8 | 配置准备已接入 | 配置准备已接入 | 严格运行时与 GPU 验收待完成 |

这里的**绝对一致性（bitwise）**指：在声明的模型、后端和环境范围内，
同一权重版本的 rollout 与 training 独立计算所得 logprob 原始字节完全相等。
MoE 还需检查独立路由的专家选择、路由权重原始位和 dispatch/combine 收据。

表格记录本仓库的接入与验收状态，目前没有配方完成 GPU C0–C10 全流程验收。
Qwen3-8B 复用 RL-Kernel 已有集成，每份绑定配方仍须保留自己的实机证据。
另外两个模型通过 `prepare-model` 生成 proxy 配置和候选配方，尚不启动 GPU；
Hybrid 默认 8 层，MoE 默认 4 层并保留 128 专家/top-8。CUDA 与 ROCm 分别验收。

## 工作流与工具

**未适配模型也可以先定位。** 新增[通用首个分歧定位器](diagnostics.md)，
保留两侧原生实现，显式映射语义边界，按依赖图定位，输出可展开的矩阵。
它检查同侧重复稳定性、输入/输出和状态指纹，并可调用原始实现做同输入局部重放。
不要求先注册模型或替换 RL-Kernel 算子；实际引擎仍需接入边界采集。

有 PyTorch 时运行 `rl-consistency diagnostic-demo --output runs/diagnostic-demo-001`，
即可看到“中间不一致、最终又相同”的 CPU 合成案例（预期退出码 1）。
矩阵位于 `runs/diagnostic-demo-001/report/matrix.html`。真实模型通过
`rl-consistency diagnose --matrix <manifest.json> --output <新目录>` 分析采集结果。
诊断可用与模型绝对一致性验收是两个状态，上表的验收范围保持适用。

C0–C10 工作流包含 C5 双路径、C4 结构性能、C6 backward/STE、θ0/θ1/θ2
和完整模型运行前的 recipe review。详见
[gate 定义](../skills/train-inference-consistency/references/gates.md)与
[工作流范围](../skills/train-inference-consistency/references/workflow-scope.md)。
旧 v1 recipe/evidence 需要按 v2 要求新建运行。

目前已经实现：命令行、配置检查、C0–C10 证据验收、逻辑 token 对齐后的原始位比较、依赖失效处理、人工审核记录、HTML 报告，以及调用现有 Qwen3/vime 启动器的桥接代码。

新增了真实产物接入：`import-rlk` 读取已有 `.pt` sidecar，检查 TP 副本、CP 分片、loss mask 和逐 token 位差异；`TraceWriter` 在 worker 内导出带身份信息、预期覆盖检查的 trace，`worker_manifest` 记录已加载模块和共享库的文件指纹。详见 [采集接入说明](capture.md)。API 已可用，尚未自动注入 vLLM/Megatron。

旧 sidecar 没有完整权重版本、稳定样本身份和执行来源，导入后只生成诊断报告，不会自动通过验收。`call_index` 也不能当作 optimizer step。

目前没有完成：GPU 端到端验证、引擎自动采集器、proxy 权重物化、Hybrid/MoE 数值运行时、已认证配置发布。三个模型的六份 CUDA/ROCm 配方都是 candidate。仓库已有的 Qwen3 成果是接入基础，不能替代本项目固定版本后的重新验收。

## 第一次怎么用

1. 安装：`python -m pip install -e .`。
2. 用 `rl-consistency list` 查看候选配置，用 `export-recipe` 导出本地配置。
3. 填入实际模型、权重、数据、源码、运行环境和机器 profile 的身份。用 `check --bound` 检查。
4. 接入真实引擎采集器，生成 `docs/evidence-format.md` 约定的原始数据。
5. `init` 创建一个运行目录，`gate C0` 开始检查；基线 C1 与一致性 C2–C9 独立推进。
6. 查看 `status`，用 `report` 输出 HTML。失败和缺失项会保留。

## 重要规则

- 同一权重版本下，training 必须独立重算。直接复用 rollout logprob 不算一致性证据。
- 一致性与精度分开验证；前向一致不能证明梯度正确。
- 小模型保留原层尺寸、精度与量化方式，完整模型及 checkpoint 恢复仍须验证。
- 原生基线调优后由人审核并冻结，一致性开发不能修改它来缩小性能差距。
- 修改源码、配置、精度规则、容差或输入范围后重新建立运行。不能沿用旧版本的发布结论。
- 同一平台内部训推一致、跨 TP/CP 一致、CUDA/ROCm 之间一致、优化器轨迹一致是不同承诺。

模板中的精度阈值和最少比较数量只用于搭建流程，正式使用前必须按模型、张量及任务审核。review 命令只记录已经发生的人类决定，不能代替真实审核。

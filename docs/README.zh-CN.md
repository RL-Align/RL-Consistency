# RL-Consistency

这是 RL-Kernel 上层的通用 Skill 和验收工具。Skill 规定开发顺序，工具检查证据，底层数值计算继续由 RL-Kernel 承担。

**现在上 ROCm 跑：直接看 [实机运行步骤](rocm-quickstart.md)。** 配好已有环境路径后，
`rocm-plan` 看设置，`rocm-check` 启动真实固定策略诊断并自动生成 JSON/HTML。

C0–C10 已按 vLLM PR #61551 的职责调整为 v2，补了 C5 双路径、C4 结构性能、
C6 backward/STE、θ0/θ1/θ2 和 full-model 前的 recipe review。实现选择继续适配
RL-Kernel/vime/ROCm；Rust Bench 等参考 PR 的专用部分尚未实现，不能声称完整等同。
旧 v1 recipe/evidence 不兼容新 gate 定义，需要新建运行，不能直接改编号。

目前已经实现：命令行、配置检查、C0–C10 证据验收、逻辑 token 对齐后的原始位比较、依赖失效处理、人工审核记录、HTML 报告，以及调用现有 Qwen3/vime 启动器的桥接代码。

新增了真实产物接入：`import-rlk` 读取已有 `.pt` sidecar，检查 TP 副本、CP 分片、loss mask 和逐 token 位差异；`TraceWriter` 在 worker 内导出带身份信息、预期覆盖检查的 trace，`worker_manifest` 记录已加载模块和共享库的文件指纹。详见 [采集接入说明](capture.md)。API 已可用，尚未自动注入 vLLM/Megatron。

旧 sidecar 没有完整权重版本、稳定样本身份和执行来源，导入后只生成诊断报告，不会自动通过验收。`call_index` 也不能当作 optimizer step。

目前没有完成：GPU 端到端验证、引擎自动采集器、自动缩层、新模型运行时、已认证配置发布。CUDA/ROCm 两个配置都是 candidate。仓库已有的 Qwen3 成果是接入基础，不能替代本项目固定版本后的重新验收。

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

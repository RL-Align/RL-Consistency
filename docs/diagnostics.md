# 通用首个分歧定位器

目录：[直接运行](#直接运行) · [接入原生模型](#接入原生模型) ·
[采集协议](#采集协议) · [矩阵和定位规则](#矩阵和定位规则) ·
[局部重放](#局部重放) · [实现边界](#实现边界)

这条诊断路径不要求模型已经注册，也不要求先替换为 RL-Kernel 算子。
保留两侧原生实现，在固定权重与逻辑输入上采集对应边界，找出**依赖图中最早已观测到的分歧**。
若边界内部是融合计算，结果定位到该区域；不能据此宣称已找到内部第一条出错指令。

## 直接运行

安装本项目后，有 PyTorch 的机器可以运行 CPU 示例：

```bash
rl-consistency diagnostic-demo --output runs/diagnostic-demo-001
```

这是故意注入中间层误差的合成模型，**预期退出码为 1**。最后一层把差异消掉，
但报告仍定位到 `block.10.native`；`replay/replay.json` 记录两侧各运行两次的
`local_replay_mismatch`。它检验工具，不是任何 LLM 的一致性成绩。

产物：

| 路径 | 内容 |
|---|---|
| `plan.json`、`inventory.json` | 逻辑依赖图、实际模块清单 |
| `left/`、`right/`、`left_repeat/`、`right_repeat/` | 四次独立采集及二进制张量 |
| `matrix-input.json` | 比较矩阵输入清单 |
| `report/matrix.html`、`report/matrix.json` | 可展开矩阵、机器可读定位结果 |
| `replay/replay.json` | 绑定原始采集的局部重放记录 |

已有采集后，离线分析不需要安装 PyTorch：

```bash
rl-consistency diagnose --matrix runs/diagnostic-demo-001/matrix-input.json \
  --output runs/diagnostic-demo-001/reanalysis
```

退出码：`0` 表示全部计划边界的输出相等且两侧重复观测稳定；`1` 表示观测到分歧；
`2` 表示证据不完整、不可比较或文件无效。`0` 仍只代表这次诊断范围，不能通过 C0–C10。
输出目录必须是新目录，避免覆盖旧证据。

## 接入原生模型

只需要原生框架本身能运行该模型。PyTorch eager 模型可以用模块 hooks；
vLLM、Megatron、编译图与融合算子的内部边界需要在实际 worker 中接入采集 API。
本版本不会自动启动这些引擎，也不会根据模型名字推断语义映射。

1. 调用 `discover_modules(model)` 获取路径、类型、参数尺寸和共享模块关系。
   清单用于发现候选边界，不能证明两个同名或同尺寸模块语义相同。
2. 固定真实 checkpoint、权重版本、canonical token、位置、mask 和状态初值。
   单独比较 rollout / old-logprob、rollout / 实际 grad actor、prefill / decode 等关系。
3. 声明对应边界和依赖图。先覆盖层，再细化失败区域的模块或算子。
   更细计划需要重新采集；工具不会跳过中间层去假定误差单调传播。
4. 为每侧配置显式的模块调用映射与输入/输出选择器。选择器处理 padding、
   batch 位置、全局 token、QKV 拆分和逻辑分片；保留原始 dtype，不进行近似转换。
5. 每侧至少独立采集两次，重置相同状态；再运行 `diagnose`。

PyTorch 接入骨架如下，`model`、`plan`、`canonical_input` 和身份信息必须来自实际工作负载：

```python
from rl_consistency.capture import worker_manifest
from rl_consistency.diagnostics import DiagnosticWriter, ModuleCapture, discover_modules

inventory = discover_modules(model)
manifest = worker_manifest(
    role="training", rank=rank, world_size=world_size,
    module_names=[type(model).__module__], mapped_libraries=True,
)
writer = DiagnosticWriter(
    "runs/native-left", plan, side="old-logprob",
    provenance={"provider": "actual-provider-name", "manifest": manifest},
    execution={"phase": "old-logprob", "mode": "eager", "observer_effects": []},
    max_bytes=64 * 1024 * 1024,
)
with ModuleCapture(
    model, writer,
    {"layers.0": ["sample0.layer0.call0"], "layers.1": ["sample0.layer1.call0"]},
    inputs=lambda module, args, kwargs: {"hidden": args[0]},
    outputs=lambda module, args, kwargs, output: {"hidden": output},
):
    result = model(canonical_input)
trace_path = writer.finish()
```

上述选择器适用于输入输出都是单个 tensor 的相应模块。tuple/dict 输出、kwargs、
融合 QKV、路由索引等要按实际语义写选择器；不支持的结构应显式拒绝。
默认状态快照记录模块参数、buffer、training 和 grad mode，但
`state_complete=False`：它没有自动覆盖 KV cache、路由容量、RNG、通信和全局引擎状态。
只有提供覆盖所有相关状态的 `state(module)` 回调后，才能声明 `state_complete=True`。
worker manifest 应在实际 provider 已加载、完成预热后采集；记录真实实现和运行配置，
不要仅填写源码 checkout 的 commit。相同 worker 的独立进程重复允许 PID 改变，
但代码指纹、主机、rank、路由和其他上下文差异仍会阻止稳定性确认。

同一个共享模块多次调用，绑定一个模块路径到多个计划节点，例如
`{"shared_block": ["layer0.call0", "layer0.call1"]}`。额外调用报错；未调用节点标记缺失。
hooks 返回 `None`，不会替换张量或切断原始 autograd。异常退出会移除 hooks 并阻止发布完成采集。
不要在 selector/state 回调里修改模型或输入。

非模块边界直接调用 `writer.begin(node, inputs, state=..., state_complete=...)`，
执行原生计算，再调用 `writer.end(node, outputs)`。`begin` 必须在计算前执行，
否则原地修改会污染输入快照。独立 producer 分别写自己的目录，禁止共用同一个 writer。

## 采集协议

`diagnostic-plan.v1` 的最小结构（哈希必须由真实文件/规范化数据计算）：

```json
{
  "schema": "diagnostic-plan.v1",
  "identity": {
    "weights_sha256": "<actual checkpoint identity>",
    "weight_version": "theta0",
    "fixture_sha256": "<actual tokens, positions, masks and state-fixture identity>",
    "semantics_sha256": "<actual comparison semantics identity>"
  },
  "nodes": [
    {
      "id": "sample0.layer0.call0",
      "parents": [],
      "kind": "module",
      "scope": {"sample": "sample0", "positions": [0, 1], "layer": "0",
                "boundary": "hidden-output", "call": 0, "shard": "full"}
    },
    {
      "id": "sample0.layer1.call0",
      "parents": ["sample0.layer0.call0"],
      "kind": "opaque",
      "scope": {"sample": "sample0", "positions": [0, 1], "layer": "1",
                "boundary": "fused-block-output", "call": 0, "shard": "full"}
    }
  ]
}
```

`kind` 为 `module`、`operator` 或 `opaque`。不同 token、调用次数、状态阶段和分片
必须分配正确的逻辑身份；比较阶段另记录在每侧 `execution.phase`。
计划应显式列出全部预期边界。不要从实际采集结果反推“预期覆盖”，否则漏采会消失。
重复身份、循环依赖和未知父节点会被拒绝。多 rank 的逻辑重组、重复副本检查和通信因果边
仍由引擎 adapter 提供；本工具不从墙钟时间推断跨 rank 的先后关系。

每个 `diagnostic-trace.v1` 保存唯一采集 ID、计划哈希、权重/输入身份、实际
provider/worker manifest、执行上下文、观测记录和缺失清单。每条记录包含：

| 字段 | 说明 |
|---|---|
| `node` | 计划中的语义调用身份 |
| `inputs` / `outputs` | 命名 tensor ports，含 dtype、shape、SHA256、相对 blob 路径 |
| `state` | 计算前相关状态的指纹、键清单和完整性声明 |

张量存储为小端原始字节，支持 FP16/BF16/FP32/FP64、int64/int32/int16/int8/uint8/bool。
先验证 blob 内容哈希，再比较实际字节；`+0/-0` 不相等，整数路由不会转为浮点。
NaN/Inf、dtype/shape 不匹配、缺失 port、越界路径不能变成通过结果。
空专家分片可以作为部分 port 存在，但全空输出不能作为有效观测。

张量按内容寻址存储。默认每次采集输入输出总预算 64 MiB；超限要求缩小计划或显式提高预算。
状态张量只保留指纹，逐项临时复制到 CPU。大型模型应先选择少量样本和边界，
不应默认捕获所有层、所有 token、所有权重。该实现适合有界诊断，尚非异步大规模追踪系统。

## 矩阵和定位规则

`diagnostic-matrix.v1` 中路径相对于清单目录，必须位于该目录内：

```json
{
  "schema": "diagnostic-matrix.v1",
  "plan": "plan.json",
  "cases": [
    {"id": "rollout-vs-old-logprob", "left": "rollout/trace.json",
     "right": "old-logprob/trace.json", "left_repeat": "rollout-repeat/trace.json",
     "right_repeat": "old-logprob-repeat/trace.json"},
    {"id": "rollout-vs-actor", "left": "rollout/trace.json",
     "right": "actor/trace.json", "left_repeat": "rollout-repeat/trace.json",
     "right_repeat": "actor-repeat/trace.json"}
  ]
}
```

矩阵行是语义边界，列是独立比较场景。cell 展开后包含输入/输出字节差、状态比较、
重复稳定性、缺失祖先和逻辑 token 范围；列标题可展开来源与覆盖信息。
一个 tensor 包含多个 token 时，`flat_element` 是展平索引，结合 shape 和显式 selector 映射回逻辑 token。
源 trace 必须和 tensor 目录一同保留；报告不复制所有 blob，也不把声明变成可信认证。

| 结果 | 含义及下一步 |
|---|---|
| `earliest_observed` | 此节点输出不同，已采集祖先未出现输出差异；细化此边界 |
| `upstream_unresolved` | 上游有缺失/不可比边界，先补齐，不能宣称首因 |
| `same_input_state_candidate` | 同输入、完整状态指纹相同而输出不同；进入局部重放 |
| `input_or_state_changed` | 输入或状态已经不同；检查上游，也可重放排查第二个问题 |
| `unresolved_context` | 状态不完整或输入无法对齐；先补充上下文 |
| `unstable` | 同侧重复输出不同；先处理该侧可重复性 |
| `equal_observed` | 全部计划输出相同、两侧重复稳定；仅说明当前观测范围 |

`frontier` 可以有多个并行节点。没有“第一个分歧之后全部跳过”的逻辑，也不依赖误差单调：
早期差异可能收敛，后面还可能出现新的同输入分歧。
`status` 比较输出；输入/状态不同但输出相同的行仍保留其上下文差异。
重复稳定性与输出字节差独立显示；缺少 repeat 时相同输出的总 verdict 为 `inconclusive`。

## 局部重放

Python API `replay_boundary(plan, left_trace, right_trace, node, inputs=...,
left_endpoint=..., right_endpoint=..., output=...)` 接收两个 `ReplayEndpoint`：

- `call(inputs)`：执行各自实际原生实现，返回命名 tensor outputs。
- `reset()`：每次调用前恢复权重、cache、RNG、计数器等状态。
- `state()`：计算前实际相关状态，`state_complete` 明确是否覆盖完整。
- `provenance`、`execution`：原始实现来源与实际重放上下文。provider manifest 必须与对应原始采集一致。

输入必须与原始左侧边界的 dtype、shape、字节完全相同。helper 为每次调用复制输入，
两侧各执行两次；只有输入与完整状态相同且各侧重复稳定，才输出
`local_replay_mismatch` 或 `local_replay_equal`。回调仅来自调用者的 Python 代码，
证据 JSON 不会加载模块、执行 shell 命令或运行任意代码。

实现示例见 `src/rl_consistency/diagnostics/demo.py`。实际引擎要保留原始 batch 构成、
stride、精度、量化、调度、kernel route、通信和 cache 语义。helper 只能检查逻辑字节与
已采集状态，不能自动恢复这些隐式上下文。`local_replay_mismatch` 证明的是所记录重放环境下的
局部差异，仍需回到原始执行路径验证原因和修复效果。

## 实现边界

同步 CPU 快照会改变时序并可能干扰编译/CUDA Graph；每份 trace 自动记录
`synchronous_tensor_snapshot`。如果需要 eager、关闭 fusion 或改变 Graph，必须额外记录
`execution.observer_effects`，不能将诊断路径结果当作未插桩生产路径的证据。
工具不会自动切换执行方式。

定位结果独立于 `trace.v1` 和 C0–C10 的正式证据。模型未适配时可以先诊断原生路径，
但要发布“该模型绝对一致性已支持”，仍须适配严格运行时并完成相应实机验收。

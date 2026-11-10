# NPU Inductor 可重放 RNG：PR 评审串讲稿

日期：2026-11-07。建议讲解时间：约 10 分钟。

对应 [PR #47685：support graphsafe RNG state replay in NPU Inductor](https://gitcode.com/Ascend/pytorch/pull/47685)。依据已核对的提交 `994df68554a27005f3ed924078d83854def8d630`。

使用方式：每节的“可以直接讲”是口头稿；“自己理解”帮助理解，不必逐字念。代码摘录用于解释，不是独立运行脚本。本次只整理材料，没有重新执行 NPU 测试，也没有修改 PR。

## 一、开场：我们到底要做什么

### 可以直接讲

> 这个 PR 要解决的是：NPU 在 Inductor 编译训练中，使用 checkpoint 和 dropout 时，随机状态如何正确保存、传递，并在反向重算时使用。
>
> 我们没有重新实现 dropout，也没有重新设计随机算法。上游已有一套状态管理流程，我们补的是 NPU 与这套流程之间的适配。
>
> 能力位于 NPU Inductor 公共编译链路，重点用 triton_experimental 后端验证。本次不涉及 ACLGraph 设备图录制与重放。

### 自己理解

先记住一句话：**让反向重算找回原来那次前向的随机状态。**

不是“让训练不随机”，也不是“让每一步 dropout 都产生同一个结果”。

## 二、为什么训练会需要它

### 可以直接讲

> dropout 会随机把一部分输入置零，mask 就是记录哪些元素保留的标记。
>
> checkpoint 为节省内存，不保存部分前向中间结果，等到 backward 时再重新计算。如果原前向和重算使用了不同的 dropout mask，反向算出的梯度就不再对应原来的前向。
>
> 所以我们需要保存随机状态，让重算能够找回原来的随机数，而不是临时再抽一组。

### 自己理解：一个小例子

设丢弃概率为 0.5，输入是 `[1, 2, 3, 4]`：

```text
原前向 mask： [1, 0, 1, 0]
前向输出：     [2, 0, 6, 0]
loss：         输出求和
正确输入梯度： [2, 0, 2, 0]

如果反向重算换了 mask： [0, 1, 0, 1]
就会得到另一套梯度，无法对应原前向。
```

这里保留元素乘 2，是因为 dropout 会按 `1 / (1-p)` 做缩放。这个例子只解释直接 dropout 后求和的情况，不是任意复杂网络的梯度公式。

## 三、三个名词怎么解释

| 名词 | 通俗理解 |
|---|---|
| seed | 决定使用哪一条随机序列 |
| offset | 标记序列消费到哪个位置；实际单位和对齐由实现决定 |
| generator state | 可以保存、切换的随机生成器状态；不能把所有设备的布局都简单假设成两个整数 |

### 可以直接讲

> 可以把随机序列当成一本数字书。seed 决定选哪本书，offset 决定从哪里读，state 相当于保存读书位置的记录。
>
> 正常训练应该继续往后读；需要重算某一次前向时，才找回那一次对应的记录。

**提醒：**同一个可变 state 连续被使用，位置可以正常前进。“可重放”要求恢复相同初始状态后，重复相同执行序列能够复现，不要求不恢复状态也每次都相同。

## 四、原来链路断在哪里

### 可以直接讲

> AOTAutograd 会把随机状态安排进前向与反向重算流程。进入 Inductor 后，编译器需要把 GeneratorState 这样的状态描述，转换成真正能传给算子的参数。
>
> 我们实际复现过的明确断点就在这里：上游代码假设这个状态来自 CUDA，遇到 NPU 会触发设备断言。相关 functionalized RNG 路径也需要补上 NPU 的 seed、offset 接口和参数顺序适配。
>
> 所以不是 NPU 的 dropout 本身完全不能跑，而是特定编译训练组合里的状态管理链路接不上。

```text
torch.compile
  -> AOTAutograd 处理 checkpoint 随机重算
  -> graphsafe_run_with_rng_state(..., GeneratorState(npu))
  -> Inductor 构造外部 kernel 调用参数
  -> ExternKernel.process_kernel 处理 GeneratorState
  -> 上游 CUDA-only 假设：NPU 在此被拒绝
```

断点依据：此前目标环境的 `torch/_inductor/ir.py::ExternKernel.process_kernel`；本 PR 在 PTA 层给它安装兼容 wrapper，没有直接修改该上游文件。

上图是职责链路，不表示所有随机算子都依次经过 seed/offset 和 graphsafe state 两种表达。

## 五、第一个文件：把 NPU 随机状态接口接上

文件：[torch_npu/utils/_inductor.py](D:/code/pytorch-rng58-pr/torch_npu/utils/_inductor.py:102)。这是本次新增适配最多的文件。

### 5.1 登记 NPU 的能力

函数：`patch_register_graphsafe_run_with_rng_state_op`。

核心代码：

```python
register_graphsafe_rng_device_type("npu")
```

**可以直接讲：**

> 第一步先告诉 AOTAutograd，NPU 可以参与 graphsafe RNG 状态管理。这里是设备能力注册，不是本次重新实现整个 graphsafe 算子执行逻辑。

**自己理解：**先让框架认识这个设备，才有后面的状态处理。这个名字中的 graphsafe 不表示启用了 ACLGraph。

### 5.2 读取、恢复 NPU 的 seed 和 offset

函数：`patch_rng_state_helper_for_npu`。

核心读取代码：

```python
seed = torch.tensor(torch.npu.initial_seed())
offset = torch.tensor(torch.npu._get_rng_state_offset())
```

恢复时，代码把 seed、offset 转成状态字节并调用 `torch.npu.set_rng_state(new_state)`；更新 offset 时使用 `_set_rng_state_offset`，并做 4 对齐。

**可以直接讲：**

> 第二步是让框架读写 NPU 自己的随机状态。上游 helper 的名字和默认实现偏 CUDA，我们在 PTA 层补上 NPU 接口，使它能记录和恢复 NPU 的消费位置。

**自己理解：**不是伪造随机数，而是把读取状态的接口接到正确设备。4 对齐是当前代码采用的规则，不表示每个随机算子都只消费 4 个数。

**评审注意：**当前判断条件是 NPU 是否可用，而不是当前编译任务的显式设备。这一点需要评估同进程其他设备路径的隔离，不能直接承诺完全没有影响。

### 5.3 允许 NPU 通过相关设备检查

函数：`patch_functionalized_rng_decomp_for_npu`。

```python
if device.type == "npu":
    return
return original_throw_on_non_cuda(device)
```

**可以直接讲：**

> 有了读写接口，还需要让 NPU 通过相关分解流程的设备检查。我们对 NPU 放行，其余设备继续交给原来的检查逻辑。

**自己理解：**这是准入检查的适配，不等于只删除断言；它必须和实际的 NPU 状态接口配套。

### 5.4 把编译和运行参数排成相同顺序

函数：`patch_functionalized_rng_wrapper_for_npu_graphsafe`。

编译前的核心代码：

```python
insert_at = len(flat_args) - fw_metadata.num_graphsafe_rng_states
flat_args[insert_at:insert_at] = [seed, offset]
```

运行时也做对应重排，使布局匹配。示意如下：

```text
普通输入 | seed | offset | graphsafe states
```

**可以直接讲：**

> seed、offset 和 generator state 同时存在时，编译阶段和真正执行阶段必须采用同一套参数顺序。否则状态虽然传进去了，位置却可能错了。
>
> 这段代码只在相应 NPU graphsafe metadata 和 functionalized RNG 条件命中时调整，其他情况委托原 wrapper。

**自己理解：**像函数形参是“姓名、年龄”，实参就不能传成“年龄、姓名”。这次处理的是两类状态参数同时存在的组合，不是每个图都重排。

## 六、第二个文件：在进入 Inductor 时安装适配

文件：[torch_npu/utils/_dynamo.py](D:/code/pytorch-rng58-pr/torch_npu/utils/_dynamo.py:666)。函数：`_lazy_inductor_setup`。

新增的核心调用：

```python
from torch_npu.utils._inductor import (
    patch_functionalized_rng_wrapper_for_npu_graphsafe,
)
patch_functionalized_rng_wrapper_for_npu_graphsafe()
```

### 可以直接讲

> 定义适配函数还不够，必须在合适的时候真正安装。我们把涉及 AOTAutograd runtime wrapper 的适配放到 Inductor 的延迟初始化阶段，进入这条编译链路时再安装，避免过早加载相关模块带来的初始化依赖问题。

### 自己理解

这一文件负责安装时机，不负责生成随机数。它调用的是上面“参数顺序”的适配函数，不能把这一处说成所有 RNG patch 都只在这里安装。

## 七、第三个文件：打通 GeneratorState 的编译断点

文件：[torch_npu/_inductor/_npu_meta_registration.py](D:/code/pytorch-rng58-pr/torch_npu/_inductor/_npu_meta_registration.py:227)。函数：`_npu_fix_generator_state_fallback`。

### 可以直接讲：为什么需要它

> 这是针对实际阻塞点的兼容处理。Inductor 已经拿到了 NPU 的 GeneratorState 描述，但上游参数构造代码仍按 CUDA generator 获取状态。
>
> 我们在 PTA 层包一层 wrapper，为 NPU 状态提供适配，再复用上游其余参数和依赖处理逻辑，避免复制整段大函数。

### 对照代码：实际做了四步

**第一步，识别 NPU GeneratorState，并取得 NPU generator 的克隆状态：**

```python
get_default_generator(device_index).clone_state()
```

这里选用的是 NPU generator；不是把 NPU 的真实随机计算切换到 CUDA。

**第二步，保留原状态名字，临时生成上游可接受的设备标记：**

```python
ir.GeneratorState(name=x.name, device=torch.device("cuda", device_index))
```

这是当前临时兼容方案的一部分，不能隐瞒成“代码已经完全设备无关”。

**第三步，在调用期间，把对应的默认 generator 入口临时替换成 NPU clone：**

```python
torch.cuda.default_generators = tuple(default_generators)
```

**第四步，调用原上游函数，并在结束后恢复入口：**

```python
try:
    return orig(cls, kernel, *new_args, **new_kwargs)
finally:
    torch.cuda.default_generators = old_default_generators
```

### 可以直接讲：设计考虑与风险

> 当前方案的考虑是，先在 PTA 侧接通必要链路，保留上游已有的参数建模，不直接修改上游源码文件。
>
> 但我要明确，这里临时修改了进程级共享属性。finally 可以保证异常退出时执行恢复，不代表已经解决所有并发访问问题。这个兼容方式及其隔离性，是希望专家重点评审的内容。

### 自己理解

“没有修改上游源码文件”与“没有改变上游运行行为”是两回事：这里确实用 wrapper 替换了运行时方法。没有 NPU GeneratorState 的调用会委托原函数，但这并不足以自动证明所有交错调用都安全。

这个适配发生在编译期参数构造，不是每次 dropout 执行都去修改这张默认 generator 表。

## 八、第四个文件：测试怎么证明功能

文件：[test_graphsafe_rng_replay.py](D:/code/pytorch-rng58-pr/test/_inductor/triton_experimental/test_graphsafe_rng_replay.py:31)。当前新增四项测试。

| 用例 | 具体检查 | 可以证明的范围 |
|---|---|---|
| `test_graphsafe_run_with_rng_state_replays_and_restores_npu_state` | 指定 state 连续执行两次，与参考序列的第一段、第二段精确比较；再检查外部随机结果 | 指定状态正常消费，外部随机序列得到恢复 |
| `test_philox_seed_offset_helpers_are_npu_aware` | patch 标记、NPU 设备放行、offset helper 对两个 shape 的返回值 | 基础接入检查；不是完整状态读写和每个 kernel 消费量验收 |
| `test_functionalized_seed_offset_replays_in_triton_experimental` | functionalized RNG 开启，fullgraph 编译两个随机算子；同 seed 结果一致，不同序列作差异检查 | 基础编译随机序列可重现，没有被简单固定死 |
| `test_checkpoint_dropout_preserves_rng_state_in_triton_experimental` | 两个 shape；前向输出推导的 mask 与梯度推导的 mask 比较；同 seed 重跑 output/grad 精确比较 | 目标 checkpoint/dropout 训练场景的状态与梯度对应 |

### 可以直接讲

> 测试不是只看程序有没有跑完。我们分别检查状态执行、基础接口、编译随机序列，以及最关键的 checkpoint/dropout 前向和梯度是否对应。
>
> 简单 dropout 后求和时，输入梯度就是 mask 除以保留概率，所以可以直接检查前向和反向使用的 mask 是否一致。

### 自己理解：不要扩大测试结论

- 当前 PR 描述记录四项通过，本材料没有重新跑测试。
- checkpoint 用例当前没有设置 fullgraph=True，不能据此宣称该用例已经证明零 graph break。
- seed/offset 用例建议补状态读写、恢复行为断言；现有基础检查不等于完整覆盖所有 offset 边界。
- 普通编译训练测试不是 ACLGraph 验收；本次不要求后者。

## 九、为什么改公共适配层，而不是只改 TE

### 可以直接讲

> TE 是我们重点验证的后端，但这个问题不只发生在 TE 生成 Triton 代码的时候。随机状态的组织由 AOTAutograd 负责，GeneratorState 参数构造由公共 Inductor 负责。
>
> 因此适配需要放到 PTA 对这些公共流程的兼容层。只在 TE lowering 中跳过状态包装，可能让图能执行，却丢掉原前向与重算之间的随机状态配对，不能算正确修复。

### 自己理解

可以把层次理解为：AOTAutograd 安排“前后向需要带哪些状态”，Inductor 决定“如何构造调用”，TE 处理实际算子图的编译。这次主要修前两者与 NPU 的衔接。

公共位置意味着值得复用，不意味着所有后端已经验证；当前收益通过 TE 用例检查。

## 十、专家可能追问的四个问题

### 1. 为什么代码里还有 CUDA 字样

> 上游存在 CUDA 专用接口，当前 PTA 兼容实现借用了这些入口。真实 generator 来自 NPU，没有在 CUDA 执行随机算子。但它仍是临时适配，不是最终设备无关方案。

### 2. 能保证完全不影响其他设备吗

> 目前不能做绝对承诺。部分 wrapper 按 NPU metadata 判断，其他路径委托原实现；但 helper 按 NPU 可用性选择，以及临时共享属性替换，仍需审查隔离性。我们希望评审明确这一点，而不是用四项功能测试替代风险判断。

### 3. 这是给 dropout 写原生 Triton kernel 吗

> 不是。我们接通的是状态协议，随机算子可以保留必要的算子级 fallback。只要状态正确、原前向与重算一致，就能满足这次功能范围；不是把整段训练退回 eager 绕过编译。

### 4. 上游以后解耦后怎么办

> 如果上游提供设备无关的状态 helper 和 GeneratorState 获取扩展点，PTA 应改为正式注册 NPU 实现，删除临时 wrapper、设备标记和共享入口替换。保留回归测试，验证迁移后语义不变。这是后续收敛方向，不是当前已经存在的接口承诺。

## 十一、最后一分钟总结

### 可以直接照着读

> 总结一下，这个 PR 的目标是补齐 NPU Inductor 编译训练中的 RNG 状态接入，典型场景是 checkpoint 加 dropout。
>
> 四个文件分别负责：接通 NPU 随机状态接口、在正确时机安装适配、跨过 GeneratorState 的 CUDA 假设，以及增加四项回归测试。
>
> 功能上我们关心三件事：恢复状态后随机序列能复现，checkpoint 原前向与重算的 mask 能配对，外部随机序列不被扰乱。
>
> 方案上本次采用 PTA 临时兼容实现，重点请专家评审公共 helper 和 GeneratorState wrapper 的边界，特别是共享属性替换的隔离性。不把它说成新的随机算法，也不扩展成 ACLGraph 能力。

## 附：最短复述版

> checkpoint 反向时会重算前向，dropout 必须找回原来的随机状态，否则梯度不对应。上游已有这套机制，但部分地方只认 CUDA。我们的 PR 在 PTA 侧补 NPU 的注册、seed/offset、参数顺序和 GeneratorState 适配，再用四项测试检查能否重放及 mask/grad 是否一致。它解决的是 Inductor 编译训练状态链路，不是重写 dropout，也不涉及 ACLGraph。

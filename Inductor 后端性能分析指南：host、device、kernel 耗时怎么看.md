# Inductor 后端性能分析指南：host、device、kernel 耗时怎么看

在对 PyTorch Inductor 后端做模型性能测试时，经常会看到几类时间：

- host 侧耗时
- device 侧耗时
- 算子 kernel 耗时
- 端到端 wall time

这些指标不是同一个维度，不能简单混着看。理解它们之间的关系，才能判断模型到底是：

- CPU/host 瓶颈
- GPU/device 瓶颈
- kernel launch / 调度瓶颈
- 同步开销瓶颈
- 单个 kernel 本身效率低

---

## 1. 基本概念

### 1.1 host 侧耗时

host 一般指 CPU 侧，也就是 PyTorch Python/C++ runtime 所做的事情。

host 侧耗时可能包括：

1. Python 代码执行
2. PyTorch dispatcher 调度
3. Inductor 生成的 host 代码
4. tensor 元信息计算
5. 算子 launch 准备
6. CUDA API 调用，例如 `cudaLaunchKernel`
7. CPU 端数据预处理
8. CPU/GPU 同步
9. autograd 相关逻辑
10. Inductor 编译、autotune 等一次性开销

简单说：

> host time 是 CPU 为了推动 GPU 执行所花的时间，或者 CPU 自己执行逻辑的时间。

---

### 1.2 device 侧耗时

device 一般指 GPU 侧，也就是真正在 GPU 上执行 kernel 的时间。

device 侧耗时包括：

1. GEMM kernel
2. elementwise kernel
3. reduce kernel
4. attention kernel
5. Triton kernel
6. CUDA memcpy
7. memset
8. 多个 stream 上的 GPU 任务

简单说：

> device time 是 GPU 上真正执行计算、拷贝等任务的时间。

---

### 1.3 kernel 耗时

kernel 是 GPU 上执行的最小计算单元之一。

Inductor 可能会把多个算子融合成一个或几个 Triton kernel，例如：

```text
triton_poi_fused_add_mul_relu_0
triton_red_fused__to_copy_1
triton_per_fused_softmax_2
```

kernel time 通常指的是这些 GPU kernel 从开始执行到结束的时间。

在 profiler 里你可能会看到：

```text
Name                      CUDA total
triton_poi_fused_add_0      1.20ms
triton_red_fused_norm_1     3.50ms
ampere_sgemm_64x32_nn       8.80ms
```

这就是 kernel 维度的耗时。

---

## 2. host time、device time、wall time 的关系

假设你执行一次 forward：

```python
out = model(x)
torch.cuda.synchronize()
```

你测到的总时间叫 wall time，也就是真实流逝时间。

它可以粗略理解为：

```text
wall time =
    host 准备和提交任务的时间
  + GPU 执行任务的时间
  + CPU/GPU 等待、同步、调度空隙
  + launch overhead
  + 其他系统开销
```

但要注意：

> host time 和 device time 可能是重叠的，不能简单相加。

例如：

```text
CPU:  launch kernel A -> launch kernel B -> launch kernel C
GPU:              [kernel A][kernel B][kernel C]
```

CPU 在 launch kernel 的同时，GPU 可能已经在执行前面的 kernel。

所以：

```text
host time + device time != wall time
```

---

## 3. 为什么要区分 host 和 device？

因为优化方向完全不同。

---

### 3.1 device-bound

device-bound 表示 GPU 很忙，主要时间花在 kernel 执行上。

典型现象：

- GPU kernel 总耗时接近端到端时间
- profiler 里 GPU 几乎一直有任务
- top kernel 很明显
- CPU launch 时间相对较短
- 提高 batch size 或模型变大后耗时明显增加
- GPU utilization 高

优化方向：

1. 减少计算量
2. 做算子融合
3. 提高 kernel 效率
4. 用更好的 Triton config
5. 使用 `max-autotune`
6. 用更高效的 attention
7. 减少低效 elementwise
8. 调整 block/grid
9. 优化 memory bandwidth
10. 使用 cudnn、cublas、flash attention 等高性能实现

---

### 3.2 host-bound

host-bound 表示 GPU 经常闲着，CPU 来不及提交任务。

典型现象：

- GPU trace 里有很多空隙
- kernel 很碎、很多、很短
- CPU op 或 CUDA API 时间很长
- launch overhead 占比高
- 小 batch 下性能很差
- Python 代码复杂
- 动态 shape 导致频繁重编译
- Inductor graph 没有捕获完整
- 没有用 CUDA Graphs

优化方向：

1. 减少 kernel 数量
2. 用 Inductor fusion
3. 用 CUDA Graphs
4. 减少 Python 开销
5. 固定 shape
6. 避免动态控制流
7. 减少 CPU/GPU sync
8. 使用 `torch.compile(mode="reduce-overhead")`
9. 减少 `.item()`、`print(tensor)`、`tensor.cpu()` 等强制同步
10. 把 CPU 预处理移出主循环

---

### 3.3 launch-bound

launch-bound 是 host-bound 的一种常见形式。

典型现象：

- 每个 kernel 只跑几微秒
- 但有几百上千个 kernel
- GPU 上出现大量空隙
- CUDA API `cudaLaunchKernel` 很多
- 小 batch、小模型特别明显

优化方向：

1. 算子融合
2. CUDA Graphs
3. 减少碎片算子
4. 使用 Inductor 生成更大的 fused kernel
5. 避免过度切分图
6. 使用 static shape

---

## 4. Inductor 场景下应该重点看什么？

Inductor 是 PyTorch 的编译后端，它会把 eager 模式下的算子图转换成分组后的 kernel，很多是 Triton kernel。

使用 Inductor 测性能时，通常要看这几个层面：

1. 端到端 latency
2. device time / kernel total time
3. host time / CPU op time
4. top kernel
5. GPU 空隙
6. 是否存在 graph break
7. 是否存在 recompile
8. 是否 autotune 充分
9. 是否使用 CUDA Graphs
10. 是否存在大量 dtype cast 或 memcpy

---

## 5. 端到端 latency 是最重要的指标

无论 profiler 里 host/device/kernel 看起来如何，最终性能首先要看端到端 latency。

一个基本测试方式：

```python
import time
import torch

model = torch.compile(model)

x = torch.randn(batch_size, seq_len, hidden, device="cuda")

# warmup
for _ in range(10):
    model(x)
torch.cuda.synchronize()

start = time.time()
for _ in range(100):
    model(x)
torch.cuda.synchronize()
end = time.time()

print("avg latency:", (end - start) / 100 * 1000, "ms")
```

注意几个点：

1. 必须 warmup
2. 必须 `torch.cuda.synchronize()`
3. 第一次运行包含编译和 autotune，不能算入稳定性能
4. 如果用 `torch.compile`，第一次会特别慢
5. 如果 shape 变化，可能触发重新编译

也可以使用 CUDA Event：

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
model(x)
end.record()

torch.cuda.synchronize()
print(start.elapsed_time(end), "ms")
```

---

## 6. 使用 PyTorch Profiler 看 host/device/kernel

### 6.1 基本 profiling 示例

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for _ in range(10):
        model(x)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
```

重点看：

```text
Name
CPU total
CUDA total
Self CUDA
Self CUDA %
```

其中：

- `CPU total`：CPU 侧总耗时
- `Self CPU`：当前节点自身 CPU 耗时
- `CUDA total`：这个算子关联的所有 GPU 时间
- `Self CUDA`：当前算子自身 GPU 时间

如果想知道哪些算子最占 GPU：

```python
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
```

如果想知道哪些 CPU 逻辑最耗时：

```python
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
```

---

### 6.2 导出 trace

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    model(x)
    torch.cuda.synchronize()

prof.export_chrome_trace("trace.json")
```

`trace.json` 可以用 Chrome 打开：

```text
chrome://tracing
```

也可以用 TensorBoard 查看。

---

### 6.3 使用 TensorBoard 查看 PyTorch Profiler

示例：

```python
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=2, active=5, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./tb_trace"),
    record_shapes=True,
    with_stack=True,
) as prof:
    for step in range(8):
        model(x)
        prof.step()
        torch.cuda.synchronize()
```

然后启动：

```bash
tensorboard --logdir=./tb_trace
```

TensorBoard 里可以看到：

- CPU/GPU summary
- kernel 耗时排名
- GPU 时间线
- CPU op
- CUDA API
- memory
- trace 时间线

---

## 7. 怎么判断性能瓶颈？

---

### 7.1 情况一：GPU kernel 时间很长，GPU 很满

现象：

```text
wall time ≈ device kernel total time
GPU trace 几乎连续执行
top kernel 很集中
```

结论：

```text
device-bound
```

优化重点：

- 看 top kernel
- 看是计算瓶颈还是访存瓶颈
- 看是否能 fuse
- 看是否能用更优 Triton config
- 看是否能换更优库实现

---

### 7.2 情况二：GPU kernel 很短，但数量很多，GPU 有大量空隙

现象：

```text
每个 kernel 只有几 us
kernel 数量很多
trace 上 GPU 断断续续
host 上大量 cudaLaunchKernel
```

结论：

```text
launch-bound / host-bound
```

优化重点：

- Inductor fusion
- CUDA Graphs
- 减少碎片算子
- 固定 shape
- 减少 Python 控制流
- `mode="reduce-overhead"`

例如：

```python
model = torch.compile(model, mode="reduce-overhead")
```

`reduce-overhead` 通常会尝试启用 CUDA Graphs，对小 kernel、多 launch 的模型有帮助。

但 CUDA Graphs 对输入 shape、地址稳定性有要求，动态 shape 可能不适用。

---

### 7.3 情况三：kernel 时间不长，但端到端很慢

现象：

```text
CUDA kernel total 很小
wall time 很大
profiler 里 CPU op 很长
有很多 cudaStreamSynchronize
```

结论：

```text
CPU/host 或同步瓶颈
```

常见原因：

1. 数据在 CPU 上
2. 每步都有 `.item()`
3. loss 打印导致同步
4. shape 推断复杂
5. Python 开销大
6. Inductor 编译没有完整捕获图
7. 有 graph break

优化：

- 去掉不必要同步
- 数据提前搬到 GPU
- 使用 static shape
- 减少 Python 分支
- 检查 graph break
- 看 Inductor 日志

---

### 7.4 情况四：第一次很慢，后面正常

这是 `torch.compile` 常见现象。

第一次包含：

1. TorchDynamo trace
2. Inductor 编译
3. Triton codegen
4. autotune
5. kernel cache 建立

所以测性能时一定要：

```text
先 warmup，再统计稳定阶段。
```

例如：

```python
for _ in range(10):
    model(x)
torch.cuda.synchronize()

# 开始计时
```

如果你要测 Inductor autotune 后的性能，更应该 warmup 足够多次。

---

## 8. 怎么理解 Inductor 生成的 Triton kernel 名字？

Inductor 生成的 Triton kernel 名字通常类似：

```text
triton_poi_fused_add_mul_relu_0
triton_red_fused_layer_norm_1
triton_per_fused_softmax_2
triton_poi_fused__to_copy_3
```

大致可以这样理解：

| 名字片段 | 含义 |
|---|---|
| `triton` | Inductor 生成的 Triton kernel |
| `poi` | pointwise，逐元素类 |
| `red` | reduce |
| `per` | persistent / reduction 相关，不同版本语义可能有差异 |
| `fused_add_mul_relu` | 融合了 add/mul/relu |
| `_to_copy` | dtype/device copy 或 cast |
| 后面的数字 | kernel id |

例如：

```text
triton_poi_fused_add_mul_relu_0
```

说明这个 kernel 融合了：

```text
add + mul + relu
```

这类融合可以减少 kernel launch 和全局内存读写。

如果你看到很多：

```text
triton_poi_fused__to_copy
```

说明可能存在大量 dtype 转换，比如：

```text
fp32 -> fp16
bf16 -> fp32
```

这时要检查：

1. AMP 配置
2. dtype 是否一致
3. 是否有不必要的 cast
4. LayerNorm/Softmax 是否因为精度问题频繁升降精度
5. 输入、参数、buffer dtype 是否统一

---

## 9. 算子 kernel 的性能怎么看？

看 kernel 性能不能只看 duration。

一个 kernel 耗时高，可能是因为：

1. 计算量本来就大
2. 访存量大
3. 没打满 GPU
4. launch 配置差
5. occupancy 低
6. shared memory bank conflict
7. memory bandwidth 瓶颈
8. register spill
9. 不必要的类型转换
10. kernel 没有融合，读写中间结果太多

---

### 9.1 先看 kernel duration

例如：

```text
triton_red_fused_layer_norm_gelu_0    4.2ms
triton_poi_fused_add_mul_1            0.6ms
cutlass_tensorop_sgemm_128x64_nn      9.8ms
```

你要关注：

- 哪些 kernel 排 top
- top kernel 占总时间比例
- 是否有明显异常的小 kernel 很多
- 是否有大量 copy / cast kernel
- 是否有本应被融合但没有融合的 kernel

Inductor 的优势之一就是 fuse 很多 elementwise/reduce 算子，所以如果你看到大量：

```text
triton_poi_fused__to_copy
triton_poi_fused_add
triton_poi_fused_mul
triton_poi_fused_relu
```

可能说明融合不够理想，或者 eager 写法导致不能融合。

---

### 9.2 看 kernel 是计算密集还是访存密集

#### 计算密集型 kernel

比如：

- matmul
- convolution
- attention 中的 QK^T
- projection
- FFN 中的 GEMM

这类 kernel 通常看：

```text
TFLOPS / SOL compute
```

如果 GPU 算力没有打满，可能原因：

1. tile size 不好
2. autotune 没选好
3. shape 不友好
4. 非标准 stride
5. 频繁 transpose
6. dtype 不合适
7. 没有使用 Tensor Core
8. kernel 实现不佳

---

#### 访存密集型 kernel

比如：

- layernorm
- softmax
- elementwise
- reduce
- transpose
- copy
- cast
- embedding lookup
- activation

这类 kernel 通常看：

```text
memory bandwidth
DRAM throughput
L2 hit rate
```

如果 memory bandwidth 没打满，可能原因：

1. grid/block 配置差
2. memory access 不 coalesce
3. shared memory conflict
4. 小 tensor 导致 GPU 吃不饱
5. kernel launch overhead 太高
6. 频繁读写全局内存
7. 没融合，导致中间结果反复读写

---

### 9.3 看 GPU SOL / utilization

SOL 是 speed of light，表示理论上硬件能力的利用率。

常见指标：

```text
Compute SOL
Memory SOL
SM utilization
Occupancy
DRAM throughput
L2 throughput
```

例如 Nsight Compute 里可能看到：

```text
Compute Throughput: 82%
Memory Throughput: 35%
```

这可能说明 kernel 是 compute-bound。

反过来：

```text
Compute Throughput: 12%
Memory Throughput: 88%
```

说明大概率是 memory-bound。

如果两个都很低：

```text
Compute Throughput: 10%
Memory Throughput: 15%
```

说明 kernel 没跑满，可能是：

1. occupancy 太低
2. grid 太小
3. launch 配置差
4. kernel 太短
5. 有 stall
6. 依赖 host
7. 同步太多

---

### 9.4 看 occupancy

occupancy 表示 GPU SM 上 active warp 的占用情况。

低 occupancy 不一定一定差，但通常会影响延迟隐藏。

低 occupancy 常见原因：

1. register 使用太多
2. shared memory 使用太多
3. block size 太小
4. grid size 太小
5. kernel 并行度不足

对于 memory-bound kernel，通常需要足够 occupancy 来隐藏访存延迟。

对于 compute-bound kernel，有时 occupancy 不是唯一关键，指令流水和 Tensor Core 利用率更重要。

---

## 10. Inductor 性能测试常见坑

---

### 10.1 把第一次编译时间算进性能

错误方式：

```python
model = torch.compile(model)

start = time.time()
model(x)
torch.cuda.synchronize()
end = time.time()
```

这会把编译时间算进去。

正确方式：

```python
model = torch.compile(model)

for _ in range(10):
    model(x)
torch.cuda.synchronize()

start = time.time()
for _ in range(100):
    model(x)
torch.cuda.synchronize()
end = time.time()
```

---

### 10.2 没有同步导致测出来的时间偏小

CUDA 是异步的。

错误方式：

```python
start = time.time()
model(x)
end = time.time()
```

因为 `model(x)` 可能只是提交了 kernel，GPU 还没执行完。

正确方式：

```python
start = time.time()
model(x)
torch.cuda.synchronize()
end = time.time()
```

或者使用 CUDA event：

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
model(x)
end.record()

torch.cuda.synchronize()
print(start.elapsed_time(end), "ms")
```

---

### 10.3 动态 shape 导致不断重编译

如果每次输入 shape 都变，`torch.compile` 可能不断重新编译。

现象：

- 每个 step 都很慢
- log 里持续 compile
- profiler 里有大量 compile 相关 CPU 时间
- GPU 利用率低

解决：

1. 固定 batch size
2. 固定 seq len
3. 使用 padding
4. 使用 static shape
5. 减少动态控制流
6. 必要时对少量 shape 分别 compile

---

### 10.4 graph break 导致 Inductor 没有完整优化

如果模型里有：

```python
print(x)
if x.sum().item() > 0:
    ...
```

或者一些不支持的 Python 逻辑，可能导致 graph break。

graph break 后，一部分图还是 eager 执行，Inductor 优化效果下降。

可以看：

```python
import torch._dynamo as dynamo

explanation = dynamo.explain(model)(x)
print(explanation)
```

或者设置日志：

```bash
TORCH_LOGS="graph_breaks,recompiles" python xxx.py
```

---

### 10.5 autotune 没充分运行

Inductor 有时会用 autotune 选择更好的 Triton config。

如果你只跑一两次，可能还没稳定。

可以考虑：

```python
model = torch.compile(model, mode="max-autotune")
```

或者：

```python
model = torch.compile(model, mode="max-autotune-no-cudagraphs")
```

区别大概是：

| mode | 作用 |
|---|---|
| `default` | 常规编译优化 |
| `reduce-overhead` | 更关注降低 host/launch 开销，可能启用 CUDA Graphs |
| `max-autotune` | 更充分搜索 kernel 配置，编译更慢，运行可能更快 |
| `max-autotune-no-cudagraphs` | 充分 autotune，但不用 cudagraphs |

如果模型是小 kernel、多 launch，优先试：

```python
mode="reduce-overhead"
```

如果模型 GPU kernel 本身重，想压榨 kernel 性能，可以试：

```python
mode="max-autotune"
```

---

## 11. host/device/kernel 三者应该怎么对照看？

你可以建立一个简单的对照表。

---

### 场景 1：kernel 总时间高，host 时间低

现象：

```text
wall time       100 ms
kernel total     95 ms
host total       10 ms
```

因为 host 和 device 可以重叠，所以 host 10ms 不一定叠加到 device 95ms 上。

判断：

```text
device-bound
```

优化：

- top kernel
- fuse
- autotune
- memory/compute 优化

---

### 场景 2：kernel 总时间低，wall time 高

现象：

```text
wall time       100 ms
kernel total     30 ms
host total       80 ms
```

或者 GPU trace 很多空隙。

判断：

```text
host-bound / launch-bound / sync-bound
```

优化：

- CUDA Graphs
- 减少 Python 开销
- 减少同步
- 减少小 kernel
- 固定 shape
- 增强 fusion

---

### 场景 3：kernel 数量多，每个都很短

现象：

```text
kernel 数量：2000 个
平均 kernel duration：5 us
总 kernel time：10 ms
wall time：50 ms
```

判断：

```text
launch overhead 严重
```

优化：

- CUDA Graphs
- Inductor fusion
- 减少算子碎片化
- 避免 graph break
- 使用 static shape
- 减少 Python 层逐个小操作

---

### 场景 4：kernel 很少，但单个 kernel 特别慢

现象：

```text
一个 GEMM kernel 80ms
总 wall time 100ms
```

判断：

```text
单 kernel device-bound
```

优化：

- 看 GEMM shape
- 看 dtype
- 看是否用 Tensor Core
- 看是否 stride/transpose 导致效率低
- 看是否能拆/合
- 看是否可用更高效 backend
- 用 Nsight Compute 分析

---

## 12. 看 Inductor 性能时，推荐的分析顺序

建议按这个顺序看。

---

### 第一步：看稳定端到端 latency

先确保测试方法正确：

```text
warmup -> synchronize -> timing -> synchronize
```

得到稳定的 average latency。

如果没有这个，后面 profiler 数据容易误导。

---

### 第二步：看 GPU 是否忙

用 PyTorch Profiler 或 trace 看：

```text
GPU kernel total time / wall time
```

粗略判断：

```text
如果比例很高：device-bound
如果比例很低：host-bound/launch-bound/sync-bound
```

---

### 第三步：看 top kernel

排序：

```text
cuda_time_total
self_cuda_time_total
```

重点看前 5 或前 10 个 kernel。

问自己：

1. 它是什么类型？GEMM？reduce？elementwise？copy？
2. 它是否合理这么慢？
3. 它是否能被融合？
4. 它是不是大量 dtype cast？
5. 它是不是因为 shape 太小导致 launch 开销突出？
6. 它是不是 Inductor 生成的 Triton kernel？
7. eager 下是不是也慢？

---

### 第四步：看 host gap

打开 trace，看 GPU 时间线。

如果 GPU 上出现大段空白，而 CPU 上有：

```text
aten::item
cudaStreamSynchronize
python function
CUDA API
```

说明 host/sync 导致 GPU 空闲。

---

### 第五步：看 Inductor 是否生效

确认：

1. 有没有 graph break
2. 有没有 recompile
3. 是否生成 Triton kernel
4. 是否 fuse 了预期算子
5. 是否启用 CUDA Graphs
6. 是否 autotune
7. 是否 fallback 到 eager

---

### 第六步：对 top kernel 做深入分析

如果只是 PyTorch Profiler，只能看到 duration。

如果要看 kernel 内部瓶颈，需要：

- Nsight Compute
- Nsight Systems
- Triton benchmark
- Triton autotune log
- roofline 工具

---

## 13. 工具怎么选？

### 13.1 PyTorch Profiler

适合：

- 快速看 CPU/CUDA 耗时
- 找 top op
- 找 top kernel
- 看 host/device 关系
- 导出 trace 给 TensorBoard

命令：

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(x)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
```

---

### 13.2 TensorBoard PyTorch Profiler

适合：

- 可视化 trace
- 看 GPU summary
- 看 kernel 时间线
- 看 host/device overlap
- 看 top kernels
- 看 memory
- 看推荐建议

启动：

```bash
tensorboard --logdir=./tb_trace
```

---

### 13.3 Nsight Systems

适合：

- 看系统级 timeline
- 看 CPU/GPU overlap
- 看 CUDA API
- 看 kernel launch
- 看 GPU gap
- 看多 stream
- 找 host-bound

命令示例：

```bash
nsys profile -o model_trace \
  --trace=cuda,cudnn,cublas,osrt,nvtx \
  python train.py
```

如果你想知道：

```text
GPU 为什么空？
CPU 在干什么？
launch 是否来不及？
```

Nsight Systems 很有用。

---

### 13.4 Nsight Compute

适合：

- 分析单个 GPU kernel
- 看 compute throughput
- 看 memory throughput
- 看 occupancy
- 看 warp stall
- 看 roofline
- 看 shared memory bank conflict
- 看 instruction mix

如果你已经知道：

```text
某个 Triton kernel 很慢
```

想进一步知道：

```text
它是访存瓶颈还是计算瓶颈？
occupancy 为什么低？
```

用 Nsight Compute。

命令示例：

```bash
ncu --set full -o kernel_report python model.py
```

但注意：`ncu` 会 replay kernel，对长时间训练或动态模型不太方便，更适合分析固定 step 的热点 kernel。

---

## 14. 一个实用的判断模板

你每次分析 Inductor 性能时，可以按这个模板填：

```text
1. 端到端 latency: xx ms/iter
2. warmup 是否充分: 是/否
3. 是否同步测量: 是/否
4. GPU kernel total time: xx ms
5. kernel total / wall time: xx%
6. top 5 kernel:
   - xxx: xx ms
   - xxx: xx ms
   - xxx: xx ms
7. GPU 是否有大段空隙: 是/否
8. host 是否有长 CPU op: 是/否
9. 是否存在大量小 kernel: 是/否
10. 是否有 .item()/cpu()/sync: 是/否
11. 是否动态 shape: 是/否
12. 是否有 graph break: 是/否
13. 是否使用 CUDA Graphs: 是/否
14. 是否使用 max-autotune: 是/否
15. 初步判断:
   - device-bound / host-bound / launch-bound / sync-bound
```

---

## 15. 结合 Inductor 的典型优化策略

根据瓶颈不同，策略不一样。

---

### 15.1 host-bound / launch-bound

优先考虑：

```python
torch.compile(model, mode="reduce-overhead")
```

原因：

- 会尽量降低 host overhead
- 可能使用 CUDA Graphs
- 对小 kernel 多、launch 多的模型效果明显

同时：

1. 固定输入 shape
2. 避免动态控制流
3. 去掉 `.item()`
4. 减少 CPU/GPU 同步
5. 减少 Python 侧逻辑
6. 尽量让 Inductor 捕获完整 graph

---

### 15.2 device-bound，kernel 本身慢

优先考虑：

```python
torch.compile(model, mode="max-autotune")
```

或者：

```python
torch.compile(model, mode="max-autotune-no-cudagraphs")
```

原因：

- Inductor/Triton 会搜索更优 kernel config
- 对 GEMM、reduce、pointwise 等可能有收益
- 但编译时间会明显变长

同时：

1. 看 top kernel
2. 看是否 memory-bound
3. 看 dtype
4. 看是否能融合
5. 看是否有不必要 cast
6. 看 shape 是否友好

---

### 15.3 大量 dtype cast

如果 profiler 里看到很多：

```text
triton_poi_fused__to_copy
aten::_to_copy
Memcpy HtoD / DtoD
```

优化：

1. 统一 dtype
2. 避免 fp32/bf16/fp16 来回转
3. 检查 AMP
4. 检查参数 dtype
5. 检查输入 dtype
6. 检查 LayerNorm/Softmax 是否强制 fp32
7. 减少 `.float()`、`.half()`、`.cpu()`

---

### 15.4 小 batch 性能差

小 batch 下常见：

```text
kernel 很小
launch overhead 高
GPU 空转
```

优化：

1. 增大 batch
2. CUDA Graphs
3. 算子融合
4. 减少 kernel 数量
5. static shape
6. 避免每步 sync

---

### 15.5 大 batch 性能差

大 batch 下常见：

```text
device-bound
某个 GEMM/attention kernel 很重
```

优化：

1. 看 top kernel
2. 用 flash attention
3. 用更优 GEMM
4. autotune
5. 优化 seq len / head dim
6. 检查是否触发 fallback
7. 检查 memory 是否 OOM 导致 offload/recompute

---

## 16. 一个完整例子：如何解读 profiler 输出

假设你看到：

```text
Name                         CPU total   Self CPU    CUDA total   Self CUDA
---------------------------  ----------  ----------  -----------  ----------
torch.compile...               500ms       2ms         0.000us      0.000us
triton_red_fused_layernorm     1.2ms      0.1ms       18.3ms       18.3ms
triton_poi_fused_add_mul       0.9ms      0.1ms        6.1ms        6.1ms
ampere_sgemm_128x64_nn         1.5ms      0.2ms       45.2ms       45.2ms
cudaLaunchKernel              25.0ms     25.0ms        0.000us      0.000us
```

这时可以这样分析：

1. CUDA total 最高的是 `ampere_sgemm`，说明 GEMM 是 GPU 热点
2. `triton_red_fused_layernorm` 也占不少 GPU 时间
3. `cudaLaunchKernel` CPU 时间高，说明 launch 开销可能不小
4. 如果 GPU trace 连续，主要瓶颈是 device
5. 如果 GPU trace 有很多空隙，且 `cudaLaunchKernel` 很高，说明 launch-bound 也明显
6. 如果是小模型，考虑 CUDA Graphs
7. 如果是大模型，优先优化 GEMM 和 attention

---

## 17. 最重要的几个经验

---

### 经验 1：端到端 time 是最终标准

profiler 里的 host/device/kernel time 都是辅助。

最终要看：

```text
stable wall time per iteration
```

因为 host 和 device 会重叠，kernel total 也不能完全代表 latency。

---

### 经验 2：先判断瓶颈，再决定优化方向

不要一上来就调 kernel。

先问：

```text
是 GPU 忙，还是 GPU 空？
是 kernel 慢，还是 launch 太多？
是 device-bound，还是 host-bound？
```

不同瓶颈优化方式差别很大。

---

### 经验 3：Inductor 下要特别关注 fusion 和 graph break

Inductor 的性能很大程度来自：

1. 算子融合
2. Triton codegen
3. autotune
4. CUDA Graphs
5. 减少 eager dispatch 开销

如果有大量 graph break，效果会大打折扣。

---

### 经验 4：kernel duration 只是第一层

要看 kernel 好不好，还要看：

```text
duration
compute throughput
memory throughput
occupancy
memory bandwidth
launch config
stall reason
```

尤其是 Triton kernel，config 很关键。

---

### 经验 5：测 Inductor 一定要区分编译期和运行期

第一次运行慢不代表真实性能。

要区分：

```text
compile time
autotune time
warmup time
steady-state inference/training time
```

性能测试一般报 steady-state。

---

## 18. 最简版结论

你可以这样理解：

```text
host 侧耗时：CPU 提交、调度、Python、同步等花的时间。
device 侧耗时：GPU 上执行 kernel、memcpy 等花的时间。
kernel 耗时：GPU 上单个计算 kernel 的执行时间。
```

分析时看：

```text
端到端 latency
    ↓
GPU kernel total time 是否接近 wall time
    ↓
如果接近：device-bound，看 top kernel
如果不接近：host/launch/sync-bound，看 CPU 和 GPU gap
    ↓
再看 top kernel 是计算瓶颈还是访存瓶颈
    ↓
决定用 fusion、autotune、CUDA Graphs、Nsight Compute 等手段
```

简单判断：

```text
GPU 很满，kernel 很慢      -> device-bound
GPU 很空，CPU/launch 很多  -> host-bound / launch-bound
kernel 很多且都很短        -> launch overhead 严重
kernel 少但单个很重        -> 单 kernel 优化
很多 _to_copy/cast        -> dtype/精度转换问题
第一次慢后面快            -> compile/autotune 开销，应 warmup
```
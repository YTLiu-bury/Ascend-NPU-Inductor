# TorchBench A100 Benchmark 完整部署与运行指南

## 1. 目标

本文用于在一台 NVIDIA A100 GPU 机器上部署 PyTorch TorchBench，并运行完整模型集进行性能测试。

目标测试包括：

- Eager
- `torch.compile`
- TorchInductor
- Triton
- CUDA Graph
- Compilation Time
- Cold Start Latency
- Steady-State Latency
- 全量 TorchBench 模型

最终希望得到类似：

| Model | Eager | Compile Time | Inductor | CUDA Graph | Speedup |
|---|---:|---:|---:|---:|---:|
| resnet50 | 2.1 ms | 1.2 s | 1.5 ms | 1.3 ms | 1.62x |
| bert | 5.4 ms | 3.4 s | 3.8 ms | 3.4 ms | 1.59x |

---

## 2. CUDA 12.6 到底怎么办

首先执行：

```bash
nvidia-smi
```

重点看：

```text
Driver Version: xxx.xx
CUDA Version: 12.6
```

这里需要注意：

> `nvidia-smi` 显示的 CUDA Version 主要代表当前 NVIDIA Driver 支持的 CUDA API/runtime 版本，并不等于你的 Docker 容器必须使用 CUDA 12.6。

### 2.1 Host CUDA 和 Container CUDA 可以不同

例如：

```text
Host
CUDA Toolkit = 12.6

        ↓

Docker Container
CUDA = 12.8

        ↓

PyTorch
cu128
```

这种架构是可以工作的。

真正决定 container 能否运行 CUDA 12.8 的关键是 NVIDIA Driver 是否足够新。

### 2.2 如果 Driver 支持 CUDA >= 12.8

推荐直接使用官方 TorchBench nightly Dockerfile，不要修改。

### 2.3 如果 Driver 真的只支持 CUDA 12.6

则需要考虑：

```text
CUDA 12.6
+
PyTorch cu126
```

这种情况下不能只修改 Dockerfile 的 `FROM`，还需要把 PyTorch wheel 从 `cu128` 改成 `cu126`。

例如：

```dockerfile
FROM nvidia/cuda:12.6.x-devel-ubuntu22.04

RUN uv pip install --pre \
    torch torchvision torchaudio torchao \
    --index-url https://download.pytorch.org/whl/nightly/cu126
```

是否有对应日期的 nightly `cu126` wheel，需要以 PyTorch nightly 仓库实际可用版本为准。

---

## 3. 安装 Docker

确认：

```bash
docker --version
```

---

## 4. 安装 NVIDIA Container Toolkit

确认：

```bash
nvidia-ctk --version
```

如果不存在，需要安装 NVIDIA Container Toolkit。

安装完成后：

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 5. 验证 Docker GPU

例如：

```bash
docker run --rm \
    --gpus all \
    nvidia/cuda:12.6.3-base-ubuntu22.04 \
    nvidia-smi
```

如果能够看到 A100，说明：

```text
Docker
    ↓
NVIDIA Container Toolkit
    ↓
NVIDIA Driver
    ↓
A100
```

已经打通。

---

## 6. 获取 TorchBench

```bash
mkdir -p ~/torchbench-work
cd ~/torchbench-work

git clone https://github.com/pytorch/benchmark.git

cd benchmark
```

官方仓库：

https://github.com/pytorch/benchmark

Dockerfile：

https://github.com/pytorch/benchmark/blob/main/docker/torchbench-nightly.dockerfile

---

## 7. 构建官方 TorchBench nightly 镜像

如果 Driver 支持 CUDA 12.8：

```bash
cd ~/torchbench-work/benchmark

docker build \
    -f docker/torchbench-nightly.dockerfile \
    -t torchbench-nightly:latest \
    .
```

注意最后的 `.` 很重要。

正确：

```bash
docker build -f docker/torchbench-nightly.dockerfile .
```

不要：

```bash
docker build docker/
```

---

## 8. 启动 TorchBench Container

```bash
mkdir -p ~/torchbench-work/results
mkdir -p ~/torchbench-work/cache

docker run \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v ~/torchbench-work/results:/workspace/results \
    -v ~/torchbench-work/cache:/root/.cache \
    -it \
    --name torchbench-a100 \
    torchbench-nightly:latest \
    /bin/bash
```

参数：

- `--gpus all`：允许 container 使用 GPU
- `--ipc=host`：共享 IPC，减少 PyTorch shared memory 问题
- `--ulimit memlock=-1`：允许锁定内存
- `--ulimit stack=67108864`：增加 stack limit
- `-v .../results`：把 benchmark 结果保存到宿主机
- `-v .../cache`：保存 PyTorch / Triton / Inductor cache

---

## 9. 验证 A100 和 PyTorch

进入 container 后：

```bash
nvidia-smi
```

然后：

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
PY
```

期望看到：

```text
torch: 2.x.x.dev...
cuda: 12.8
cuda available: True
device: NVIDIA A100...
```

如果使用 cu126 环境，则 CUDA 应显示 `12.6`。

---

## 10. 验证 Triton

```bash
python - <<'PY'
import triton
print("triton:", triton.__version__)
PY
```

---

## 11. 验证 TorchInductor

```bash
python - <<'PY'
import torch

@torch.compile
def f(x):
    return torch.relu(x + 1)

x = torch.randn(1024, 1024, device="cuda")

for _ in range(5):
    y = f(x)

torch.cuda.synchronize()

print("torch.compile OK")
PY
```

---

## 12. TorchBench Runner

完整 TorchBench Dynamo / Inductor benchmark 使用：

```bash
python benchmarks/dynamo/torchbench.py
```

不要把根目录：

```text
run.py
```

和完整 benchmark runner 混淆。

建议先查看当前版本参数：

```bash
python benchmarks/dynamo/torchbench.py --help
```

---

## 13. 第一次先跑单模型

不要第一次就跑全量。

例如：

```bash
cd /workspace/benchmark

python benchmarks/dynamo/torchbench.py \
    --performance \
    --inference \
    --amp \
    --backend inductor \
    --device cuda \
    --only resnet50
```

如果成功，说明 TorchBench → Dynamo → Inductor → GPU 基本打通。

---

## 14. Eager Baseline

```bash
python benchmarks/dynamo/torchbench.py \
    --performance \
    --inference \
    --amp \
    --device cuda \
    --output /workspace/results/eager.csv
```

概念上：

```text
PyTorch
    ↓
Eager
    ↓
CUDA kernels
```

---

## 15. Inductor Benchmark

```bash
python benchmarks/dynamo/torchbench.py \
    --performance \
    --inference \
    --amp \
    --backend inductor \
    --device cuda \
    --output /workspace/results/inductor.csv
```

执行路径：

```text
PyTorch
    ↓
torch.compile
    ↓
TorchDynamo
    ↓
AOTAutograd
    ↓
TorchInductor
    ↓
Triton
    ↓
CUDA
```

---

## 16. Triton

NVIDIA GPU 上，Inductor 会根据计算图生成 GPU kernel。

典型路径：

```text
FX Graph
    ↓
Inductor
    ↓
Triton IR
    ↓
PTX
    ↓
SASS
    ↓
A100
```

因此：

```bash
--backend inductor
```

就是测试 TorchInductor + Triton 的主要入口。

---

## 17. CUDA Graph

如果要测试 CUDA Graph：

```bash
TORCHINDUCTOR_CUDAGRAPHS=1 \
python benchmarks/dynamo/torchbench.py \
    --performance \
    --inference \
    --amp \
    --backend inductor \
    --device cuda \
    --output /workspace/results/inductor_cudagraph.csv
```

注意：

> `backend=inductor` 不等于所有模型一定使用 CUDA Graph。

CUDA Graph 需要满足 capture 条件。

可以使用：

```bash
TORCH_LOGS=perf_hints
```

辅助观察原因。

例如：

```bash
TORCH_LOGS=perf_hints \
TORCHINDUCTOR_CUDAGRAPHS=1 \
python benchmarks/dynamo/torchbench.py \
    --performance \
    --inference \
    --amp \
    --backend inductor \
    --device cuda \
    --only resnet50
```

---

## 18. Compilation Time

如果需要编译时间和 cold start：

```bash
python benchmarks/dynamo/torchbench.py \
    --performance \
    --cold-start-latency \
    --print-compilation-time \
    --inference \
    --amp \
    --backend inductor \
    --device cuda \
    --output /workspace/results/inductor_compile.csv
```

重点参数：

```text
--cold-start-latency
--print-compilation-time
```

---

## 19. 为什么一定要测 Compilation Time

例如：

```text
Eager
    latency = 10 ms
```

而：

```text
torch.compile
    compile = 5 sec
    latency = 5 ms
```

如果只看 steady-state latency，会认为：

```text
2x speedup
```

但第一次运行还有：

```text
5000 ms compile overhead
```

因此完整 benchmark 应记录：

```text
Eager latency
Compile time
Cold-start latency
Steady-state latency
```

---

## 20. 全量模型

确认单模型跑通后，再跑全量。

### Eager

```bash
python benchmarks/dynamo/torchbench.py \
    --performance \
    --inference \
    --amp \
    --device cuda \
    --output /workspace/results/eager.csv
```

### Inductor

```bash
python benchmarks/dynamo/torchbench.py \
    --performance \
    --cold-start-latency \
    --print-compilation-time \
    --inference \
    --amp \
    --backend inductor \
    --device cuda \
    --output /workspace/results/inductor.csv
```

### Inductor + CUDA Graph

```bash
TORCHINDUCTOR_CUDAGRAPHS=1 \
python benchmarks/dynamo/torchbench.py \
    --performance \
    --cold-start-latency \
    --print-compilation-time \
    --inference \
    --amp \
    --backend inductor \
    --device cuda \
    --output /workspace/results/inductor_cudagraph.csv
```

---

## 21. 推荐 Benchmark Matrix

至少建议：

```text
                         Eager    Inductor    CUDAGraph
--------------------------------------------------------
FP32                       ✓         ✓           ✓
FP16                       ✓         ✓           ✓
BF16                       ✓         ✓           ✓
```

如果当前只想快速建立 baseline：

```text
BF16 / FP16
    ↓
Eager
    ↓
Inductor
    ↓
Inductor + CUDA Graph
```

---

## 22. BF16

A100 支持 BF16 Tensor Core，因此 BF16 benchmark 很有价值。

```bash
python benchmarks/dynamo/torchbench.py \
    --performance \
    --cold-start-latency \
    --print-compilation-time \
    --inference \
    --bfloat16 \
    --backend inductor \
    --device cuda
```

具体参数以当前 checkout 的：

```bash
python benchmarks/dynamo/torchbench.py --help
```

为准。

---

## 23. 最终结果目录

建议：

```text
~/torchbench-work/
│
├── benchmark/
│
├── cache/
│
└── results/
    ├── eager.csv
    ├── inductor.csv
    └── inductor_cudagraph.csv
```

---

## 24. 推荐最终收集指标

不要只看 latency。

至少收集：

```text
model
device
dtype
eager_latency
compile_time
cold_start_latency
steady_state_latency
speedup
memory
```

如果要深入分析 Inductor，再增加：

```text
graph_break
kernel_count
fusion_count
generated_kernel
CUDA Graph enabled
CUDA Graph disabled reason
```

---

## 25. Benchmark 环境记录

每次 benchmark 建议保存：

```bash
nvidia-smi

python -c "import torch; print('torch:', torch.__version__)"

python -c "import torch; print('cuda:', torch.version.cuda)"

python -c "import triton; print('triton:', triton.__version__)"

git rev-parse HEAD
```

建议写入：

```text
environment.txt
```

例如：

```bash
{
    date
    nvidia-smi
    python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('device:', torch.cuda.get_device_name(0))"
    python -c "import triton; print('triton:', triton.__version__)"
    git rev-parse HEAD
} 2>&1 | tee /workspace/results/environment.txt
```

这样以后性能变化可以追溯：

- NVIDIA Driver
- CUDA
- PyTorch
- Triton
- TorchBench commit

---

## 26. 为后续 Ascend / torch_npu 对比做准备

最终可以设计成：

```text
                    CUDA A100
                       │
             ┌─────────┴─────────┐
             │                   │
           Eager              Compile
                                 │
                              Inductor
                                 │
                        ┌────────┴────────┐
                        │                 │
                      Triton          CUDA Graph


                    Ascend NPU
                       │
             ┌─────────┴─────────┐
             │                   │
           Eager              Compile
                                 │
                              Inductor
                                 │
                           torch_npu backend
                                 │
                       triton_experimental
```

统一比较：

```text
Model
    │
    ├── Eager Latency
    ├── Compile Time
    ├── Cold Start
    ├── Steady State
    ├── Speedup
    ├── Memory
    ├── Graph Break
    └── Kernel/Fusion
```

这样可以使用同一套 TorchBench workload 对 CUDA Inductor 和 Ascend Inductor 做性能对比。

---

## 27. 一键复制版本

### Host

```bash
mkdir -p ~/torchbench-work
cd ~/torchbench-work

git clone https://github.com/pytorch/benchmark.git

cd benchmark

docker build \
    -f docker/torchbench-nightly.dockerfile \
    -t torchbench-nightly:latest \
    .

mkdir -p ~/torchbench-work/results
mkdir -p ~/torchbench-work/cache

docker run \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v ~/torchbench-work/results:/workspace/results \
    -v ~/torchbench-work/cache:/root/.cache \
    -it \
    --name torchbench-a100 \
    torchbench-nightly:latest \
    /bin/bash
```

### Container

```bash
cd /workspace/benchmark

nvidia-smi

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
PY
```

### 单模型

```bash
python benchmarks/dynamo/torchbench.py \
    --performance \
    --inference \
    --amp \
    --backend inductor \
    --device cuda \
    --only resnet50
```

### Eager

```bash
python benchmarks/dynamo/torchbench.py \
    --performance \
    --inference \
    --amp \
    --device cuda \
    --output /workspace/results/eager.csv
```

### Inductor + Compile Time

```bash
python benchmarks/dynamo/torchbench.py \
    --performance \
    --cold-start-latency \
    --print-compilation-time \
    --inference \
    --amp \
    --backend inductor \
    --device cuda \
    --output /workspace/results/inductor.csv
```

### Inductor + CUDA Graph + Compile Time

```bash
TORCHINDUCTOR_CUDAGRAPHS=1 \
python benchmarks/dynamo/torchbench.py \
    --performance \
    --cold-start-latency \
    --print-compilation-time \
    --inference \
    --amp \
    --backend inductor \
    --device cuda \
    --output /workspace/results/inductor_cudagraph.csv
```

---

## 28. 最重要的注意事项

### 1. 不要看到 CUDA 12.6 就马上修改 Dockerfile

先：

```bash
nvidia-smi
```

看 Driver Version。

### 2. Host CUDA 和 Container CUDA 可以不同

例如：

```text
Host Driver
    ↓
Container CUDA 12.8
    ↓
PyTorch cu128
```

是正常架构。

### 3. torch / torchvision / torchaudio 要保持匹配

尤其 nightly benchmark，尽量使用同一个 nightly channel / 时间点的构建。

### 4. Inductor 不等于一定使用 CUDA Graph

更准确：

```text
torch.compile
    ↓
Inductor
    ├── Triton
    ├── Fusion
    ├── Autotune
    └── CUDA Graph
```

CUDA Graph 是其中一个优化路径。

### 5. 第一次一定先跑单模型

先：

```bash
--only resnet50
```

确认环境没问题，再全量。

### 6. 固定 benchmark 环境

至少记录：

```text
Driver
CUDA
PyTorch
Triton
TorchBench commit
```

---

## 29. 参考

- PyTorch Benchmark: https://github.com/pytorch/benchmark
- TorchBench nightly Dockerfile: https://github.com/pytorch/benchmark/blob/main/docker/torchbench-nightly.dockerfile
- PyTorch documentation: https://pytorch.org/docs/
- PyTorch `torch.compile`: https://docs.pytorch.org/docs/stable/generated/torch.compile.html

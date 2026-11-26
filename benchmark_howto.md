# Benchmark 跑法指南（PR 合入前 40 模型自检）

> 适用对象：第一次跑 NPU benchmark 的同学。目标：对一组有代表性的 40 个模型，分别用**基线构建**（不含待合 PR）和**含 PR 构建**各跑一遍，对比**精度回归**与**大幅性能回归**。
>
> 官方文档（权威，参数以它为准）：
> - PTA 仓库 README：<https://gitcode.com/Ascend/pytorch/blob/master/benchmarks/torchbench/README.md>
> - 本地路径：`<pytorch 源码>/benchmarks/torchbench/README.md`

---

## 1. 一分钟流程

```text
装环境(一次) → 准备数据集(一次) → 建 40 模型清单(一次)
  → 基线构建: 跑 2 个 suite → extract_log → analysis.xlsx
  → 含PR构建: 同样命令再跑一遍 → extract_log → analysis.xlsx
  → 两份 xlsx 逐模型对比: 精度列 + E2E 时间列
```

两个构建之间**只有 pip 装的 torch/torch_npu 版本不同**，机器、卡、环境变量、命令参数必须完全一致，否则数据不可比。

## 2. 准备环境（只做一次）

前置：一台 Ascend NPU 机器（CANN 与要测的 torch_npu 版本匹配），能 `npu-smi info` 看到卡。

```shell
# 1) 源码
git clone https://gitcode.com/Ascend/pytorch.git
cd pytorch

# 2) benchmark 依赖
cd benchmarks/torchbench
pip install -r requirements.txt

# 3) torch / torch_npu（版本必须成对：torch 2.13.0 ↔ torch_npu 2.13.0）
pip install torch torch_npu

# 4) triton 后端需要
pip install triton-ascend

# 5) 内网/无外网环境：模型权重下载源
export HF_ENDPOINT=https://hf-mirror.com
# 个别环境还需 ca-bundle（按团队 wiki 配置 REQUESTS_CA_BUNDLE / SSL_CERT_FILE）

# 6) pytorch/benchmark 子树（README 要求固定 commit）
cd benchmarks/torchbench
git clone https://github.com/pytorch/benchmark.git --depth=1
cd benchmark
git fetch --depth 1 origin fb14629994956c16d27c7a9940c5f081afbf68dc
git checkout fb14629994956c16d27c7a9940c5f081afbf68dc
```

> 基线构建 vs 含 PR 构建：分别安装对应构建产物（如两个 torch_npu wheel）。**换构建后建议新开终端**，避免旧进程残留。

## 3. 准备数据集（只做一次）

部分模型需要输入数据（Super_SloMo / stargan / LearningToPaint / speech_transformer / coco128 / obs.pkl / sam 权重 / truck.jpg），下载地址与放置方式见 README「准备数据集」一节。下载解压后：

```shell
export TORCHBENCH_DATA_PATH=/path/to/dataset   # 两次跑都用同一个目录
```

## 4. 40 模型清单

来源：`每日流水的模型范围.xlsx`（"是否纳入每日流水"=是，共 40 个）。按入口脚本分两组：

**huggingface 入口（16 个）→ `huggingface.py`**，存成 `hf_models.txt`：
```text
AlbertForMaskedLM
BartForCausalLM
BertForMaskedLM
DistilBertForMaskedLM
ElectraForCausalLM
GPT2ForSequenceClassification
MT5ForConditionalGeneration
MobileBertForMaskedLM
OPTForCausalLM
PegasusForCausalLM
Qwen/Qwen3-0.6B
RobertaForCausalLM
T5ForConditionalGeneration
XLNetLMHeadModel
google/gemma-2-2b
openai/whisper-tiny
```

**torchbench 入口（24 个，含 timm）→ `torchbench.py`**，存成 `tb_models.txt`：
```text
inception_v3
repvgg_a2
swin_base_patch4_window7_224
tf_efficientnet_b0
vit_base_patch16_siglip_256
BERT_pytorch
basic_gnn_gcn
basic_gnn_gin
basic_gnn_sage
cm3leon_generate
dcgan
demucs
densenet121
dlrm
llama
llava
mobilenet_v2
nanogpt
resnet50
speech_transformer
torch_multimodal_clip
tts_angular
vision_maskrcnn
yolov3
```

> 注：`--model-list` 一次只跑一个入口脚本的模型，所以分两条命令跑。

## 5. 正式跑（每个构建 2 条命令）

```shell
cd pytorch/benchmarks/torchbench
export ASCEND_RT_VISIBLE_DEVICES=0        # 固定一张卡，两次构建用同一张

# 跑前清缓存（重要！避免跨构建污染）
rm -rf /tmp/torchinductor_root/* ~/.triton/dump ~/.triton/cache

# torchbench 入口
python3 torchbench.py --accuracy --cold-start-latency --train --float32 \
    --backend inductor --npu-backend triton --disable-aclgraph \
    --model-list /path/tb_models.txt --iterations 50 --dump-compile-time \
    2>&1 | tee tb_baseline.log

# huggingface 入口
python3 huggingface.py --accuracy --cold-start-latency --train --float32 \
    --backend inductor --npu-backend triton --disable-aclgraph \
    --model-list /path/hf_models.txt --iterations 50 --dump-compile-time \
    2>&1 | tee hf_baseline.log
```

换成含 PR 的构建后，**参数一字不改**，只把日志名换成 `*_pr.log` 再跑一遍。

变体（按 lane 要求加，两次构建要一致）：
- `--npu-backend dvm`（dvm lane）；`--npu-backend triton_experimental`（实验 triton lane）
- `--dynamic-shapes`（动态 shape 列）
- `--enable-profiler`（需要算子时间时）

## 6. 出结果

```shell
python3 extract_log.py --log_file tb_baseline.log   # → ./analysis.xlsx
python3 extract_log.py --log_file hf_baseline.log
# 含 PR 构建同样两个
```

`analysis.xlsx` 列：`model_name | accuracy | op_compile_time | eager_E2E_avg_time | compile_E2E_avg_time | E2E_speed_up_rate | eager_OP_avg_time | compile_OP_avg_time | OP_speed_up_rate`

## 7. 怎么判（对比规则）

把两份构建的 xlsx 按 `model_name` 对齐，逐模型看：

**a) 精度回归（重点）**
- `accuracy` 列：`pass_accuracy` 正常。
- 基线 pass → 含 PR fail（`fail_accuracy` / `eager_two_runs_differ` / `fail_to_run` / OOM）：**疑似精度回归，先重跑 2~3 次确认**（排除 flaky）。稳定复现 → 报出，附两次日志。
- 已知非确定性模型（`mobilenet_v3_large`、`fastNLP_Bert` 等，见 benchmark 仓库 `torchbench.yaml` 的 `non_deterministic:` 清单）允许 `eager_two_runs_differ` 被 relabel 成 pass，属预期。
- `demucs` 等 `only_training` 模型保持训练态、有 dropout，属 RNG 相关路径，对 flaky 多一分容忍，但连续失败仍要报。

**b) 性能回归（看"大幅"）**
- 主指标：`compile_E2E_avg_time`（图模式单步端到端，ms）。
- 建议判据（团队可再校准）：**单模型 E2E 变慢 > 10% 且绝对差 > 20ms**，或 `E2E_speed_up_rate` 相对下降 > 20% → 标为性能回归。
- 性能波动天然存在：**对命中判据的模型单独重跑 3 次取中位数**再下结论；基线侧同模型也要有 3 次数据。
- `op_compile_time` 只作参考（编译时间对合入决策影响小）。

**c) 数据有效性（先于 a/b 检查）**
- 跑前 `npu-smi info`：卡无 Alarm、无其他租户在跑；机器高负载（load 高）的结果作废重跑。
- 两个构建必须同机、同卡、同 CANN、同环境变量、同命令。
- 换构建后没有清 inductor/triton 缓存 → 数据作废。

## 8. FAQ（踩坑实录）

| 现象 | 原因/处理 |
|---|---|
| `npu-smi` 某卡 Alarm | 换卡：`ASCEND_RT_VISIBLE_DEVICES=<好卡号>` |
| 模型 import 崩（cv2/numpy 不匹配、kaldi_io 旧参数） | 依赖版本坑：`pip list` 核对 numpy/cv2/kaldi_io 与 README 要求；这是环境问题不是 PR 问题 |
| HF 模型权重下载失败 | `HF_ENDPOINT=https://hf-mirror.com`；内网另配 ca-bundle |
| 两次结果忽好忽坏 | 先查数据有效性（8c 三查）；再重跑 3 次取中位 |
| 某模型 `--training` 没进训练态 | `INFERENCE_ONLY_MODELS` 名单自动降级推理，属设计行为 |
| `eager_two_runs_differ` | 先看 `non_deterministic:` 清单（已知非确定性豁免）；不在清单里且稳定复现 → 按精度回归报出 |
| OOM | 降 `--iterations` 无效（OOM 在加载/编译）：确认卡够、`empty_cache` 后重跑，仍 OOM 则记录为环境/模型问题 |

## 9. 产出物约定

每个构建留：`tb_*.log`、`hf_*.log`（原始终端日志）、对应 `analysis.xlsx`。对比结论表建议格式：

| model | acc_baseline | acc_pr | e2e_baseline(ms) | e2e_pr(ms) | Δ% | 判定 |
|---|---|---|---|---|---|---|

判定取值：`OK` / `精度回归(待确认)` / `精度回归` / `性能回归` / `环境噪音(已重跑)`。
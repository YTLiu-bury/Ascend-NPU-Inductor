# torch_npu 自定义 API 扫描问题修改清单

## 基本信息

- 源问题单：`D:\documents\自定义API扫描问题\自定义API扫描问题.csv`
- CSV 读取编码：`GB18030`
- 复核仓库：`D:\code\op-plugin`
- 分支：`26.1.0`
- 复核提交：`0fc818ac`
- 复核日期：2026-07-21
- 本文件仅记录拟修改内容；未修改 `op-plugin` 主仓文件，未提交、未推送、未创建 PR。

## 结论汇总

| 序号 | 复核结论 | 处理建议 |
| --- | --- | --- |
| 171 | 问题仍存在，并发现函数原型参数顺序与 YAML/C++ 不一致 | 修改 Markdown 与 Python docstring |
| 189 | 当前主仓已修复 | 不再修改 |
| 208 | 原扫描问题仍存在，并发现 `quant_mode`、`dst_type_max` 未同步到文档/docstring | 修改 Markdown 与 Python docstring；`dst_type_max` 的完整取值约束需补充权威证据 |
| 227 | 已验证 `min_scale=0` 合法；现有公式把返回的量化尺度写成了倒数 | 保留 `min_scale >= 0`，修正文档公式并同步 `dst_type_max` |
| 240 | Markdown 已补充 `weight_quant_mode=3/4/5`；Python docstring 仍旧，另有一处标点错误 | 同步两个 API 的 docstring，并修正 Markdown 标点 |
| 260 | CSV 所述问题已修复，且 `comm_alg` 是真实参数；但函数原型仍残留 Python 语法错误 | 保留 `comm_alg`，仅删除三个参数前多余的 `int` |
| 281 | 当前主仓已修复；CSV 建议中的 `torch.nn.Swish` 并不存在 | 不再修改，保留 `torch.nn.SiLU` |
| 303 | 问题仍存在，并发现三个真实参数未写入 Markdown/docstring | 修改 Markdown 与 Python docstring，补齐参数说明 |

## 171：npu_prompt_flash_attention

### 修改文件

- `docs/zh/custom_APIs/torch_npu/torch_npu-npu_prompt_flash_attention.md`
- `codegen/templates/_op_plugin_docs.py`

### 核对依据

- YAML `op_plugin/config/op_plugin_functions.yaml:6738` 的关键字参数顺序为 `padding_mask`、`atten_mask`、`pse_shift`、`actual_seq_lengths`。
- C++ `op_plugin/ops/opapi/FlashAttentionKernelNpuOpApi.cpp:959` 的实现签名顺序与 YAML 一致。
- Markdown `:24` 和 docstring `:7795` 当前把 `pse_shift` 放在 `padding_mask`、`atten_mask` 前面。
- Markdown `:52` 和 docstring `:7818` 当前写成“shape 的 s 长度”；同一文档使用 `S(Seq-Length)`、`Q_S`、`KV_S` 表示序列长度，因此应改为无歧义的中文术语。
- 现有 `test_prompt_flash_attention.py` 用例支持省略 `actual_seq_lengths`，与默认值 `None` 一致。

### 拟修改内容

1. Markdown 函数原型按 YAML 调整前三个可选 Tensor 参数的顺序，并补齐逗号后的空格：

```python
torch_npu.npu_prompt_flash_attention(query, key, value, *, padding_mask=None, atten_mask=None, pse_shift=None, actual_seq_lengths=None, deq_scale1=None, quant_scale1=None, deq_scale2=None, quant_scale2=None, quant_offset2=None, num_heads=1, scale_value=1.0, pre_tokens=2147483647, next_tokens=0, input_layout="BSH", num_key_value_heads=0, actual_seq_lengths_kv=None, sparse_mode=0) -> Tensor
```

2. Markdown `actual_seq_lengths` 说明中的句子改为：

```text
如果不指定有效序列长度，可以传入 `None`，表示与 `query` 的序列长度相同。
```

3. Python docstring 原型同步为 YAML 顺序；对应说明改为：

```text
如果不指定有效序列长度，可以传入None，表示与query的序列长度相同。
```

4. Markdown 参数说明块也应按 `padding_mask`、`atten_mask`、`pse_shift`、`actual_seq_lengths` 的顺序排列，保证与函数原型一致。

### 验证

- 静态核对通过：参数名、默认值和顺序可与 YAML/C++ 一一对应。
- 未执行算子测试；本项只改文档表述和参数顺序。

## 189：torch_npu.profiler.profile

### 修改文件

- 无。

### 核对依据

- `docs/zh/custom_APIs/torch_npu-profiler/torch_npu-profiler-profile.md:18` 当前已经是 `with_modules=False`。
- 当前原型还包含 `custom_trace_id_callback=None`，其参数说明和调用示例均已存在。

### 处理结论

CSV 记录对应的 Python 语法问题已在当前主仓修复，不重复修改。

### 验证

- 静态检查通过：`with_modules` 位于带默认值参数之后且自身也有默认值。

## 208：npu_dynamic_quant_asymmetric

### 修改文件

- `docs/zh/custom_APIs/torch_npu/torch_npu-npu_dynamic_quant_asymmetric.md`
- `codegen/templates/_op_plugin_docs.py`

### 核对依据

- Markdown `:42` 当前写成“tensor 的取值必须递增”，指代不明确。
- YAML `op_plugin/config/op_plugin_functions.yaml:6088` 的真实签名还包含 `quant_mode="pertoken"`、`dst_type_max=0.0`。
- C++ `op_plugin/ops/opapi/DynamicQuantKernelNpuOpApi.cpp:188-230` 接收并透传这两个参数；`quant_mode` 对 `perchannel`、`pertensor` 有单独的输出 shape 分支，非默认模式要求较新的 ACLNN 版本；非零 `dst_type_max` 要求 `aclnnDynamicQuantV4`。
- 测试 `test/test_custom_ops/test_npu_dynamic_quant_asymmetric.py:68-69` 已覆盖 `quant_mode="perchannel"`。
- Markdown `:31` 和 docstring `:4726` 尚未包含这两个参数。

### 拟修改内容

1. 将 Markdown 中 `group_index` 的相关句子改为：

```text
`group_index` 的取值必须递增且范围为[1, S]，最后一个值必须等于S（S代表输入`x`的行数，是`x`的shape除最后一维度外的乘积）。
```

2. Markdown 函数原型同步 YAML：

```python
torch_npu.npu_dynamic_quant_asymmetric(x, *, smooth_scales=None, group_index=None, dst_type=None, quant_mode="pertoken", dst_type_max=0.0) -> (Tensor, Tensor, Tensor)
```

3. Python docstring 原型同步为：

```text
torch_npu.npu_dynamic_quant_asymmetric(Tensor x, *, Tensor? smooth_scales=None, Tensor? group_index=None, ScalarType? dst_type=None, str quant_mode="pertoken", float dst_type_max=0.0) -> (Tensor, Tensor, Tensor)
```

4. Markdown 与 docstring 补充参数说明。可确认部分建议写为：

```text
quant_mode：str类型，可选参数，指定量化粒度，默认值为"pertoken"。实现中对"pertoken"、"perchannel"和"pertensor"分别生成对应的scale/offset形状；非默认模式依赖支持aclnnDynamicQuantV3或V4的CANN版本。
dst_type_max：float类型，可选参数，指定量化目标类型的最大值，默认值为0.0。传入非0值时依赖支持aclnnDynamicQuantV4的CANN版本。
```

5. 返回值说明应同步说明：`scale`、`offset` 的 shape 由 `quant_mode` 决定；`pertensor` 为 `[1]`，`pertoken` 为输入 shape 去掉最后一维，`perchannel` 的具体维度由实现按输入最后一维生成。

### 验证

- 静态核对通过：新增参数的名称、类型、顺序和默认值与 YAML/C++ 一致。
- `perchannel` 有已有测试调用证据。
- 仓内没有找到 `dst_type_max` 的完整合法取值范围测试或约束说明，因此不应在文档中自行补写范围。

## 227：npu_dynamic_block_quant

### 修改文件

- `docs/zh/custom_APIs/torch_npu/torch_npu-npu_dynamic_block_quant.md`
- `codegen/templates/_op_plugin_docs.py`

### 核对依据

- YAML `op_plugin/config/op_plugin_functions.yaml:7509` 明确给出 `min_scale=0.0`，并包含尚未写入文档的 `dst_type_max=0.0`。
- C++ `op_plugin/ops/opapi/DynamicBlockQuantNpuOpApi.cpp:25-64` 不限制 `min_scale > 0`，而是把它直接传给 `aclnnDynamicBlockQuant`/`V2`。
- meta 实现 `_meta_registrations.py:6448` 使用默认值 `min_scale=0.0`。
- 测试 `test/test_custom_ops/test_npu_dynamic_block_quant.py:21-30` 的 golden 计算为 `scale = input_max / 127`，测试调用使用默认 `min_scale=0`。
- 文档当前公式写成 `min(MAX / input_max, 1 / min_scale)`，随后又写 `y = x / scale`。这与测试 golden 的 `scale = input_max / MAX` 不一致。
- 真实 NPU/CANN 最小验证结果：
  - 输入 `[1.0, -2.0, 0.5]`、`min_scale=0` 时，输出 `scale=0.0157`，约等于 `2/127`；输出 `y=[64, -127, 32]`，与 `x/scale` 后取整、截断一致。
  - 全零输入、`min_scale=0` 时，输出 `scale=0`、`y=0`，且 `scale` 的有限性检查为 True。
  - 运行过程中出现的 torchair 未编译、32 字节 padding 和 double 转 float 警告均未导致算子失败，不影响本项结论。

### 拟修改内容

1. 不采用 CSV 中把 `min_scale` 改成大于 0 的建议。Markdown 与 docstring 均保留“支持取值大于等于 0”，并明确默认值为 `0.0`。

2. Markdown 计算公式修正为：

```text
scale = max(input_max / DST_TYPE_MAX, min_scale)
y = cast_to_[FP8/HiF8/INT8](x / scale)
```

其中 `DST_TYPE_MAX` 由 `dst_type` 决定。对于已经验证的 INT8 场景，其值为 127。

3. 在公式后补充零值说明：

```text
当一个量化块中的input_max为0且min_scale为0时，scale为0，算子将该块对应的量化输出置为0。
```

4. YAML 和 C++ 已确认当前接口公开 `dst_type_max`，Markdown/docstring 原型补为：

```python
torch_npu.npu_dynamic_block_quant(x, *, min_scale=0.0, round_mode="rint", dst_type=1, row_block_size=1, col_block_size=128, dst_type_max=0.0) -> (Tensor, Tensor)
```

5. Markdown 与 docstring 补充参数说明：

```text
dst_type_max（float）：可选参数，指定量化目标类型的最大值，默认值为0.0。该参数仅在支持aclnnDynamicBlockQuantV2的CANN版本中生效。
```

完整合法取值范围在仓内仍无充分证据，不额外猜测。

### 最小验证用例

```python
import torch
import torch_npu

for x in (
    torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float16, device="npu"),
    torch.zeros((1, 3), dtype=torch.float16, device="npu"),
):
    y, scale = torch_npu.npu_dynamic_block_quant(x, min_scale=0.0)
    print(x, y, scale, torch.isfinite(scale).all())
```

并将非零输入的 `scale` 与按 128 列补齐后的 block `amax / 127` 比较。

### 验证结果

- **验证通过：`min_scale=0` 是有效输入，普通非零输入和全零输入均未发生除零异常。**
- 非零 INT8 输入的实际 `scale` 与 `input_max / 127` 一致，确认现有文档公式方向错误。
- 全零 block 的实际行为为 `scale=0`、`y=0`，文档需作为特殊情况说明。

## 240：npu_mla_prolog_v3 / npu_mla_prolog_v3_functional

### 修改文件

- `docs/zh/custom_APIs/torch_npu/torch_npu-npu_mla_prolog_v3.md`
- `codegen/templates/_op_plugin_docs.py`

### 核对依据

- Markdown `:145` 已准确列出 `weight_quant_mode=0..5`，其中 3 表示 `weight_dq`、`weight_uq_qr`、`weight_dkv_kr` 使用 MXFP8 量化；`weight_uk` 仍为 bfloat16。
- C++ `op_plugin/ops/opapi/MlaPrologV3KernelNpuOpApi.cpp:51-66` 在 mode 3 下校验 MXFP8 对应的 `float8_e8m0fnu` scale 类型。
- C++ `:105-107`、`:297-299` 明确：`weight_quant_mode=3` 且 `kv_cache_quant_mode=1` 时属于 MXFP8 全量化 KV 场景。
- `test/test_custom_ops/test_mla_prolog_v3.py:939-1139` 存在 mode 3 的原地版和 functional 版调用。
- `kv_cache_quant_mode=3` 才是 KV cache 的 pertoken-pergroup（tile）量化；不能把该语义写到 `weight_quant_mode=3` 上。
- Markdown `:147` 已写正确模式但存在 `3-表示` 标点错误。
- docstring `:9557`、`:9833` 的两个函数原型缺少 `actual_seq_len` 之后的大量参数；`:9586`、`:9862` 的 `weight_quant_mode` 仍只写 0/1/2。

### 拟修改内容

1. Markdown 将：

```text
3-表示pertoken-pergroup量化
```

改为：

```text
3表示pertoken-pergroup量化
```

2. 两个 Python docstring 的接口原型完整同步 YAML `op_plugin_functions.yaml:6466`、`:6470`。关键是补齐：

```text
actual_seq_len=None, k_nope_clip_alpha=None, query_norm_flag=False,
weight_quant_mode=0, kv_cache_quant_mode=0, query_quant_mode=0,
ckvkr_repo_mode=0, quant_scale_repo_mode=0, tile_size=128,
token_x_dtype=None, weight_dq_dtype=None, weight_uq_qr_dtype=None,
weight_dkv_kr_dtype=None, kv_cache_dtype=None
```

并保留 YAML 中 `*` 所表示的关键字参数分隔位置。

3. 两个 docstring 的 `weight_quant_mode` 说明统一改为：

```text
weight_quant_mode（int）：可选参数，表示weight_dq、weight_uq_qr、weight_uk、weight_dkv_kr的量化模式。0表示非量化；1表示weight_uq_qr量化；2表示weight_dq、weight_uq_qr、weight_dkv_kr进行int8量化；3表示weight_dq、weight_uq_qr、weight_dkv_kr进行mxfp8量化；4表示weight_dq、weight_uq_qr、weight_dkv_kr进行fp8量化；5表示weight_dq、weight_uq_qr、weight_dkv_kr进行hif8量化。默认值为0。
```

4. 两个 docstring 的 `kv_cache_quant_mode` 中将 `3-表示` 改为 `3表示`，并保持其语义为 pertoken-pergroup 量化。

5. docstring 中缺失参数的说明应从当前 Markdown 对应参数块同步，不改动已验证的默认值和场景约束。

### 验证

- 静态核对通过：mode 3 的精确语义有 C++ dtype 校验、场景注释和已有测试共同支撑。
- 结论：CSV 中“mode 3 表示全量化/per-tile”的建议不够准确，不能照抄。

## 260：npu_moe_distribute_combine_add_rms_norm

### 修改文件

- `docs/zh/custom_APIs/torch_npu/torch_npu-npu_moe_distribute_combine_add_rms_norm.md`

### 核对依据

- Markdown `:132` 当前约束已使用 `comm_quant_mode`，CSV 所述原问题已经修复。
- YAML `op_plugin/config/op_plugin_functions.yaml:6678` 同时包含 `comm_quant_mode=0` 和 `comm_alg=""`。
- C++ `op_plugin/ops/opapi/MoeDistributeCombineAddRmsNormKernelOpApi.cpp:42-114` 接收 `commAlg` 并传给 ACLNN，因此 `comm_alg` 不是错误参数，不能删除或替换。
- Markdown `:48` 的 Python 风格原型仍写有 `int zero_expert_num=0`、`int copy_expert_num=0`、`int const_expert_num=0`，这是无效 Python 语法。
- `codegen/templates/_op_plugin_docs.py` 中未找到该 API 的对应 docstring，故本项没有可同步的现有 docstring 条目。

### 拟修改内容

仅修正原型末尾三项，保留 `comm_alg`：

```python
..., comm_alg="", norm_eps=1e-06, zero_expert_num=0, copy_expert_num=0, const_expert_num=0) -> (Tensor, Tensor, Tensor)
```

### 验证

- 静态核对通过：修改后的参数名、顺序、默认值与 YAML 一致，且是合法 Python 调用式原型。
- CSV 建议“把 `comm_alg` 替换为 `comm_quant_mode`”不应执行。

## 281：torch_npu.contrib.Swish

### 修改文件

- 无。

### 核对依据

- `docs/zh/custom_APIs/scrap_API.md:117-119` 当前已经把 `torch_npu.contrib.Swish` 的替代接口写为 `torch.nn.SiLU`。
- 对应独立文档 `torch_npu-contrib-Swish.md:4` 和 `torch_npu-contrib-module-SiLU.md:4` 也使用 `torch.nn.SiLU`。
- `op_plugin/config/deprecated.yaml:29-30` 对函数式 `npu_silu` 给出的替代接口是 `torch.nn.functional.silu`。

### 处理结论

- 模块式替代：`torch.nn.SiLU`。
- 函数式替代：`torch.nn.functional.silu`。
- `torch.nn.Swish` 不是可用的 PyTorch 接口，因此不能采用 CSV 修改建议中的该名称。
- 当前 `scrap_API.md` 已正确，不再修改。

### 验证

- 静态核对通过：废弃映射和两份独立 API 文档相互一致。

## 303：npu_dequant_swiglu_quant

### 修改文件

- `docs/zh/custom_APIs/torch_npu/torch_npu-npu_dequant_swiglu_quant.md`
- `codegen/templates/_op_plugin_docs.py`

### 核对依据

- Markdown `:66` 和 docstring `:11579-11594` 把 `float glu_alpha=...`、`float glu_bias=...` 写入 Python 调用式原型，语法错误。
- YAML `op_plugin/config/op_plugin_functions.yaml:7023` 的真实签名在 `quant_mode` 后还包含 `dst_type=None`、`round_mode=None`、`activate_dim=None`。
- C++ `op_plugin/ops/opapi/DequantSwigluQuantOpApi.cpp:26-32` 接收这三个参数；`:90-93` 说明未传时分别按 int8、0、-1 处理；`:113-120` 校验 `round_mode` 范围和 `activate_dim`。
- C++ 还要求 `clamp_limit` 为大于 0 的有限数，`glu_alpha`、`glu_bias` 为有限数。
- 测试 `test/test_custom_ops/test_npu_dequant_swiglu_quant.py:379-413` 已实际传入 `dst_type`、`round_mode`、`activate_dim`。

### 拟修改内容

1. Markdown 原型改为合法 Python 调用式，并补齐真实参数：

```python
torch_npu.npu_dequant_swiglu_quant(x, *, weight_scale=None, activation_scale=None, bias=None, quant_scale=None, quant_offset=None, group_index=None, activate_left=False, quant_mode=0, dst_type=None, round_mode=None, activate_dim=None, swiglu_mode=0, clamp_limit=7.0, glu_alpha=1.702, glu_bias=1.0) -> (Tensor, Tensor)
```

2. Python docstring 原型按 YAML 改为：

```text
torch_npu.npu_dequant_swiglu_quant(Tensor x, *, Tensor? weight_scale=None, Tensor? activation_scale=None, Tensor? bias=None, Tensor? quant_scale=None, Tensor? quant_offset=None, Tensor? group_index=None, bool activate_left=False, int quant_mode=0, int? dst_type=None, int? round_mode=None, int? activate_dim=None, int swiglu_mode=0, float clamp_limit=7.0, float glu_alpha=1.702, float glu_bias=1.0) -> (Tensor y, Tensor scale)
```

3. Markdown 和 docstring 在 `quant_mode` 后补充：

```text
dst_type：int类型，可选参数，指定输出数据类型，默认值为None；未传时实现按int8处理。仓内测试存在数值291的调用，但未提供完整枚举含义，不在本文档中猜测其他取值语义。
round_mode：int类型，可选参数，指定取整模式，默认值为None（按0处理）。支持0至4，依次表示rint、round、floor、ceil、trunc。
activate_dim：int类型，可选参数，指定沿哪个维度将输入等分后执行Swiglu，默认值为None（按-1处理）。负值按Python维度规则转换；转换后的取值范围为[0, x.dim()-1]，对应维度长度必须为偶数。
```

4. 同步补充约束：

```text
clamp_limit必须为大于0的有限数；glu_alpha和glu_bias必须为有限数。
```

### 验证

- 静态核对通过：修改后原型与 YAML/C++ 的参数名、顺序、类型和默认值一致。
- 已有测试覆盖三个补充参数的传入路径。
- `dst_type=291` 的准确枚举名称在仓内证据不足，单独保留为待确认项，不猜测。

## 无法确认项目

1. **序号 208：`dst_type_max` 的完整合法取值范围**
   - YAML/C++ 能确认参数、默认值和版本选择行为，但仓内没有完整范围约束或专项测试。

2. **序号 303：`dst_type=291` 的准确公开枚举名称**
   - 测试能确认该数值被传入，但仓内当前证据不足以给出面向用户的准确类型名称。

## 建议落仓顺序

1. 先落 171、208 的确定文本与原型同步。
2. 落 240 的 docstring 同步和标点修正。
3. 落 260、303 的 Python 原型修复及 303 参数补齐。
4. 落 227 的公式修正、零值特殊行为说明和 `dst_type_max` 同步；保留 `min_scale >= 0`。
5. 189、281 保持当前主仓内容不动。

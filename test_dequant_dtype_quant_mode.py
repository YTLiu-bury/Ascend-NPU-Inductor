"""
测试 npu_grouped_matmul_swiglu_quant_v2 的 dequant_dtype 和 quant_mode 参数
测试目标：
  1. 不传参数时的默认值是多少
  2. 传 0 / None / 不传 是否会报错
  3. 950 上 dequant_dtype 真实默认值是 torch.int8 还是 torch.uint8
  4. 各参数组合的兼容性
"""

import torch
import torch_npu
import math
import traceback
from datetime import datetime

# ============================================================
# 测试配置
# ============================================================

# 检测当前设备类型
def detect_device():
    """检测当前是 A2/A3 还是 950 系列"""
    try:
        # 950 支持 float8_e4m3fn，A2/A3 不支持
        x = torch.randint(0, 10, (2, 4), dtype=torch.int8).npu()
        # 尝试创建一个 float8 tensor，看是否支持
        x_f8 = x.to(torch.float8_e4m3fn)
        return "950"
    except Exception:
        pass
    return "A2A3"

DEVICE_TYPE = detect_device()
print(f"[{datetime.now().strftime('%H:%M:%S')}] 检测到设备类型: {DEVICE_TYPE}")

# 基础输入参数（950 pertoken 场景）
E = 2       # expert 数
M = 16      # token 数
K = 8       # 特征维度（8 方便计算）
N = 128     # 输出维度（需偶数）

# ============================================================
# 数据生成函数
# ============================================================

def gen_pertoken_input(E, M, K, N):
    """生成 pertoken 场景的输入数据"""
    x = torch.randint(-128, 127, (M, K), dtype=torch.int8).to(torch.float8_e5m2).npu()
    weight = torch.randint(-128, 127, (E, K, N), dtype=torch.int8).to(torch.float8_e5m2).npu()
    weight_scale = torch.randn(E, N, dtype=torch.float32).npu()
    x_scale = torch.randn(M, dtype=torch.float32).npu()
    group_list = torch.tensor([M // 2, M // 2 + (M % 2)], dtype=torch.int64).npu()
    return x, [weight], [weight_scale], x_scale, group_list

def gen_mx_input(E, M, K, N):
    """生成 MX 量化场景的输入数据（950 only）"""
    x = torch.randint(-128, 127, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
    weight = torch.randint(-128, 127, (E, K, N), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
    weight_scale = torch.randint(0, 256, (E, math.ceil(K / 64), N, 2), dtype=torch.uint8).npu()
    x_scale = torch.randint(0, 256, (M, math.ceil(K / 64), 2), dtype=torch.uint8).npu()
    group_list = torch.tensor([M // 2, M // 2 + (M % 2)], dtype=torch.int64).npu()
    return x, [weight], [weight_scale], x_scale, group_list

# ============================================================
# 测试执行器
# ============================================================

results = []

def run_test(name, test_fn):
    """运行单个测试用例，捕获结果"""
    try:
        test_fn()
        results.append((name, "✅ PASS", ""))
        print(f"  ✅ {name}")
    except Exception as e:
        err_msg = str(e).split('\n')[0][:100]  # 截断过长错误信息
        results.append((name, "❌ FAIL", err_msg))
        print(f"  ❌ {name}: {err_msg}")

def call_api(x, weight, weight_scale, x_scale, group_list,
             dequant_dtype=None, quant_mode=None, dequant_mode=None,
             quant_dtype=None, weight_scale_dtype=None, x_scale_dtype=None,
             **kwargs):
    """统一调用 API，过滤 None 参数"""
    kwargs_dict = dict(
        dequant_dtype=dequant_dtype,
        quant_mode=quant_mode,
        dequant_mode=dequant_mode,
        quant_dtype=quant_dtype,
        weight_scale_dtype=weight_scale_dtype,
        x_scale_dtype=x_scale_dtype,
    )
    # 过滤 None 值（即不传该参数）
    kwargs_dict = {k: v for k, v in kwargs_dict.items() if v is not None}
    kwargs_dict.update(kwargs)
    
    y, y_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
        x, weight, weight_scale, x_scale, group_list,
        **kwargs_dict
    )
    return y, y_scale

# ============================================================
# 测试用例
# ============================================================

print(f"\n{'='*60}")
print(f"测试 1: dequant_dtype 参数探索 (设备: {DEVICE_TYPE})")
print(f"{'='*60}")

# --- 1.1 不传 dequant_dtype（默认行为）---
def test_default_no_dequant_dtype():
    x, w, ws, xs, gl = gen_pertoken_input(E, M, K, N)
    y, ys = call_api(x, w, ws, xs, gl, quant_mode=0)
    assert y.shape == (M, N // 2), f"shape 不匹配: {y.shape}"
run_test("不传 dequant_dtype (默认)", test_default_no_dequant_dtype)

# --- 1.2 传 dequant_dtype=0 ---
def test_dequant_dtype_int_0():
    x, w, ws, xs, gl = gen_pertoken_input(E, M, K, N)
    y, ys = call_api(x, w, ws, xs, gl, quant_mode=0, dequant_dtype=0)
    assert y.shape == (M, N // 2)
run_test("dequant_dtype=0 (int)", test_dequant_dtype_int_0)

# --- 1.3 传 dequant_dtype=torch.int8 ---
def test_dequant_dtype_torch_int8():
    x, w, ws, xs, gl = gen_pertoken_input(E, M, K, N)
    y, ys = call_api(x, w, ws, xs, gl, quant_mode=0, dequant_dtype=torch.int8)
    assert y.shape == (M, N // 2)
run_test("dequant_dtype=torch.int8", test_dequant_dtype_torch_int8)

# --- 1.4 传 dequant_dtype=torch.uint8 ---
def test_dequant_dtype_torch_uint8():
    x, w, ws, xs, gl = gen_pertoken_input(E, M, K, N)
    y, ys = call_api(x, w, ws, xs, gl, quant_mode=0, dequant_dtype=torch.uint8)
    assert y.shape == (M, N // 2)
run_test("dequant_dtype=torch.uint8", test_dequant_dtype_torch_uint8)

# --- 1.5 传 dequant_dtype=torch.float32 ---
def test_dequant_dtype_float32():
    x, w, ws, xs, gl = gen_pertoken_input(E, M, K, N)
    y, ys = call_api(x, w, ws, xs, gl, quant_mode=0, dequant_dtype=torch.float32)
    assert y.shape == (M, N // 2)
run_test("dequant_dtype=torch.float32", test_dequant_dtype_float32)

# --- 1.6 传 dequant_dtype=torch.bfloat16 ---
def test_dequant_dtype_bfloat16():
    x, w, ws, xs, gl = gen_pertoken_input(E, M, K, N)
    y, ys = call_api(x, w, ws, xs, gl, quant_mode=0, dequant_dtype=torch.bfloat16)
    assert y.shape == (M, N // 2)
run_test("dequant_dtype=torch.bfloat16", test_dequant_dtype_bfloat16)

# --- 1.7 传 dequant_dtype=torch.float16 ---
def test_dequant_dtype_float16():
    x, w, ws, xs, gl = gen_pertoken_input(E, M, K, N)
    y, ys = call_api(x, w, ws, xs, gl, quant_mode=0, dequant_dtype=torch.float16)
    assert y.shape == (M, N // 2)
run_test("dequant_dtype=torch.float16", test_dequant_dtype_float16)

# --- 1.8 传 dequant_dtype=1（int 值 1，测试是否表示其他类型）---
def test_dequant_dtype_int_1():
    x, w, ws, xs, gl = gen_pertoken_input(E, M, K, N)
    y, ys = call_api(x, w, ws, xs, gl, quant_mode=0, dequant_dtype=1)
    assert y.shape == (M, N // 2)
run_test("dequant_dtype=1 (int)", test_dequant_dtype_int_1)

# --- 1.9 传 dequant_dtype=2（int 值 2）---
def test_dequant_dtype_int_2():
    x, w, ws, xs, gl = gen_pertoken_input(E, M, K, N)
    y, ys = call_api(x, w, ws, xs, gl, quant_mode=0, dequant_dtype=2)
    assert y.shape == (M, N // 2)
run_test("dequant_dtype=2 (int)", test_dequant_dtype_int_2)


print(f"\n{'='*60}")
print(f"测试 2: quant_mode 参数探索")
print(f"{'='*60}")

# --- 2.1 不传 quant_mode ---
def test_default_no_quant_mode():
    x, w, ws, xs, gl = gen_pertoken_input(E, M, K, N)
    y, ys = call_api(x, w, ws, xs, gl, dequant_dtype=torch.float32)
    assert y.shape == (M, N // 2)
run_test("不传 quant_mode (默认)", test_default_no_quant_mode)

# --- 2.2 quant_mode=0 (pertoken) ---
def test_quant_mode_0():
    x, w, ws, xs, gl = gen_pertoken_input(E, M, K, N)
    y, ys = call_api(x, w, ws, xs, gl, dequant_dtype=torch.float32, quant_mode=0)
    assert y.shape == (M, N // 2)
run_test("quant_mode=0 (pertoken)", test_quant_mode_0)

# --- 2.3 quant_mode=1 (pergroup，950 不支持) ---
def test_quant_mode_1():
    x, w, ws, xs, gl = gen_pertoken_input(E, M, K, N)
    y, ys = call_api(x, w, ws, xs, gl, dequant_dtype=torch.float32, quant_mode=1)
    assert y.shape == (M, N // 2)
run_test("quant_mode=1 (pergroup)", test_quant_mode_1)

# --- 2.4 quant_mode=2 (mx) ---
def test_quant_mode_2():
    x, w, ws, xs, gl = gen_mx_input(E, M, K, N)
    y, ys = call_api(
        x, w, ws, xs, gl, 
        dequant_dtype=torch.float32, quant_mode=2, dequant_mode=2,
        quant_dtype=torch.float8_e4m3fn,
        weight_scale_dtype=torch_npu.float8_e8m0fnu,
        x_scale_dtype=torch_npu.float8_e8m0fnu
    )
    assert y.shape == (M, N // 2)
run_test("quant_mode=2 (mx)", test_quant_mode_2)


print(f"\n{'='*60}")
print(f"测试 3: dequant_dtype + quant_mode 组合测试 (950 MX 场景)")
print(f"{'='*60}")

# 950 MX 场景下 dequant_dtype 的组合
DEQUANT_DTYPES_TO_TEST = [
    ("torch.float32", torch.float32),
    ("torch.bfloat16", torch.bfloat16),
    ("torch.float16", torch.float16),
    ("torch.int8", torch.int8),
    ("torch.uint8", torch.uint8),
    ("不传 (默认)", "DEFAULT"),
    ("0 (int)", 0),
]

for name, dtype in DEQUANT_DTYPES_TO_TEST:
    def make_test(d):
        def test():
            x, w, ws, xs, gl = gen_mx_input(E, M, K, N)
            kwargs = dict(
                quant_mode=2, dequant_mode=2,
                quant_dtype=torch.float8_e4m3fn,
                weight_scale_dtype=torch_npu.float8_e8m0fnu,
                x_scale_dtype=torch_npu.float8_e8m0fnu
            )
            if d == "DEFAULT":
                pass  # 不传 dequant_dtype
            else:
                kwargs["dequant_dtype"] = d
            y, ys = call_api(x, w, ws, xs, gl, **kwargs)
            assert y.shape == (M, N // 2)
        return test
    run_test(f"MX场景 dequant_dtype={name}", make_test(dtype))


print(f"\n{'='*60}")
print(f"测试 4: 不传 dequant_dtype 时检测真实默认值")
print(f"{'='*60}")

# 思路：在不传 dequant_dtype 的情况下，检查输出精度/范围
# 如果能成功执行，说明默认值是一个有效值
# 进一步，对比不传 vs 传不同值的结果是否一致

def test_dequant_dtype_default_detection():
    """
    通过对比不同 dequant_dtype 值下输出是否一致，推断默认值的真实含义。
    如果默认值是 torch.int8，则输出应该与传入 torch.int8 时相同。
    """
    x, w, ws, xs, gl = gen_pertoken_input(E, M, K, N)
    
    # 不传 dequant_dtype
    y_default, ys_default = call_api(x, w, ws, xs, gl, quant_mode=0)
    
    # 传 torch.float32（文档示例用的值）
    y_f32, ys_f32 = call_api(x, w, ws, xs, gl, quant_mode=0, dequant_dtype=torch.float32)
    
    # 传 0（文档说的"默认值0"）
    y_0, ys_0 = call_api(x, w, ws, xs, gl, quant_mode=0, dequant_dtype=0)
    
    # 比较输出是否一致
    match_f32 = torch.allclose(y_default.float(), y_f32.float(), atol=1e-3)
    match_0 = torch.allclose(y_default.float(), y_0.float(), atol=1e-3)
    
    print(f"    默认 vs float32: {'一致' if match_f32 else '不一致'}")
    print(f"    默认 vs 0:      {'一致' if match_0 else '不一致'}")
    
    # 记录结果到全局
    results.append(("默认值 == float32?", "✅" if match_f32 else "❌", ""))
    results.append(("默认值 == 0?", "✅" if match_0 else "❌", ""))

run_test("默认值推断（对比输出）", test_dequant_dtype_default_detection)


# ============================================================
# 结果汇总
# ============================================================

print(f"\n{'='*60}")
print(f"测试汇总 (设备: {DEVICE_TYPE})")
print(f"{'='*60}")
print(f"{'测试名称':<45} {'结果':<10} {'错误信息'}")
print(f"{'-'*45} {'-'*10} {'-'*40}")

pass_count = 0
fail_count = 0
for name, status, err in results:
    if "PASS" in status:
        pass_count += 1
    elif "FAIL" in status:
        fail_count += 1
    # 跳过纯信息行
    if status in ("✅", "❌"):
        print(f"{'  ↳ '+name:<45} {status:<10}")
        continue
    print(f"{name:<45} {status:<10} {err}")

print(f"\n总计: {pass_count} 通过, {fail_count} 失败")

# ============================================================
# 结论
# ============================================================

print(f"\n{'='*60}")
print("结论")
print(f"{'='*60}")

# 分析 dequant_dtype 测试结果
print("\n【dequant_dtype 参数】")
print("  - 不传时: 默认行为（文档说 950 默认 torch.int8，A2/A3 默认 0/float32）")
print("  - 传 0: 表示 float32（A2/A3 的写法）")
print("  - 传 torch.float32/bfloat16/float16: 950 支持")
print("  - 传 torch.int8/uint8: 需看测试结果")

print("\n【quant_mode 参数】")
print("  - 不传时: 默认 0 (pertoken)")
print("  - 0: pertoken 量化")
print("  - 2: MX 量化（950 only）")

print("\n【建议】")
print("  1. 950 上推荐显式传 dequant_dtype=torch.float32（与文档示例一致）")
print("  2. A2/A3 上仅支持 dequant_dtype=0")
print("  3. quant_mode 按场景选: 0=pertoken, 2=mx")
print(f"\n测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Python 浮点数除法的精度：跟随 PyTorch 的真实规则，而不是猜一个

- 状态：已验证（CUDA 实机，与 PyTorch 2.11 逐位对拍）。
- 日期：2026-11-01
- 基线提交：`d8e30464c`（改前）；改动涉及
  `compat/torch/installers/tensor/method_api.py` 的 `_true_division` 与
  `src/bindings/pyjt/py_compat_fast.cc` 的 `scalar_division`。
- 验证范围：RTX 4090 24 GB，CUDA 12.2.140 + `jittor[cuda12]` 组件轮子；对照
  PyTorch 2.11.0+cu128（oracle 解释器）。
- 维护者：Jittor 核心维护者
- 复查条件：`_true_division`/`scalar_division` 的准入、PyTorch 的 Python 标量
  除法语义变化、或换 PyTorch 版本重测

## 现象与既有假设

KI-COMPAT-009 记录的是：`float32 张量 / Python 浮点` 在 CPU 与 CUDA 上把张量升到
float64 再除回 float32，"为了与 PyTorch 逐 ulp 一致"，代价是消费级卡上 1/64 速率的
双精度。这条一直挂着"要不要为 1-ulp 一致付 FP64"的决定。

## 实测：假设反了

在 oracle 解释器（真 PyTorch）上对 20000 个随机 float32 值测三种写法，按位比较：

| 写法 | 与 `torch` 不同的元素 |
| --- | ---: |
| `x / float32(scale)`（f32 除法） | **0** / 20000 |
| `float32(x) / float64(scale)`（f64 路线，旧实现） | 12615 / 20000 |
| `x * float32(1/scale)`（倒数乘） | 5106 / 20000 |

也就是说 **PyTorch 自己就在张量 dtype 里做除法**（弱标量规则），旧实现的 float64
路线才是那个偏离 1 ulp 的一侧；而此前固定在测试里的"torch rounding"位模式，其实是
Jittor 自己的结果。float64 值域下同样：f64 除法与 torch 一致 0 / 20000。

反射方向（`Python 浮点 / 张量`）则是另一种形式：PyTorch 给出的位模式等于
`dtype(scalar) * (1 / tensor)`——float32 与 float64 各 20000 个值上都是 0 差异，
而 IEEE 除法分别是 5106 与 5109 个不同。半精度不遵循这两条，保持原样（先用 float32
做 opmath 再舍入）。

## 修复

- `_true_division`：直接除法在结果 dtype 里计算（`calc_dt = tgt`，只有 half 升到
  float32 做 opmath）；反射除法按上面测到的规则构造成"标量 × 倒数"。
- `py_compat_fast.cc::scalar_division`：原生快路径同样不再升 float64；float32 直接
  除 float32，half 仍升 float32 后舍回。
- `test_python_float_truediv_preserves_torch_rounding_on_cpu` 的期望位模式改成
  oracle 实测值（此前固定的是旧实现的 float64 结果）。

## 结果

- 同一份权重下，shim 进程与 oracle 进程各算 5000 个值，四种组合全部 0 差异：
  float32/float64 × 直接/反射。
- `compat/tests/torch/test_torch_compat_promotion.py` 全绿（28 条，含位模式用例与
  "原生快路径与 Python 路径逐位相同"用例）；`test_native_fast_paths.py` 与
  `test_torch_half_precision_numerics.py` 合计 107 passed，唯一失败是既有的
  `test_an_inference_group_norm_and_the_silu_after_it`（与本次改动无关：把改动整体
  还原到基线提交后该用例同样失败，见下）。
- 副作用：一次标量除法从"cast 到 float64 + 除法 + cast 回"变成一次 float32 除法，
  KI-COMPAT-009 里量到的 0.55 ms/UNet2D 步的 FP64 开销不再存在。

## 未结

- `test_an_inference_group_norm_and_the_silu_after_it` 在 CUDA 上失败：`jt.nn.silu`
  与 `torch.nn.Sequential(SiLU)`（后者与真 PyTorch 逐位相同）在半精度输入上相差
  1 个 half-ulp（约 25% 的元素）。已确认与本页改动无关（把两处改动都还原到基线提交、
  重新编译内核后同样复现），单独运行时不依赖其它测试的状态。待定位：疑似 half 的
  silu 在不同 grad 模式下走了不同的实现。

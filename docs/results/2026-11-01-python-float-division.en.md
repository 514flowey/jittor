# Python-float division: follow PyTorch's actual rule instead of guessing one

- Status: verified on a real CUDA device, bit-compared against PyTorch 2.11.
- Date: 2026-11-01
- Baseline commit: `d8e30464c` (before); the change is in `_true_division`
  (`compat/torch/installers/tensor/method_api.py`) and `scalar_division`
  (`src/bindings/pyjt/py_compat_fast.cc`).
- Verification scope: RTX 4090 24 GB, CUDA 12.2.140 with the `jittor[cuda12]`
  component wheels, against PyTorch 2.11.0+cu128 (the oracle interpreter).
- Maintainer: Jittor core maintainers
- Review conditions: the admission of `_true_division`/`scalar_division`, a
  change to PyTorch's Python-scalar division semantics, or a new PyTorch version

## The symptom and the standing assumption

KI-COMPAT-009 recorded that `float32_tensor / python_float` widens the tensor to
float64 on CPU and CUDA, "to match PyTorch to the last ulp", at the cost of a
1/64-rate double on consumer cards. It sat open as a decision: is 1-ulp parity
worth FP64?

## Measurement: the assumption was backwards

On the oracle interpreter, 20000 random float32 values, compared bit for bit:

| Spelling | elements differing from `torch` |
| --- | ---: |
| `x / float32(scale)` (float32 division) | **0** / 20000 |
| `float32(x) / float64(scale)` (the old route) | 12615 / 20000 |
| `x * float32(1/scale)` (reciprocal multiply) | 5106 / 20000 |

PyTorch divides in the tensor's dtype (its weak-scalar rule); the float64 route
was the side that differed by an ulp, and the bit patterns pinned in the test as
"torch rounding" were Jittor's own. In float64 the plain division also matches
torch on all 20000.

The reflected direction (`python_float / tensor`) is a different form: PyTorch's
bits equal `dtype(scalar) * (1 / tensor)` -- 0 differences over 20000 values in
both float32 and float64, while an IEEE division differs on 5106 and 5109 of
them. Half precision follows neither, so it keeps its float32-opmath division.

## Fix

- `_true_division`: the direct division is computed in the result dtype
  (`calc_dt = tgt`; only half widens to float32 for its opmath); the reflected
  division is built as the measured "scalar times reciprocal" form.
- `py_compat_fast.cc::scalar_division`: the native fast path stops widening to
  float64 as well; float32 divides by float32 and half still widens and rounds
  back.
- `test_python_float_truediv_preserves_torch_rounding_on_cpu` now expects the
  oracle-measured bit patterns instead of the old implementation's.

## Results

- With the same weights, a shim process and the oracle process each compute 5000
  values: all four combinations (float32/float64 x direct/reflected) differ in
  0 elements.
- `compat/tests/torch/test_torch_compat_promotion.py` is green (28 cases,
  including the bit-pattern case and "the native fast path answers bit for bit
  what the Python path answers"); with `test_native_fast_paths.py` and
  `test_torch_half_precision_numerics.py` that is 107 passed, the only failure
  being the pre-existing `test_an_inference_group_norm_and_the_silu_after_it`
  (unrelated: it fails identically with both changes reverted and the core
  rebuilt -- see below).
- Side effect: a scalar division is one float32 division rather than a cast to
  float64, a division and a cast back, so the 0.55 ms/UNet2D-step of FP64 work
  KI-COMPAT-009 measured is gone.

## Open

- `test_an_inference_group_norm_and_the_silu_after_it` fails on CUDA:
  `jt.nn.silu` and `torch.nn.Sequential(SiLU)` (the latter bit-identical to real
  PyTorch) differ by one half-ulp on about 25% of half-precision elements. It
  reproduces with both changes reverted to the baseline commit and the core
  rebuilt, and it fails when run alone, so it does not depend on another test's
  state. Suspected: the half silu takes different implementations under
  different grad modes.

# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Haoyang Peng <2247838039@qq.com>
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Shared array algebra and native-complex bridges."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
import numpy as np


def _reconnect(cached, live):
    """Give a stop-grad'd numpy_code result a real, higher-order-differentiable
    gradient path, without changing its value.

    ``cached`` is a value already computed once inside a ``jt.Function``'s
    ``execute()`` -- ``jt.Function`` tapes and stop-grads its own execute()
    inputs/outputs, so a value cached there (e.g. ``self.mx = inv_via_numpy(x)``)
    is permanently detached from ``x``'s autograd graph. ``live`` is the SAME
    mathematical quantity recomputed via a live, autograd-connected recursive
    call (e.g. calling the public ``inv(x)`` again from inside ``inv``'s own
    ``grad()``, closing over the still-live outer ``x``).

    Returns a Var whose VALUE is exactly ``cached`` (bit-identical --
    ``live - live.detach()`` is a subtraction of a tensor from its own detached
    copy, so it is exactly zero regardless of ``live``'s own numerical
    precision) but whose GRADIENT w.r.t. ``live``'s inputs matches ``live``'s.
    This keeps first-order precision identical to a plain numpy_code op (no
    duplicated rounding from an independent recompute) while still making
    second/third-order derivatives available through the ``live`` path -- the
    standard "stop-gradient trick".
    """
    return cached + (live - live.detach())


def _transpose(x):
    """Batched transpose: swap the last two axes."""
    return np.swapaxes(x, -1, -2)


def _conj_transpose(x):
    """Batched conjugate transpose (the Hermitian adjoint)."""
    return np.conj(np.swapaxes(x, -1, -2))


def _matmul(a, b):
    """Batched matrix product over the last two axes."""
    return np.einsum('...ij,...jk->...ik', a, b)


def _stack_to_complex(x):
    """float ``[..., 2]`` (real, imag) stack -> complex array."""
    return x[..., 0] + 1j * x[..., 1]


def _complex_to_stack(x):
    """complex array -> float ``[..., 2]`` (real, imag) stack."""
    return np.stack([np.real(x), np.imag(x)], axis=-1)


def _is_native_complex(x):
    # A native complex64 Var (NOT an nn.ComplexNumber, which is a plain object).
    import jittor as jt
    return isinstance(x, jt.Var) and "complex" in _jittor_dtype_name(x.dtype)


def _native_to_cn(z):
    # native complex64 [...]  ->  nn.ComplexNumber (differentiable, via P1 bridge)
    import jittor as jt
    from ..nn import ComplexNumber
    return ComplexNumber(jt.nn.view_as_real(z), is_concat_value=True)


def _cn_to_native(cn):
    # nn.ComplexNumber  ->  native complex64 [...]  (differentiable, via P1 bridge)
    # cn.value is the float32 [..., 2] stack; _real2_to_complex64 rebuilds complex64.
    from ..nn.functional.complex import _real2_to_complex64
    return _real2_to_complex64(cn.value)

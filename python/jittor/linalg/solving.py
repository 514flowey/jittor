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
"""Linear solves, inverses, determinants and matrix powers."""
from ._helpers import (
    _cn_to_native, _is_native_complex, _matmul, _native_to_cn, _reconnect,
    _transpose,
)
from .results import INVEX


def inv(x):
    r"""
    calculate the inverse of x.
    :param x (...,M,M):
    :return:x^-1 (...,M,M).
    """
    import jittor as jt
    from ..nn import ComplexNumber
    from .complex import complex_inv
    if _is_native_complex(x):
        # native complex64 -> bridge to the ComplexNumber path, return native.
        return _cn_to_native(complex_inv(_native_to_cn(x)))
    if isinstance(x, ComplexNumber):
        return complex_inv(x)
    def forward_code(np, data):
        a = data["inputs"][0]
        m_a = data["outputs"][0]
        t_a = np.linalg.inv(a)
        np.copyto(m_a, t_a)

    class _Inv(jt.Function):
        # A plain numpy_code op's analytic backward is built as a SECOND,
        # opaque numpy_code call with no backward of its own -- so a second
        # derivative through `inv` raised instead of computing a real answer.
        # Real higher-order AD instead: express the backward *formula* with
        # ordinary differentiable jittor ops (matmul/transpose) and recompute
        # `inv(x)` via a live, non-taped recursive call to this same function.
        # `jt.Function` tapes and stop-grads its own `execute()` inputs/
        # outputs (so a value cached as `self.mx` is permanently detached
        # from x's autograd graph) -- but a value computed by calling
        # `inv(x)` again, closing over the *outer*, still-live `x`, is not,
        # so `_reconnect` reconnects the backward expression to x's real
        # graph. `jt.grad(jt.grad(loss, x), x)` then differentiates straight
        # through it like any composed op. The one-time cost is recomputing
        # the inversion during backward (the cached forward value can't be
        # reused for this) -- the standard trade-off for a recompute-based
        # higher-order-differentiable op.
        def execute(self, xt):
            self.mx = jt.numpy_code([xt.shape], [xt.dtype], [xt], forward_code)[0]
            return self.mx

        def grad(self, dout):
            mx = _reconnect(self.mx, inv(x))
            mxT = mx.transpose(-1, -2)
            return -jt.matmul(jt.matmul(mxT, dout), mxT)

    return _Inv()(x)


def inv_ex(x, *, check_errors=False, out=None):
    r"""
    Compute a matrix inverse and return ``(inverse, info)`` like
    ``torch.linalg.inv_ex``.

    .. warning::
        ``info`` is **always zero**. torch reports a singular input by returning
        ``info > 0`` for the offending matrix and leaving ``check_errors=False``
        callers to build a validity mask from ``info == 0``; Jittor's :func:`inv`
        raises instead, so a singular input never reaches the ``info`` tensor and
        a mask built from it marks every matrix valid. Detect singular inputs by
        catching the exception until non-raising reporting is implemented.
    """
    import jittor as jt
    from .. import _arg_policy
    if not check_errors:
        # check_errors=True happens to be honoured -- jt.linalg.inv raises on a
        # singular input, which is what torch does for that flag.  It is the
        # *default* that is broken: torch promises the caller can keep going and
        # read the failure out of `info`, and here `info` never reports anything.
        _arg_policy.ignored(
            "jittor.linalg.inv_ex", "check_errors", check_errors,
            "info is always 0 -- a singular input raises out of jt.linalg.inv "
            "instead of being reported through info, so an `info == 0` validity "
            "mask is unconditionally all-true (check_errors=True *is* honoured)")
    inverse = inv(x)
    info_shape = tuple(int(s) for s in x.shape[:-2])
    info = jt.zeros(info_shape, dtype="int32")
    if out is not None:
        out_inverse, out_info = out
        out_inverse.assign(inverse)
        out_info.assign(info)
        inverse, info = out_inverse, out_info
    return INVEX(inverse, info)


def pinv(x):
    r"""
    calculate the pseudo-inverse of a x.
    :param x (...,M,N)
    :return: x's pinv (...N,M)
    """
    import jittor as jt
    from ..nn import ComplexNumber
    from .complex import complex_pinv
    if _is_native_complex(x):
        # native complex64 -> bridge to the ComplexNumber path, return native.
        return _cn_to_native(complex_pinv(_native_to_cn(x)))
    if isinstance(x, ComplexNumber):
        # complex pseudo-inverse on the legacy ComplexNumber type. (The real
        # path below cannot take a ComplexNumber — previously this raised.)
        return complex_pinv(x)
    def forward_code(np, data):
        a = data["inputs"][0]
        m_a = data["outputs"][0]
        t_a = np.linalg.pinv(a)
        np.copyto(m_a, t_a)

    def backward_code(np, data):
        T = _transpose
        _dot = _matmul
        dout = data["dout"]
        out = data["outputs"][0]
        inp = data["inputs"][0]
        lmx = data["f_outputs"]
        mx = lmx[0]
        t = T(
            -_dot(_dot(mx, T(dout)), mx)
            + _dot(_dot(_dot(mx, T(mx)), dout), np.eye(inp.shape[-2]) - _dot(inp, mx))
            + _dot(_dot(_dot(np.eye(mx.shape[-2]) - _dot(mx, inp), dout), T(mx)), mx)
        )
        np.copyto(out, t)
    sw = list(x.shape[:-2]) + [x.shape[-1]] + [x.shape[-2]]
    lmx = jt.numpy_code(
        [sw],
        [x.dtype],
        [x],
        forward_code,
        [backward_code],
    )
    mx = lmx[0]
    return mx


def matrix_power(x, n):
    r"""
    Compute the ``n``-th power of a (batch of) square matrix.

    Equivalent to ``torch.linalg.matrix_power`` / ``numpy.linalg.matrix_power``.
    The power is formed entirely from existing jittor ops (``jt.matmul`` and
    :func:`inv`), so the result is differentiable on-device with no numpy
    round-trip.

    :param x (...,M,M): batch of square matrices.
    :param n (int): integer exponent. ``n == 0`` returns identity matrices,
        ``n < 0`` uses the matrix inverse ``x^{-1}`` raised to ``-n``.
    :return: ``x ** n`` (...,M,M).
    """
    import jittor as jt
    if not isinstance(n, int):
        # mirror numpy/torch: only integer exponents are supported
        if hasattr(n, "__index__"):
            n = n.__index__()
        else:
            raise TypeError("matrix_power: exponent 'n' must be an integer")
    if x.shape[-2] != x.shape[-1]:
        raise ValueError("matrix_power expects square matrices (last two dims equal)")

    if n == 0:
        # batched identity, broadcast to x's batch shape and dtype
        m = x.shape[-1]
        eye = jt.init.eye(m, dtype=x.dtype)
        batch = list(x.shape[:-2])
        if batch:
            eye = eye.broadcast(batch + [m, m])
        return eye
    if n < 0:
        x = inv(x)
        n = -n

    # binary exponentiation to keep the matmul count at O(log n)
    result = None
    base = x
    e = n
    while e > 0:
        if e & 1:
            result = base if result is None else jt.matmul(result, base)
        e >>= 1
        if e > 0:
            base = jt.matmul(base, base)
    return result


def det(x):
    r"""
    calculate the determinant of x.
    :param x (...,M,M):
    :return:|x| (...,1)
    """
    import jittor as jt
    def forward_code(np, data):
        a = data["inputs"][0]
        L = data["outputs"][0]
        tL = np.linalg.det(a)
        np.copyto(L, tL)

    s = x.shape
    x_s = s[:-2]
    if len(s) == 2:
        x_s.append(1)

    class _Det(jt.Function):
        # See `inv`/`_Inv` above for the general real-higher-order-AD
        # pattern (live recursive call instead of an opaque numpy_code
        # backward-of-backward). `d(det)/dx = dout * det(x) * inv(x)^T`;
        # both `det(x)` and `inv(x)` recurse into their own (now
        # higher-order-differentiable) public functions, closing over the
        # live `x`, so this composes to any order.
        def execute(self, xt):
            self.d = jt.numpy_code([x_s], [xt.dtype], [xt], forward_code)[0]
            return self.d

        def grad(self, dout):
            d = _reconnect(self.d, det(x))
            n_d = dout.reshape(list(dout.shape) + [1, 1])
            n_o = d.reshape(list(d.shape) + [1, 1])
            # inv(x) was never cached during forward (det's forward never
            # computes it) -- this recompute is not new, the original numpy
            # backward_code called np.linalg.inv(inp) fresh every time too.
            xinvT = inv(x).transpose(-1, -2)
            return (n_d * n_o * xinvT).reshape(x.shape)

    return _Det()(x)


def slogdet(x):
    r"""
    calculate the sign and log of the determinant of x.
    :param x (...,M,M):
    :return sign, x's logdet.
    sign array decides the sign of determinant and their values can be -1,0,1.Only Real number now.0 means det is 0 and logdet is -inf.
    logdet in shape (...,1).
    """
    import jittor as jt
    def forward_code(np, data):
        a = data["inputs"][0]
        sign, m_a = data["outputs"]
        sign_, t_a = np.linalg.slogdet(a)
        np.copyto(m_a, t_a)
        np.copyto(sign, sign_)

    def backward_code(np, data):
        T = _transpose
        _dot = _matmul
        dout = data["dout"]
        out = data["outputs"][0]
        inp = data["inputs"][0]
        out_index = data["out_index"]
        if out_index == 0:
            np.copyto(out, 0)
        if out_index == 1:
            t = np.reshape(dout, np.shape(dout) + (1, 1))
            t = t * T(np.linalg.inv(inp))
            np.copyto(out, t)

    s = x.shape
    det_s = s[:-2]
    if len(det_s) == 0:
        det_s.append(1)
    sign, mx = jt.numpy_code(
        [det_s, det_s],
        [x.dtype, x.dtype],
        [x],
        forward_code,
        [backward_code],
    )
    return sign, mx


def solve(a,b):
    r"""
    Solve a linear matrix equation Ax = B.This is done by calculating x = A^-1B.So A must not be singular.
    :param a:(...,M,M)
    :param b:(...,M)
    :return:solution of Ax = b formula.x in the shape of (...M)
    """
    import jittor as jt
    def forward_code(np, data):
        a, b = data["inputs"]
        L = data["outputs"][0]
        ans = np.linalg.solve(a, b)
        np.copyto(L, ans)

    def _T(v):
        # conjugate (Hermitian) transpose; .conj() is a no-op for real
        # dtypes, so this also covers the real case unchanged, matching the
        # Wirtinger-conjugate convention used throughout this module.
        return v.transpose(-1, -2).conj()

    class _Solve(jt.Function):
        # See `inv`/`_Inv` above for the general real-higher-order-AD
        # pattern. dL/db = A^-H @ dout = solve(A^H, dout); dL/dA = -db @ x^H
        # (x = solve(A,b)), with a vector rhs promoted to a column vector for
        # the outer product and squeezed back for the return shape, matching
        # the original numpy backward's `updim` handling. Both `solve(...)`
        # recursive calls close over the live `a`/`b`, so this composes to
        # any order.
        def execute(self, at, bt):
            self.x = jt.numpy_code([bt.shape], [bt.dtype], [at, bt], forward_code)[0]
            return self.x

        def grad(self, dout):
            aH = _T(a)
            # db has no cached forward value to reuse (it depends on dout,
            # only known at backward time) -- this is not a new recompute,
            # the original numpy backward already called np.linalg.solve
            # independently for the a- and b-gradients.
            db = solve(aH, dout)
            x = _reconnect(self.x, solve(a, b))
            need_squeeze = db.ndim == a.ndim - 1
            db_col = db.unsqueeze(-1) if need_squeeze else db
            x_col = x.unsqueeze(-1) if need_squeeze else x
            dA = -jt.matmul(db_col, _T(x_col))
            return dA.reshape(a.shape), db.reshape(b.shape)

    return _Solve()(a, b)

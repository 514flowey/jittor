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
"""Legacy ComplexNumber matrix decompositions and inversion."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from functools import partial
from ..nn import ComplexNumber
from ._helpers import (
    _complex_to_stack, _conj_transpose, _matmul, _stack_to_complex,
)


def complex_inv(x:ComplexNumber):
    r"""
    calculate the inverse of x.
    :param x (...,M,M):
    :return:x^-1 (...,M,M).

    TODO: Faster Implementation; Check backward.
    """
    import jittor as jt
    if not isinstance(x, ComplexNumber):
        raise TypeError("complex_inv is implemented for nn.ComplexNumber")
    if not (_jittor_dtype_name(x.real.dtype) == "float32" and _jittor_dtype_name(x.imag.dtype) == "float32"):
        raise TypeError("real and imag in ComplexNumber should be jt.float32")
    if x.shape[-2] != x.shape[-1]:
        raise ValueError("only square matrix is supported for complex_inv")

    def forward_code(np, data):

        a = _stack_to_complex(data["inputs"][0])
        m_a = data["outputs"][0]
        t_a = np.linalg.inv(a)
        np.copyto(m_a, _complex_to_stack(t_a))


    def backward_code(np, data):
        T = _conj_transpose
        _dot = _matmul
        dout = _stack_to_complex(data["dout"])
        out = data["outputs"][0]
        mx = _stack_to_complex(data["f_outputs"][0])
        t = -_dot(_dot(T(mx), dout), T(mx))
        np.copyto(out, _complex_to_stack(t))

    lmx = jt.numpy_code(
        x.value.shape,
        x.value.dtype,
        [x.value],
        forward_code,
        [backward_code],
    )

    return ComplexNumber(lmx, is_concat_value=True)


def complex_eig(x:ComplexNumber):
    r"""
    calculate the eigenvalues and eigenvectors of x.
    :param x (...,M,M):
    :return:w, v.
    w (...,M) : the eigenvalues.
    v (...,M,M) : normalized eigenvectors.
    """
    import jittor as jt
    if not isinstance(x, ComplexNumber):
        raise TypeError("complex_eig is implemented for nn.ComplexNumber")
    if not (_jittor_dtype_name(x.real.dtype) == "float32" and _jittor_dtype_name(x.imag.dtype) == "float32"):
        raise TypeError("real and imag in ComplexNumber should be jt.float32")
    if x.shape[-2] != x.shape[-1]:
        raise ValueError("only square matrix is supported for complex_eig")
    def forward_code(np, data):
        a = _stack_to_complex(data["inputs"][0])
        w, v = data["outputs"]
        tw, tv = np.linalg.eig(a)
        np.copyto(w, _complex_to_stack(tw))
        np.copyto(v, _complex_to_stack(tv))

    def backward_code(np, data):
        raise NotImplementedError

    sw = x.shape[:-2] + x.shape[-1:] + (2,)
    sv = x.value.shape
    w, v = jt.numpy_code(
        [sw, sv],
        [x.value.dtype, x.value.dtype],
        [x.value],
        forward_code,
        [backward_code],
    )
    return ComplexNumber(w, is_concat_value=True), ComplexNumber(v, is_concat_value=True)


def complex_eigh(x:ComplexNumber):
    r"""
    Hermitian eigendecomposition of a complex matrix (counterpart of the real
    :func:`eigh`). ``x`` is assumed Hermitian; only the lower triangle is read
    (``UPLO='L'``), matching the real ``eigh``. Returns ``(w, v)`` as
    ``ComplexNumber``\ s for type-consistency with :func:`complex_eig`; the
    eigenvalues ``w`` are mathematically real (carried with a zero imaginary
    part). First-order backward supports distinct eigenvalues and losses
    invariant to the per-eigenvector complex phase. Higher-order derivatives
    through the numpy-code callback are not supported.

    :param x (...,M,M):
    :return: w (...,M) eigenvalues, v (...,M,M) eigenvectors.
    """
    import jittor as jt
    if not isinstance(x, ComplexNumber):
        raise TypeError("complex_eigh is implemented for nn.ComplexNumber")
    if not (_jittor_dtype_name(x.real.dtype) == "float32" and _jittor_dtype_name(x.imag.dtype) == "float32"):
        raise TypeError("real and imag in ComplexNumber should be jt.float32")
    if x.shape[-2] != x.shape[-1]:
        raise ValueError("only square matrix is supported for complex_eigh")
    def forward_code(np, data):
        a = _stack_to_complex(data["inputs"][0])
        w, v = data["outputs"]
        # np.linalg.eigh handles complex Hermitian natively: w real, v complex.
        tw, tv = np.linalg.eigh(a, UPLO='L')
        # carry the (real) eigenvalues as a complex stack (imag = 0) so the
        # ComplexNumber wrapper round-trips cleanly through the P1 bridge.
        np.copyto(w, _complex_to_stack(tw.astype(_jittor_dtype_name(a.dtype))))
        np.copyto(v, _complex_to_stack(tv))

    def backward_code(np, data):
        # Port of 32001665's first-order adjoint. UPLO='L' reads only the
        # lower triangle: each strict-lower entry also contributes through
        # its conjugate mirror, the real diagonal once, and the upper never.
        H = _conj_transpose
        dout = _stack_to_complex(data["dout"])
        out = data["outputs"][0]
        w, v = data["f_outputs"]
        w = np.real(_stack_to_complex(w))
        v = _stack_to_complex(v)
        k = v.shape[-1]
        if data["out_index"] == 0:
            # Eigenvalues have an identically zero imaginary component.
            t = _matmul(v * np.real(dout)[..., np.newaxis, :], H(v))
        else:
            if not np.any(dout):
                np.copyto(out, np.zeros_like(out))
                return
            eye = np.eye(k, dtype=w.dtype)
            differences = w[..., np.newaxis, :] - w[..., :, np.newaxis]
            # The diagonal of V^H dV is a free phase, not an input gradient.
            factors = (1 - eye) / (differences + eye)
            t = _matmul(_matmul(v, factors * _matmul(H(v), dout)), H(v))
        folded = np.tril(t + H(t), -1)
        indices = np.arange(k)
        folded[..., indices, indices] = np.real(np.einsum('...ii->...i', t))
        np.copyto(out, _complex_to_stack(folded))

    sw = x.shape[:-2] + x.shape[-1:] + (2,)
    sv = x.value.shape
    w, v = jt.numpy_code(
        [sw, sv],
        [x.value.dtype, x.value.dtype],
        [x.value],
        forward_code,
        [backward_code],
    )
    return ComplexNumber(w, is_concat_value=True), ComplexNumber(v, is_concat_value=True)


def complex_qr(x):
    r"""
    do the qr factorization of x in the below formula:
    x = QR where Q has orthonormal columns and R is upper-triangular.
    :param x (...,M,N): forward and backward both work for any M, N
        (tall/square M>=N and wide M<N).
    :return: q (...,M,K), r (...,K,N), K=min(M,N).
    """
    import jittor as jt
    if not isinstance(x, ComplexNumber):
        raise TypeError("linalg_qr is implemented for nn.ComplexNumber")
    if not (_jittor_dtype_name(x.real.dtype) == "float32" and _jittor_dtype_name(x.imag.dtype) == "float32"):
        raise TypeError("real and imag in ComplexNumber should be jt.float32")
    m, n = x.shape[-2:]
    k = min(m, n)
    def forward_code(np, data):
        a = _stack_to_complex(data["inputs"][0])
        q_out, r_out = data["outputs"]
        Q, R = np.linalg.qr(a)
        np.copyto(q_out, _complex_to_stack(Q))
        np.copyto(r_out, _complex_to_stack(R))

    def backward_code(np, data):
        # reference: https://github.com/tencent-quantum-lab/tensorcircuit/blob/master/tensorcircuit/backends/pytorch_ops.py
        # Linear in (dq, dr) jointly (no dq*dr cross terms), so a multi-output
        # numpy_code -- one out_index per Q/R, each called with the OTHER
        # cotangent zeroed -- sums to the same total gradient as the original
        # single combined-output call. Verified for M>N (tall) and M==N
        # (square), batched and unbatched, against a numpy finite-difference
        # oracle.
        #
        # Wide (m<n): Q is square (...,m,m), R=[R1|R2] with R1 (...,m,m) upper
        # triangular and R2 (...,m,n-m) the rest. A=[A1|A2], A1=Q@R1 is itself
        # a square QR pair, A2=Q@R2. Matching coefficients of dA1/dA2 in the
        # total differential (same derivation as the tall/square formula
        # below) gives gA2 = Q@gR2 directly, and gA1 = the SAME square-QR
        # formula (`square_grad` below) applied to (Q, R1) with an effective
        # gQ of G = gQ - Q @ gR2 @ H(R2) (the gR1 part of the coupling stays
        # inside the square formula unchanged). Reduces to the m>=n formula
        # exactly when n==m. Verified against a numpy finite-difference
        # oracle for complex64/complex128, batched/unbatched, several (m,n)
        # shapes with m<n.
        H = _conj_transpose
        def _TriangularSolve(x, r):
            return H(np.linalg.solve(r, H(x)))
        _dot = _matmul
        _diag = partial(np.einsum, '...ii->...i')

        dout = _stack_to_complex(data["dout"])
        out = data["outputs"][0]
        out_index = data["out_index"]
        q, r = data["f_outputs"]
        q = _stack_to_complex(q)
        r = _stack_to_complex(r)
        m_dim = q.shape[-2]; n_dim = r.shape[-1]

        def square_grad(rr, dq, dr):
            # Combined square-QR (A=Q@rr, rr square & invertible) adjoint
            # from (dq, dr); the tall/square-case formula, reused unchanged
            # here for both the tall/square path and, applied to R1, the
            # wide path below.
            qdq = _dot(H(q), dq)
            qdq_ = qdq - H(qdq)
            rdr = _dot(rr, H(dr))
            rdr_ = rdr - H(rdr)
            tril = np.tril(qdq_ + rdr_)
            grad_a = _dot(q, dr + _TriangularSolve(tril, rr))
            grad_b = _TriangularSolve(dq - _dot(q, qdq), rr)
            ret = grad_a + grad_b
            m_ = rdr - H(qdq)
            eyem = np.zeros_like(m_)
            _diag(eyem)[:] = _diag(m_)
            correction = eyem - np.real(eyem)
            ret = ret + _TriangularSolve(_dot(q, H(correction)), rr)
            return ret

        if m_dim >= n_dim:
            dq = dout if out_index == 0 else np.zeros_like(q)
            dr = dout if out_index == 1 else np.zeros_like(r)
            ret = square_grad(r, dq, dr)
        else:
            r1 = r[..., :, :m_dim]
            r2 = r[..., :, m_dim:]
            if out_index == 0:
                G = dout
                gR1 = np.zeros_like(r1)
                a2 = np.zeros_like(r2)
            else:
                gR1 = dout[..., :, :m_dim]
                gR2 = dout[..., :, m_dim:]
                G = -_dot(_dot(q, gR2), H(r2))
                a2 = _dot(q, gR2)
            a1 = square_grad(r1, G, gR1)
            ret = np.concatenate([a1, a2], axis=-1)

        np.copyto(out, _complex_to_stack(ret))

    sq = list(x.shape[:-2]) + [m, k, 2]
    sr = list(x.shape[:-2]) + [k, n, 2]
    q, r = jt.numpy_code(
        [sq, sr],
        [x.value.dtype, x.value.dtype],
        [x.value],
        forward_code,
        [backward_code],
    )
    return ComplexNumber(q, is_concat_value=True), ComplexNumber(r, is_concat_value=True)


def complex_svd(x:ComplexNumber):
    r'''
    calculate the Singular Value Decomposition of x.It follows the below fomula:
    x = usv*
    only support full matrices == False ver now, which means:
    x's shape (...,M,K)
    u's shape (...,M,K)
    s's shape (...,K)
    v's shape (...,K,N)
    where K is min(M,N).
    First-order backward supports distinct, nonzero singular values and
    separately phase-invariant singular-vector losses. Joint U/V phase
    coupling and higher-order callback derivatives are not supported yet.
    :param x:
    :return:u,s,v.
    '''
    import jittor as jt
    def forward_code(np, data):
        a = _stack_to_complex(data["inputs"][0])
        u, s, v = data["outputs"]
        #TODO:remove copyto
        tu, ts, tv = np.linalg.svd(a, full_matrices=0)
        np.copyto(u, _complex_to_stack(tu))
        np.copyto(s, _complex_to_stack(ts))
        np.copyto(v, _complex_to_stack(tv))

    def backward_code(np, data):
        # Port of 32001665's reduced complex SVD adjoint, retaining the
        # per-output numpy-code contract. This formula drops diagonal phase
        # terms and therefore only covers separately phase-invariant U/V
        # losses, not general jointly phase-invariant reconstruction losses.
        H = _conj_transpose
        dout = _stack_to_complex(data["dout"])
        out = data["outputs"][0]
        u, s, vh = data["f_outputs"]
        u = _stack_to_complex(u)
        s = np.real(_stack_to_complex(s))
        vh = _stack_to_complex(vh)
        v = H(vh)
        m, k = u.shape[-2:]
        n = vh.shape[-1]
        eye = np.eye(k, dtype=s.dtype)
        out_index = data["out_index"]
        if out_index == 1:
            # The imaginary singular-value output is constant zero.
            t = _matmul(u * np.real(dout)[..., np.newaxis, :], vh)
        else:
            factors = 1 / (
                s[..., np.newaxis, :] ** 2 - s[..., :, np.newaxis] ** 2 + eye
            )
            if out_index == 0:
                projected = _matmul(H(u), dout) * (1 - eye)
                middle = factors * (projected - H(projected))
                t = _matmul(_matmul(u, middle * s[..., np.newaxis, :]), vh)
                if m > n:
                    complement = np.eye(m, dtype=s.dtype) - _matmul(u, H(u))
                    t += _matmul(_matmul(complement, dout / s[..., np.newaxis, :]), vh)
            else:
                projected = _matmul(vh, H(dout)) * (1 - eye)
                middle = factors * (projected - H(projected))
                t = _matmul(_matmul(u, s[..., :, np.newaxis] * middle), vh)
                if m < n:
                    complement = np.eye(n, dtype=s.dtype) - _matmul(v, vh)
                    t += _matmul(_matmul(u / s[..., np.newaxis, :], dout), complement)
        np.copyto(out, _complex_to_stack(t))

    m, n = x.shape[-2:]
    k = min(m, n)
    s1 = list(x.shape)
    s1[-1] = k
    s2 = list(x.shape)
    s2[-2] = k
    s3 = list(x.shape)[:-2]
    s3.append(k)
    s1.append(2)
    s2.append(2)
    s3.append(2)
    u, s, v = jt.numpy_code(
        [s1, s3, s2],
        [x.value.dtype, x.value.dtype, x.value.dtype],
        [x.value],
        forward_code,
        [backward_code],
    )
    return ComplexNumber(u, is_concat_value=True), \
            ComplexNumber(s, is_concat_value=True), \
            ComplexNumber(v, is_concat_value=True)


def complex_pinv(x:ComplexNumber):
    r"""
    Moore-Penrose pseudo-inverse of a complex matrix (counterpart of the real
    :func:`pinv`). For ``x`` of shape ``(...,M,N)`` returns ``(...,N,M)``.
    Forward-only (numpy ``np.linalg.pinv`` handles complex natively), wired
    through the ComplexNumber machinery like ``complex_svd``/``complex_eig``.

    :param x (...,M,N):
    :return: x's pinv (...,N,M).
    """
    import jittor as jt
    if not isinstance(x, ComplexNumber):
        raise TypeError("complex_pinv is implemented for nn.ComplexNumber")
    if not (_jittor_dtype_name(x.real.dtype) == "float32" and _jittor_dtype_name(x.imag.dtype) == "float32"):
        raise TypeError("real and imag in ComplexNumber should be jt.float32")
    def forward_code(np, data):
        a = _stack_to_complex(data["inputs"][0])
        m_a = data["outputs"][0]
        t_a = np.linalg.pinv(a)
        np.copyto(m_a, _complex_to_stack(t_a))

    def backward_code(np, data):
        raise NotImplementedError

    # pinv transposes the last two dims (M,N) -> (N,M); the trailing 2 (re/im) stays.
    sw = list(x.shape[:-2]) + [x.shape[-1], x.shape[-2]] + [2]
    lmx = jt.numpy_code(
        sw,
        x.value.dtype,
        [x.value],
        forward_code,
        [backward_code],
    )
    return ComplexNumber(lmx, is_concat_value=True)

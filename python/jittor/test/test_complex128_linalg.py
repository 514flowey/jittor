# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Jittor Group
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# Native complex128 path of jt.linalg.{inv,svd,svdvals,qr,eig,eigh,pinv}
# (jittor-core-gaps.md SS3.2 acceptance #4), mirroring test_complex64_linalg.py
# at double precision. Each public entry point bridges a native complex128 Var
# to the legacy nn.ComplexNumber implementation and returns native complex128.
#
# History: complex_inv/complex_eig/complex_eigh/complex_qr/complex_pinv in
# linalg.py each had a hardcoded ``assert x.real.dtype == jt.float32`` left
# over from the complex64-only era; native complex128 input (whose ComplexNumber
# bridge value is float64) tripped that assert on BOTH CPU and CUDA, even
# though the forward_code/backward_code bodies were already dtype-agnostic
# (only complex_svd, which has no such assert, happened to work). Fixed by
# widening the assert to accept float32 (complex64) or float64 (complex128)
# consistently for real/imag. No math changed.
#
# We compare the FORWARD result to numpy via reconstruction (eig/svd/qr/inv/
# eigh have sign/phase/order ambiguity, so identities are checked, not raw
# values) and gradcheck backward against a numpy finite-difference oracle, on
# BOTH the CUDA and the CPU backend. Tolerances are tighter than the complex64
# suite since complex128 carries ~1e-15 relative precision through the
# forward_code/backward_code numpy math.
import unittest
import numpy as np
import jittor as jt
from jittor import linalg


def _to_complex128_var(a):
    """numpy complex array -> native complex128 jt.Var."""
    return jt.array(a.astype("complex128"))


def _np(z):
    """native complex128 jt.Var -> numpy complex128 array (for comparison)."""
    assert isinstance(z, jt.Var) and "complex" in str(z.dtype), \
        f"expected native complex128 Var, got {type(z)} dtype={getattr(z,'dtype',None)}"
    return np.asarray(z.numpy()).astype("complex128")


def _dot(a, b):
    return np.einsum("...ij,...jk->...ik", a, b)


def _diag_embed(s):
    k = s.shape[-1]
    out = np.zeros(s.shape + (k,), dtype=s.dtype)
    idx = np.arange(k)
    out[..., idx, idx] = s
    return out


class _Mixin:
    use_cuda = 0

    def setUp(self):
        jt.flags.use_cuda = self.use_cuda

    # -------------------------------------------------------------------- inv
    def test_inv(self):
        rng = np.random.RandomState(0)
        a = (rng.randn(2, 4, 4) + 1j * rng.randn(2, 4, 4))
        z = _to_complex128_var(a)
        zi = linalg.inv(z)
        self.assertEqual(str(zi.dtype), "complex128", "inv must return native complex128")
        eye = np.broadcast_to(np.eye(4), (2, 4, 4))
        rec = _dot(a, _np(zi))
        np.testing.assert_allclose(rec, eye, atol=1e-9, rtol=1e-9)
        rec2 = _dot(_np(zi), a)
        np.testing.assert_allclose(rec2, eye, atol=1e-9, rtol=1e-9)

    # -------------------------------------------------------------------- svd
    def test_svd(self):
        rng = np.random.RandomState(1)
        a = (rng.randn(3, 5, 5) + 1j * rng.randn(3, 5, 5))
        z = _to_complex128_var(a)
        u, s, vh = linalg.svd(z)
        for name, t in (("u", u), ("s", s), ("vh", vh)):
            self.assertEqual(str(t.dtype), "complex128", f"svd {name} must be native complex128")
        un, sn, vhn = _np(u), _np(s), _np(vh)
        np.testing.assert_allclose(sn.imag, 0, atol=1e-9)
        rec = _dot(_dot(un, _diag_embed(sn)), vhn)
        np.testing.assert_allclose(rec, a, atol=1e-9, rtol=1e-9)

    def test_svd_nonsquare(self):
        rng = np.random.RandomState(11)
        a = (rng.randn(2, 6, 3) + 1j * rng.randn(2, 6, 3))  # M>N
        z = _to_complex128_var(a)
        u, s, vh = linalg.svd(z)
        un, sn, vhn = _np(u), _np(s), _np(vh)
        rec = _dot(_dot(un, _diag_embed(sn)), vhn)
        np.testing.assert_allclose(rec, a, atol=1e-9, rtol=1e-9)

    # ---------------------------------------------- svd backward (jittor-core-gaps.md SS3.4)
    def _fd_grad(self, fnp, a, loss_np, eps=1e-6):
        g = np.zeros_like(a)
        for idx in np.ndindex(*a.shape):
            ar = a.copy(); ar[idx] += eps
            am = a.copy(); am[idx] -= eps
            dr = (loss_np(fnp(ar)) - loss_np(fnp(am))) / (2 * eps)
            ai = a.copy(); ai[idx] += eps * 1j
            aim = a.copy(); aim[idx] -= eps * 1j
            di = (loss_np(fnp(ai)) - loss_np(fnp(aim))) / (2 * eps)
            g[idx] = dr + 1j * di
        return g

    def _svd_recon_gradcheck(self, shape, seed):
        rng = np.random.RandomState(seed)
        a = (rng.randn(*shape) + 1j * rng.randn(*shape)).astype("complex128")
        p = (rng.randn(*shape) + 1j * rng.randn(*shape)).astype("complex128") * 0.1

        def fnp(anp):
            u, s, vh = np.linalg.svd(anp, full_matrices=False)
            return (u @ np.diag(s) @ vh if anp.ndim == 2 else
                    np.einsum("...ik,...k,...kj->...ij", u, s, vh))
        def loss_np(rec):
            return float(np.real(np.sum(rec * np.conj(p))))
        ref = self._fd_grad(fnp, a, loss_np)

        x = _to_complex128_var(a)
        x.requires_grad = True
        with jt.enable_grad():
            u, s, vh = linalg.svd(x, full_matrices=False)
            rec = jt.matmul(u * s.unsqueeze(-2), vh)
            loss = (rec * _to_complex128_var(p).conj()).real.sum()
            g = jt.grad(loss, [x])[0]
        got = _np(g)
        np.testing.assert_allclose(got, ref, atol=1e-6, rtol=1e-6,
                                   err_msg=f"svd backward vs finite-diff, shape={shape}")

    def test_svd_backward_square(self):
        self._svd_recon_gradcheck((4, 4), 0)

    def test_svd_backward_tall(self):
        self._svd_recon_gradcheck((6, 3), 1)

    def test_svd_backward_wide(self):
        self._svd_recon_gradcheck((3, 6), 2)

    def test_svd_backward_batched(self):
        self._svd_recon_gradcheck((2, 3, 3), 3)

    # --------------------------------------------- eigh backward (jittor-core-gaps.md SS3.4)
    def test_eigh_backward_eigenvalue_only(self):
        rng = np.random.RandomState(10)
        b = (rng.randn(4, 4) + 1j * rng.randn(4, 4)).astype("complex128")
        a = (b + b.conj().T) / 2
        ws = rng.randn(4).astype("float64")
        def fnp(anp):
            return np.linalg.eigh(anp, UPLO="L")[0]
        def loss_np(w):
            return float(np.sum(w * ws))
        ref = self._fd_grad(fnp, a, loss_np)
        x = _to_complex128_var(a)
        x.requires_grad = True
        with jt.enable_grad():
            w, v = linalg.eigh(x)
            loss = (w.real * jt.array(ws)).sum()
            g = jt.grad(loss, [x])[0]
        got = _np(g)
        np.testing.assert_allclose(got, ref, atol=1e-6, rtol=1e-6)

    def test_eigh_backward_eigenvector_dependent(self):
        rng = np.random.RandomState(11)
        b = (rng.randn(5, 5) + 1j * rng.randn(5, 5)).astype("complex128")
        a = (b + b.conj().T) / 2
        p = (rng.randn(5, 5) + 1j * rng.randn(5, 5)).astype("complex128") * 0.1
        def fnp(anp):
            w, v = np.linalg.eigh(anp, UPLO="L")
            return v @ np.diag(w) @ v.conj().T
        def loss_np(rec):
            return float(np.real(np.sum(rec * np.conj(p))))
        ref = self._fd_grad(fnp, a, loss_np)
        x = _to_complex128_var(a)
        x.requires_grad = True
        with jt.enable_grad():
            w, v = linalg.eigh(x)
            rec = jt.matmul(v * w.unsqueeze(-2), v.conj().transpose(-1, -2))
            loss = (rec * _to_complex128_var(p).conj()).real.sum()
            g = jt.grad(loss, [x])[0]
        got = _np(g)
        np.testing.assert_allclose(got, ref, atol=1e-6, rtol=1e-6)

    # ----------------------------------------------- qr shape/backward (jittor-core-gaps.md SS3.4)
    def test_qr_tall_forward_and_backward(self):
        rng = np.random.RandomState(12)
        shape = (6, 3)
        a = (rng.randn(*shape) + 1j * rng.randn(*shape)).astype("complex128")
        z = _to_complex128_var(a)
        q, r = linalg.qr(z)
        self.assertEqual(tuple(q.shape), (6, 3))
        self.assertEqual(tuple(r.shape), (3, 3))
        qn, rn = _np(q), _np(r)
        np.testing.assert_allclose(_dot(qn, rn), a, atol=1e-9, rtol=1e-9)
        qhq = _dot(np.conj(np.swapaxes(qn, -1, -2)), qn)
        np.testing.assert_allclose(qhq, np.eye(3), atol=1e-9, rtol=1e-9)

        p = (rng.randn(*shape) + 1j * rng.randn(*shape)).astype("complex128") * 0.1
        def fnp(anp):
            qq, rr = np.linalg.qr(anp)
            return qq @ rr
        def loss_np(rec):
            return float(np.real(np.sum(rec * np.conj(p))))
        ref = self._fd_grad(fnp, a, loss_np)
        x = _to_complex128_var(a)
        x.requires_grad = True
        with jt.enable_grad():
            qb, rb = linalg.qr(x)
            rec = jt.matmul(qb, rb)
            loss = (rec * _to_complex128_var(p).conj()).real.sum()
            g = jt.grad(loss, [x])[0]
        got = _np(g)
        np.testing.assert_allclose(got, ref, atol=1e-6, rtol=1e-6)

    def test_svdvals(self):
        rng = np.random.RandomState(2)
        a = (rng.randn(4, 4) + 1j * rng.randn(4, 4))
        z = _to_complex128_var(a)
        s = linalg.svdvals(z)
        sn = _np(s)
        ref = np.linalg.svd(a, compute_uv=False)
        np.testing.assert_allclose(np.sort(sn.real), np.sort(ref), atol=1e-9, rtol=1e-9)

    # --------------------------------------------------------------------- qr
    def test_qr(self):
        rng = np.random.RandomState(3)
        a = (rng.randn(2, 4, 4) + 1j * rng.randn(2, 4, 4))
        z = _to_complex128_var(a)
        q, r = linalg.qr(z)
        self.assertTrue("complex" in str(q.dtype) and "complex" in str(r.dtype),
                        "qr q,r must be native complex128")
        qn, rn = _np(q), _np(r)
        np.testing.assert_allclose(_dot(qn, rn), a, atol=1e-9, rtol=1e-9)
        qhq = _dot(np.conj(np.swapaxes(qn, -1, -2)), qn)
        np.testing.assert_allclose(qhq, np.broadcast_to(np.eye(4), (2, 4, 4)),
                                   atol=1e-9, rtol=1e-9)

    # -------------------------------------------------------------------- eig
    def test_eig(self):
        if self.use_cuda:
            # PRE-EXISTING platform limitation (not a regression of this fix):
            # general non-Hermitian eig runs through jt.numpy_code, which binds
            # `np` to cupy on CUDA, and cupy.linalg has NO `eig` (only `eigh`).
            # Same limitation as complex64's test_eig on CUDA.
            self.skipTest("cupy.linalg has no eig() (general eig is CPU-only); "
                          "pre-existing — see test_complex64_linalg.py::test_eig")
        rng = np.random.RandomState(4)
        a = (rng.randn(2, 4, 4) + 1j * rng.randn(2, 4, 4))
        z = _to_complex128_var(a)
        w, v = linalg.eig(z)
        self.assertTrue("complex" in str(w.dtype) and "complex" in str(v.dtype),
                        "eig w,v must be native complex128")
        wn, vn = _np(w), _np(v)
        lhs = _dot(a, vn)
        rhs = _dot(vn, _diag_embed(wn))
        np.testing.assert_allclose(lhs, rhs, atol=1e-9, rtol=1e-9)

    # ------------------------------------------------------------------- eigh
    def test_eigh(self):
        rng = np.random.RandomState(5)
        b = (rng.randn(2, 4, 4) + 1j * rng.randn(2, 4, 4))
        a = b + np.conj(np.swapaxes(b, -1, -2))  # Hermitian
        z = _to_complex128_var(a)
        w, v = linalg.eigh(z)
        self.assertTrue("complex" in str(w.dtype) and "complex" in str(v.dtype),
                        "eigh w,v must be native complex128")
        wn, vn = _np(w), _np(v)
        np.testing.assert_allclose(wn.imag, 0, atol=1e-9)
        lhs = _dot(a, vn)
        rhs = _dot(vn, _diag_embed(wn))
        np.testing.assert_allclose(lhs, rhs, atol=1e-9, rtol=1e-9)
        ref = np.linalg.eigh(a, UPLO='L')[0]
        np.testing.assert_allclose(np.sort(wn.real, axis=-1), ref, atol=1e-9, rtol=1e-9)

    # ------------------------------------------------------------------- pinv
    def test_pinv_square(self):
        rng = np.random.RandomState(6)
        a = (rng.randn(4, 4) + 1j * rng.randn(4, 4))
        z = _to_complex128_var(a)
        p = linalg.pinv(z)
        self.assertEqual(str(p.dtype), "complex128", "pinv must be native complex128")
        pn = _np(p)
        np.testing.assert_allclose(_dot(_dot(a, pn), a), a, atol=1e-9, rtol=1e-9)

    def test_pinv_nonsquare(self):
        rng = np.random.RandomState(7)
        a = (rng.randn(2, 5) + 1j * rng.randn(2, 5))  # 2x5 -> pinv 5x2
        z = _to_complex128_var(a)
        p = linalg.pinv(z)
        pn = _np(p)
        self.assertEqual(tuple(p.shape), (5, 2))
        np.testing.assert_allclose(_dot(_dot(a, pn), a), a, atol=1e-9, rtol=1e-9)
        ref = np.linalg.pinv(a)
        np.testing.assert_allclose(pn, ref, atol=1e-9, rtol=1e-9)

    # ------------------------------------------------------- complex64 path intact
    def test_complex64_path_unchanged(self):
        # native complex64 must still work after widening the ComplexNumber
        # dtype assert to accept float32 OR float64 (not just float64).
        rng = np.random.RandomState(8)
        a = (rng.randn(4, 4) + 1j * rng.randn(4, 4)).astype("complex64")
        z = jt.array(a)
        zi = linalg.inv(z)
        self.assertEqual(str(zi.dtype), "complex64")
        np.testing.assert_allclose(_dot(a.astype("complex128"), _np(zi)), np.eye(4),
                                   atol=1e-3, rtol=1e-3)

    # ------------------------------------------------------- mixed dtype rejected
    def test_mixed_real_imag_dtype_rejected(self):
        # a real/imag pair with MISMATCHED float precision must still be
        # rejected (the widened assert is not a blanket bypass). jt.array()
        # narrows a bare float64 numpy array to float32 by default, so pass
        # dtype= explicitly to actually force a float32/float64 mismatch.
        from jittor.nn import ComplexNumber
        from jittor.linalg import complex_inv
        real32 = jt.array(np.random.randn(3, 3), dtype="float32")
        imag64 = jt.array(np.random.randn(3, 3), dtype="float64")
        self.assertNotEqual(str(real32.dtype), str(imag64.dtype))
        with self.assertRaises(Exception):
            cn = ComplexNumber(real32, imag64)
            complex_inv(cn)


@unittest.skipIf(not jt.has_cuda, "no cuda found")
class TestComplex128LinalgCUDA(_Mixin, unittest.TestCase):
    use_cuda = 1


class TestComplex128LinalgCPU(_Mixin, unittest.TestCase):
    use_cuda = 0


if __name__ == "__main__":
    unittest.main()

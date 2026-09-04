# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Jittor Group
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# Phase 6 "P4": verify the NATIVE complex64 path of jt.linalg.{inv,svd,svdvals,
# qr,eig,eigh,pinv}. Each public entry point bridges a native complex64 Var to
# the legacy nn.ComplexNumber implementation and returns native complex64. We
# compare the FORWARD result to numpy via reconstruction (eig/svd/qr/inv/eigh
# have sign/phase/order ambiguity, so identities are checked, not raw values),
# on BOTH the CUDA and the CPU backend, and assert the legacy ComplexNumber
# input path is unchanged (no regression).
import unittest
import numpy as np
import jittor as jt
from jittor import linalg


def _to_complex64_var(a):
    """numpy complex array -> native complex64 jt.Var."""
    return jt.array(a.astype("complex64"))


def _np(z):
    """native complex64 jt.Var -> numpy complex128 array (for comparison)."""
    assert isinstance(z, jt.Var) and "complex" in str(z.dtype), \
        f"expected native complex64 Var, got {type(z)} dtype={getattr(z,'dtype',None)}"
    return z.numpy().astype("complex128")


def _dot(a, b):
    return np.einsum("...ij,...jk->...ik", a, b)


def _diag_embed(s):
    # s: (...,K) -> (...,K,K) diagonal matrices
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
        z = _to_complex64_var(a)
        zi = linalg.inv(z)
        self.assertTrue("complex" in str(zi.dtype), "inv must return native complex64")
        eye = np.broadcast_to(np.eye(4), (2, 4, 4))
        rec = _dot(a, _np(zi))
        np.testing.assert_allclose(rec, eye, atol=1e-3, rtol=1e-3)
        # also a @ inv(a) via inv(a) @ a
        rec2 = _dot(_np(zi), a)
        np.testing.assert_allclose(rec2, eye, atol=1e-3, rtol=1e-3)

    # -------------------------------------------------------------------- svd
    def test_svd(self):
        rng = np.random.RandomState(1)
        a = (rng.randn(3, 5, 5) + 1j * rng.randn(3, 5, 5))
        z = _to_complex64_var(a)
        u, s, vh = linalg.svd(z)
        for name, t in (("u", u), ("s", s), ("vh", vh)):
            self.assertTrue("complex" in str(t.dtype), f"svd {name} must be native complex64")
        un, sn, vhn = _np(u), _np(s), _np(vh)
        # singular values are real and non-negative
        np.testing.assert_allclose(sn.imag, 0, atol=1e-3)
        # reconstruction: a == u @ diag(s) @ vh  (numpy svd returns vh already)
        rec = _dot(_dot(un, _diag_embed(sn)), vhn)
        np.testing.assert_allclose(rec, a, atol=1e-3, rtol=1e-3)

    def test_svd_nonsquare(self):
        rng = np.random.RandomState(11)
        a = (rng.randn(2, 6, 3) + 1j * rng.randn(2, 6, 3))  # M>N
        z = _to_complex64_var(a)
        u, s, vh = linalg.svd(z)
        un, sn, vhn = _np(u), _np(s), _np(vh)
        rec = _dot(_dot(un, _diag_embed(sn)), vhn)
        np.testing.assert_allclose(rec, a, atol=1e-3, rtol=1e-3)

    # ------------------------------------------------------------------ solve
    def test_solve(self):
        rng = np.random.RandomState(3)
        a = rng.randn(4, 4) + 1j * rng.randn(4, 4)
        b = rng.randn(4) + 1j * rng.randn(4)
        za, zb = _to_complex64_var(a), jt.array(b.astype("complex64"))
        x = linalg.solve(za, zb)
        self.assertTrue("complex" in str(x.dtype), "solve must return native complex64")
        np.testing.assert_allclose(a @ _np(x), b, atol=1e-3, rtol=1e-3)

    # jittor-core-gaps.md §3.3: complex `solve` backward used a plain
    # transpose (np.swapaxes only, no conjugate) instead of the conjugate
    # transpose the Wirtinger convention requires, silently producing wrong
    # A- and b-gradients for any complex dtype. Gradcheck against a numpy
    # finite-difference oracle with a loss that is NOT gauge/phase invariant
    # (unlike the svd/eigh gradchecks below) since solve has no such
    # ambiguity -- a regression back to plain transpose would show up
    # directly here, not be masked by a symmetric test.
    def test_solve_backward_complex_conjugate(self):
        rng = np.random.RandomState(4)
        n = 4
        a = rng.randn(n, n) + 1j * rng.randn(n, n)
        b = rng.randn(n) + 1j * rng.randn(n)
        p = (rng.randn(n) + 1j * rng.randn(n)) * 0.1

        def loss_np(x):
            return float(np.real(np.sum(x * np.conj(p))))

        ref_a = self._fd_grad(lambda av: np.linalg.solve(av, b), a, loss_np)
        ref_b = self._fd_grad(lambda bv: np.linalg.solve(a, bv), b, loss_np)
        # Sanity check that the finite-difference oracle itself has a
        # non-trivial imaginary part -- otherwise this test could not
        # distinguish a conjugate transpose from a plain transpose.
        self.assertGreater(np.abs(ref_a.imag).max(), 1e-3)

        za = _to_complex64_var(a)
        zb = jt.array(b.astype("complex64"))
        za.requires_grad = True
        zb.requires_grad = True
        with jt.enable_grad():
            x = linalg.solve(za, zb)
            loss = (x * _to_complex64_var(p).conj()).real.sum()
            gA, gb = jt.grad(loss, [za, zb])
        np.testing.assert_allclose(_np(gA), ref_a, atol=3e-2, rtol=3e-2,
                                   err_msg="solve backward wrt A vs finite-diff")
        np.testing.assert_allclose(_np(gb), ref_b, atol=3e-2, rtol=3e-2,
                                   err_msg="solve backward wrt b vs finite-diff")

    # ---------------------------------------------- svd backward (jittor-core-gaps.md §3.4)
    # Complex SVD/eigh backward were previously `raise NotImplementedError`. A per-column
    # unitary phase (u_i -> e^{i theta} u_i, v_i -> e^{i theta} v_i, A = U S V^H invariant)
    # is a genuine gauge freedom the analytic formula cannot resolve, matching torch's own
    # documented limitation -- so these gradchecks use PHASE-INVARIANT losses (reconstruction:
    # U@diag(S)@Vh, or V@diag(w)@V^H for eigh), which is also what real workloads use
    # (MPS truncation, VQE energies, density matrices), not raw per-vector cotangents.
    def _fd_grad(self, fnp, a, loss_np, eps=1e-3):
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
        a = (rng.randn(*shape) + 1j * rng.randn(*shape)).astype("complex64")
        p = (rng.randn(*shape) + 1j * rng.randn(*shape)).astype("complex64") * 0.1

        def fnp(anp):
            u, s, vh = np.linalg.svd(anp, full_matrices=False)
            return (u @ np.diag(s) @ vh if anp.ndim == 2 else
                    np.einsum("...ik,...k,...kj->...ij", u, s, vh))
        def loss_np(rec):
            return float(np.real(np.sum(rec * np.conj(p))))
        ref = self._fd_grad(fnp, a, loss_np)

        x = _to_complex64_var(a)
        x.requires_grad = True
        with jt.enable_grad():
            u, s, vh = linalg.svd(x, full_matrices=False)
            rec = jt.matmul(u * s.unsqueeze(-2), vh)
            loss = (rec * _to_complex64_var(p).conj()).real.sum()
            g = jt.grad(loss, [x])[0]
        got = _np(g)
        np.testing.assert_allclose(got, ref, atol=3e-2, rtol=3e-2,
                                   err_msg=f"svd backward vs finite-diff, shape={shape}")

    def test_svd_backward_square(self):
        self._svd_recon_gradcheck((4, 4), 0)

    def test_svd_backward_tall(self):
        self._svd_recon_gradcheck((6, 3), 1)

    def test_svd_backward_wide(self):
        self._svd_recon_gradcheck((3, 6), 2)

    def test_svd_backward_batched(self):
        self._svd_recon_gradcheck((2, 3, 3), 3)

    def test_svd_backward_singular_values_real_dtype(self):
        # §3.4 design note: S must carry a gradient consistently even though it is
        # mathematically real (the native-complex bridge represents it as complex64
        # with imag~0); this locks that the S-only branch alone is correct and finite.
        rng = np.random.RandomState(9)
        a = (rng.randn(4, 4) + 1j * rng.randn(4, 4)).astype("complex64")
        w = rng.randn(4).astype("float32") * 0.1
        def fnp(anp):
            return np.linalg.svd(anp, compute_uv=False)
        def loss_np(s):
            return float(np.sum(s * w))
        ref = self._fd_grad(fnp, a, loss_np)
        x = _to_complex64_var(a)
        x.requires_grad = True
        with jt.enable_grad():
            s = linalg.svdvals(x)
            loss = (s.real * jt.array(w)).sum()
            g = jt.grad(loss, [x])[0]
        got = _np(g)
        self.assertTrue(np.isfinite(got).all())
        np.testing.assert_allclose(got, ref, atol=3e-2, rtol=3e-2)

    # --------------------------------------------- eigh backward (jittor-core-gaps.md §3.4)
    def test_eigh_backward_eigenvalue_only(self):
        rng = np.random.RandomState(10)
        b = (rng.randn(4, 4) + 1j * rng.randn(4, 4)).astype("complex64")
        a = (b + b.conj().T) / 2
        ws = rng.randn(4).astype("float32")
        def fnp(anp):
            return np.linalg.eigh(anp, UPLO="L")[0]
        def loss_np(w):
            return float(np.sum(w * ws))
        ref = self._fd_grad(fnp, a, loss_np)
        x = _to_complex64_var(a)
        x.requires_grad = True
        with jt.enable_grad():
            w, v = linalg.eigh(x)
            loss = (w.real * jt.array(ws)).sum()
            g = jt.grad(loss, [x])[0]
        got = _np(g)
        np.testing.assert_allclose(got, ref, atol=3e-2, rtol=3e-2)

    def test_qr_wide_forward(self):
        rng = np.random.RandomState(13)
        a = (rng.randn(2, 5) + 1j * rng.randn(2, 5)).astype("complex64")
        q, r = linalg.qr(_to_complex64_var(a))
        self.assertEqual(tuple(q.shape), (2, 2))
        self.assertEqual(tuple(r.shape), (2, 5))
        qn, rn = _np(q), _np(r)
        np.testing.assert_allclose(_dot(qn, rn), a, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(
            _dot(np.conj(np.swapaxes(qn, -1, -2)), qn),
            np.eye(2),
            atol=1e-3,
            rtol=1e-3,
        )

        # Wide (M<N) backward is supported too; see test_qr_wide_forward_and_backward
        # below for a gradcheck against a numpy finite-difference oracle.
        x = _to_complex64_var(a)
        x.requires_grad = True
        qb, rb = linalg.qr(x)
        loss = jt.abs(qb).sum() + jt.abs(rb).sum()
        jt.grad(loss, [x])[0].sync()

    def test_eigh_backward_eigenvector_dependent(self):
        rng = np.random.RandomState(11)
        b = (rng.randn(5, 5) + 1j * rng.randn(5, 5)).astype("complex64")
        a = (b + b.conj().T) / 2
        p = (rng.randn(5, 5) + 1j * rng.randn(5, 5)).astype("complex64") * 0.1
        def fnp(anp):
            w, v = np.linalg.eigh(anp, UPLO="L")
            return v @ np.diag(w) @ v.conj().T
        def loss_np(rec):
            return float(np.real(np.sum(rec * np.conj(p))))
        ref = self._fd_grad(fnp, a, loss_np)
        x = _to_complex64_var(a)
        x.requires_grad = True
        with jt.enable_grad():
            w, v = linalg.eigh(x)
            rec = jt.matmul(v * w.unsqueeze(-2), v.conj().transpose(-1, -2))
            loss = (rec * _to_complex64_var(p).conj()).real.sum()
            g = jt.grad(loss, [x])[0]
        got = _np(g)
        np.testing.assert_allclose(got, ref, atol=3e-2, rtol=3e-2)

    # ----------------------------------------------- qr shape/backward (jittor-core-gaps.md §3.4)
    def test_qr_tall_forward_and_backward(self):
        rng = np.random.RandomState(12)
        shape = (6, 3)
        a = (rng.randn(*shape) + 1j * rng.randn(*shape)).astype("complex64")
        z = _to_complex64_var(a)
        q, r = linalg.qr(z)
        self.assertEqual(tuple(q.shape), (6, 3))
        self.assertEqual(tuple(r.shape), (3, 3))
        qn, rn = _np(q), _np(r)
        np.testing.assert_allclose(_dot(qn, rn), a, atol=1e-3, rtol=1e-3)
        qhq = _dot(np.conj(np.swapaxes(qn, -1, -2)), qn)
        np.testing.assert_allclose(qhq, np.eye(3), atol=1e-3, rtol=1e-3)

        p = (rng.randn(*shape) + 1j * rng.randn(*shape)).astype("complex64") * 0.1
        def fnp(anp):
            qq, rr = np.linalg.qr(anp)
            return qq @ rr
        def loss_np(rec):
            return float(np.real(np.sum(rec * np.conj(p))))
        ref = self._fd_grad(fnp, a, loss_np)
        x = _to_complex64_var(a)
        x.requires_grad = True
        with jt.enable_grad():
            qb, rb = linalg.qr(x)
            rec = jt.matmul(qb, rb)
            loss = (rec * _to_complex64_var(p).conj()).real.sum()
            g = jt.grad(loss, [x])[0]
        got = _np(g)
        np.testing.assert_allclose(got, ref, atol=3e-2, rtol=3e-2)

    # jittor-core-gaps.md §3.2: complex wide (M<N) reduced QR forward used to
    # be rejected outright (`assert m >= n`), and backward for wide inputs
    # was unimplemented even where forward worked. Both are now supported;
    # gradcheck the wide-specific backward branch against a numpy
    # finite-difference oracle, since it takes a structurally different code
    # path (R split into R1|R2) from the tall/square formula already
    # covered by test_qr_tall_forward_and_backward.
    def _qr_gradcheck(self, shape, seed):
        rng = np.random.RandomState(seed)
        a = (rng.randn(*shape) + 1j * rng.randn(*shape)).astype("complex64")
        m, n = shape[-2:]
        k = min(m, n)
        p_q = (rng.randn(*(shape[:-2] + (m, k))) +
               1j * rng.randn(*(shape[:-2] + (m, k)))) * 0.1
        p_r = (rng.randn(*(shape[:-2] + (k, n))) +
               1j * rng.randn(*(shape[:-2] + (k, n)))) * 0.1

        def fnp(av):
            return np.linalg.qr(av)
        def loss_np(qr):
            q, r = qr
            return float(np.real(np.sum(q * np.conj(p_q)) + np.sum(r * np.conj(p_r))))
        ref = self._fd_grad(fnp, a, loss_np)

        x = _to_complex64_var(a)
        x.requires_grad = True
        with jt.enable_grad():
            q, r = linalg.qr(x)
            loss = ((q * _to_complex64_var(p_q).conj()).real.sum() +
                    (r * _to_complex64_var(p_r).conj()).real.sum())
            g = jt.grad(loss, [x])[0]
        np.testing.assert_allclose(_np(g), ref, atol=3e-2, rtol=3e-2,
                                   err_msg=f"qr backward vs finite-diff, shape={shape}")

    def test_qr_wide_forward_and_backward(self):
        shape = (3, 6)
        rng = np.random.RandomState(13)
        a = (rng.randn(*shape) + 1j * rng.randn(*shape)).astype("complex64")
        z = _to_complex64_var(a)
        q, r = linalg.qr(z)
        self.assertEqual(tuple(q.shape), (3, 3), "wide reduced QR: Q must be square")
        self.assertEqual(tuple(r.shape), (3, 6))
        qn, rn = _np(q), _np(r)
        np.testing.assert_allclose(_dot(qn, rn), a, atol=1e-3, rtol=1e-3)
        qhq = _dot(np.conj(np.swapaxes(qn, -1, -2)), qn)
        np.testing.assert_allclose(qhq, np.eye(3), atol=1e-3, rtol=1e-3)
        self._qr_gradcheck(shape, 13)

    def test_qr_wide_backward_batched(self):
        self._qr_gradcheck((2, 3, 6), 14)

    # ---------------------------------------------------- rq (jittor-core-gaps.md §3.2)
    # rq did not exist in this repo at all before this fix; it is implemented
    # as a composition of flip/(conjugate-)transpose/qr (see linalg.py::rq's
    # docstring), so these tests lock in the forward contract (R@Q==A,
    # Q@Q^H==I, shapes) plus an independent backward gradcheck against a
    # numpy finite-difference oracle.
    def _rq_gradcheck(self, shape, seed):
        rng = np.random.RandomState(seed)
        a = (rng.randn(*shape) + 1j * rng.randn(*shape)).astype("complex64")
        m, n = shape[-2:]
        k = min(m, n)
        p_r = (rng.randn(*(shape[:-2] + (m, k))) +
               1j * rng.randn(*(shape[:-2] + (m, k)))) * 0.1
        p_q = (rng.randn(*(shape[:-2] + (k, n))) +
               1j * rng.randn(*(shape[:-2] + (k, n)))) * 0.1

        def fnp(av):
            m_, n_ = av.shape[-2:]
            k_ = min(m_, n_)
            def one(mat):
                a_tilde = mat[::-1, ::-1]
                q0, r0 = np.linalg.qr(np.conj(a_tilde.T))
                r_ = np.conj(r0.T)[::-1, ::-1]
                q_ = np.conj(q0.T)[::-1, ::-1]
                return r_, q_
            if av.ndim == 2:
                return one(av)
            batch = av.shape[:-2]
            flat = av.reshape(-1, m_, n_)
            rs, qs = zip(*(one(flat[i]) for i in range(flat.shape[0])))
            return (np.stack(rs).reshape(batch + (m_, k_)),
                    np.stack(qs).reshape(batch + (k_, n_)))

        def loss_np(rq):
            r, q = rq
            return float(np.real(np.sum(r * np.conj(p_r)) + np.sum(q * np.conj(p_q))))
        ref = self._fd_grad(fnp, a, loss_np)

        x = _to_complex64_var(a)
        x.requires_grad = True
        with jt.enable_grad():
            r, q = linalg.rq(x)
            loss = ((r * _to_complex64_var(p_r).conj()).real.sum() +
                    (q * _to_complex64_var(p_q).conj()).real.sum())
            g = jt.grad(loss, [x])[0]
        np.testing.assert_allclose(_np(g), ref, atol=3e-2, rtol=3e-2,
                                   err_msg=f"rq backward vs finite-diff, shape={shape}")

    def _rq_forward_check(self, shape, seed):
        rng = np.random.RandomState(seed)
        a = (rng.randn(*shape) + 1j * rng.randn(*shape)).astype("complex64")
        z = _to_complex64_var(a)
        r, q = linalg.rq(z)
        m, n = shape[-2:]
        k = min(m, n)
        self.assertEqual(tuple(r.shape[-2:]), (m, k))
        self.assertEqual(tuple(q.shape[-2:]), (k, n))
        rn, qn = _np(r), _np(q)
        np.testing.assert_allclose(_dot(rn, qn), a, atol=1e-3, rtol=1e-3)
        qqh = _dot(qn, np.conj(np.swapaxes(qn, -1, -2)))
        eye = np.broadcast_to(np.eye(k), qqh.shape)
        np.testing.assert_allclose(qqh, eye, atol=1e-3, rtol=1e-3)

    def test_rq_square(self):
        self._rq_forward_check((4, 4), 20)
        self._rq_gradcheck((4, 4), 20)

    def test_rq_tall(self):
        self._rq_forward_check((6, 3), 21)
        self._rq_gradcheck((6, 3), 21)

    def test_rq_wide(self):
        self._rq_forward_check((3, 6), 22)
        self._rq_gradcheck((3, 6), 22)

    def test_rq_batched(self):
        self._rq_forward_check((2, 6, 3), 23)
        self._rq_gradcheck((2, 6, 3), 23)

    def test_svdvals(self):
        rng = np.random.RandomState(2)
        a = (rng.randn(4, 4) + 1j * rng.randn(4, 4))
        z = _to_complex64_var(a)
        s = linalg.svdvals(z)
        sn = _np(s)
        ref = np.linalg.svd(a, compute_uv=False)
        np.testing.assert_allclose(np.sort(sn.real), np.sort(ref), atol=1e-3, rtol=1e-3)

    # --------------------------------------------------------------------- qr
    def test_qr(self):
        rng = np.random.RandomState(3)
        a = (rng.randn(2, 4, 4) + 1j * rng.randn(2, 4, 4))
        z = _to_complex64_var(a)
        q, r = linalg.qr(z)
        self.assertTrue("complex" in str(q.dtype) and "complex" in str(r.dtype),
                        "qr q,r must be native complex64")
        qn, rn = _np(q), _np(r)
        # q @ r == a
        np.testing.assert_allclose(_dot(qn, rn), a, atol=1e-3, rtol=1e-3)
        # q unitary: q^H q == I
        qhq = _dot(np.conj(np.swapaxes(qn, -1, -2)), qn)
        np.testing.assert_allclose(qhq, np.broadcast_to(np.eye(4), (2, 4, 4)),
                                   atol=1e-3, rtol=1e-3)

    # -------------------------------------------------------------------- eig
    def test_eig(self):
        if self.use_cuda:
            # PRE-EXISTING platform limitation (not a regression of this bridge):
            # general non-Hermitian eig runs through jt.numpy_code, which binds
            # `np` to cupy on CUDA, and cupy.linalg has NO `eig` (only `eigh`).
            # The legacy ComplexNumber eig path is broken on CUDA for the same
            # reason; the native bridge faithfully reuses it. Verified on CPU.
            self.skipTest("cupy.linalg has no eig() (general eig is CPU-only); "
                          "pre-existing — see test_eig docstring")
        rng = np.random.RandomState(4)
        a = (rng.randn(2, 4, 4) + 1j * rng.randn(2, 4, 4))
        z = _to_complex64_var(a)
        w, v = linalg.eig(z)
        self.assertTrue("complex" in str(w.dtype) and "complex" in str(v.dtype),
                        "eig w,v must be native complex64")
        wn, vn = _np(w), _np(v)
        # a @ v == v @ diag(w)
        lhs = _dot(a, vn)
        rhs = _dot(vn, _diag_embed(wn))
        np.testing.assert_allclose(lhs, rhs, atol=1e-3, rtol=1e-3)

    # ------------------------------------------------------------------- eigh
    def test_eigh(self):
        rng = np.random.RandomState(5)
        b = (rng.randn(2, 4, 4) + 1j * rng.randn(2, 4, 4))
        a = b + np.conj(np.swapaxes(b, -1, -2))  # Hermitian
        z = _to_complex64_var(a)
        w, v = linalg.eigh(z)
        self.assertTrue("complex" in str(w.dtype) and "complex" in str(v.dtype),
                        "eigh w,v must be native complex64")
        wn, vn = _np(w), _np(v)
        # eigenvalues of a Hermitian matrix are real
        np.testing.assert_allclose(wn.imag, 0, atol=1e-3)
        # a @ v == v @ diag(w)
        lhs = _dot(a, vn)
        rhs = _dot(vn, _diag_embed(wn))
        np.testing.assert_allclose(lhs, rhs, atol=1e-3, rtol=1e-3)
        # eigh reads the LOWER triangle (UPLO='L'); compare to numpy eigh values
        ref = np.linalg.eigh(a, UPLO='L')[0]
        np.testing.assert_allclose(np.sort(wn.real, axis=-1), ref, atol=1e-3, rtol=1e-3)

    # ------------------------------------------------------------------- pinv
    def test_pinv_square(self):
        rng = np.random.RandomState(6)
        a = (rng.randn(4, 4) + 1j * rng.randn(4, 4))
        z = _to_complex64_var(a)
        p = linalg.pinv(z)
        self.assertTrue("complex" in str(p.dtype), "pinv must be native complex64")
        pn = _np(p)
        # Moore-Penrose: A P A == A
        np.testing.assert_allclose(_dot(_dot(a, pn), a), a, atol=1e-3, rtol=1e-3)

    def test_pinv_nonsquare(self):
        rng = np.random.RandomState(7)
        a = (rng.randn(2, 5) + 1j * rng.randn(2, 5))  # 2x5 -> pinv 5x2
        z = _to_complex64_var(a)
        p = linalg.pinv(z)
        pn = _np(p)
        self.assertEqual(tuple(p.shape), (5, 2))
        np.testing.assert_allclose(_dot(_dot(a, pn), a), a, atol=1e-3, rtol=1e-3)
        ref = np.linalg.pinv(a)
        np.testing.assert_allclose(pn, ref, atol=1e-3, rtol=1e-3)

    # ---------------------------------------------------- legacy CN regression
    def test_complexnumber_input_unchanged(self):
        # The legacy nn.ComplexNumber input path must still return a
        # ComplexNumber with correct values (no regression from the bridge).
        rng = np.random.RandomState(8)
        a = (rng.randn(4, 4) + 1j * rng.randn(4, 4))
        cn = jt.nn.ComplexNumber(jt.array(a.real.astype("float32")),
                                 jt.array(a.imag.astype("float32")))

        # inv
        cni = linalg.inv(cn)
        self.assertIsInstance(cni, jt.nn.ComplexNumber)
        inv_np = cni.real.numpy() + 1j * cni.imag.numpy()
        np.testing.assert_allclose(_dot(a, inv_np), np.eye(4), atol=1e-3, rtol=1e-3)

        # qr
        cq, cr = linalg.qr(cn)
        self.assertIsInstance(cq, jt.nn.ComplexNumber)
        self.assertIsInstance(cr, jt.nn.ComplexNumber)
        qn = cq.real.numpy() + 1j * cq.imag.numpy()
        rn = cr.real.numpy() + 1j * cr.imag.numpy()
        np.testing.assert_allclose(_dot(qn, rn), a, atol=1e-3, rtol=1e-3)

        # svd (returns the SVD namedtuple of ComplexNumbers)
        su, ss, sv = linalg.svd(cn)
        self.assertIsInstance(su, jt.nn.ComplexNumber)
        un = su.real.numpy() + 1j * su.imag.numpy()
        sn = ss.real.numpy() + 1j * ss.imag.numpy()
        vn = sv.real.numpy() + 1j * sv.imag.numpy()
        rec = _dot(_dot(un, _diag_embed(sn)), vn)
        np.testing.assert_allclose(rec, a, atol=1e-3, rtol=1e-3)

        # eig (CPU only: cupy.linalg has no eig — pre-existing, see test_eig)
        if not self.use_cuda:
            ew, ev = linalg.eig(cn)
            self.assertIsInstance(ew, jt.nn.ComplexNumber)
            self.assertIsInstance(ev, jt.nn.ComplexNumber)
            wn = ew.real.numpy() + 1j * ew.imag.numpy()
            evn = ev.real.numpy() + 1j * ev.imag.numpy()
            np.testing.assert_allclose(_dot(a, evn), _dot(evn, _diag_embed(wn)),
                                       atol=1e-3, rtol=1e-3)

        # eigh (Hermitian) — previously eigh(ComplexNumber) raised; now supported
        h = a + np.conj(a.T)
        cnh = jt.nn.ComplexNumber(jt.array(h.real.astype("float32")),
                                  jt.array(h.imag.astype("float32")))
        hw, hv = linalg.eigh(cnh)
        self.assertIsInstance(hw, jt.nn.ComplexNumber)
        self.assertIsInstance(hv, jt.nn.ComplexNumber)
        hwn = hw.real.numpy() + 1j * hw.imag.numpy()
        hvn = hv.real.numpy() + 1j * hv.imag.numpy()
        np.testing.assert_allclose(_dot(h, hvn), _dot(hvn, _diag_embed(hwn)),
                                   atol=1e-3, rtol=1e-3)

        # pinv — previously pinv(ComplexNumber) raised; now supported
        cp = linalg.pinv(cn)
        self.assertIsInstance(cp, jt.nn.ComplexNumber)
        pn = cp.real.numpy() + 1j * cp.imag.numpy()
        np.testing.assert_allclose(_dot(_dot(a, pn), a), a, atol=1e-3, rtol=1e-3)

    # ------------------------------------------------------- real path intact
    def test_real_path_unchanged(self):
        # real inputs must still go through the real code path unchanged.
        rng = np.random.RandomState(9)
        a = rng.randn(4, 4).astype("float32")
        z = jt.array(a)
        ri = linalg.inv(z)
        self.assertFalse("complex" in str(ri.dtype))
        np.testing.assert_allclose(_dot(a, ri.numpy()), np.eye(4), atol=1e-3, rtol=1e-3)
        # symmetric for eigh
        s = a + a.T
        w, v = linalg.eigh(jt.array(s))
        self.assertFalse("complex" in str(w.dtype))
        ref = np.linalg.eigh(s, UPLO='L')[0]
        np.testing.assert_allclose(np.sort(w.numpy()), ref, atol=1e-3, rtol=1e-3)


@unittest.skipIf(not jt.has_cuda, "no cuda found")
class TestComplex64LinalgCUDA(_Mixin, unittest.TestCase):
    use_cuda = 1


class TestComplex64LinalgCPU(_Mixin, unittest.TestCase):
    use_cuda = 0


if __name__ == "__main__":
    unittest.main()

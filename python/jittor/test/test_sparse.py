# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Xiangli Li <1905692338@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# Native COO/CSR sparse tensors and sparse-dense matmul (jittor-core-gaps.md SS3.6).
#
# jittor.sparse.{SparseVar, SparseCSR, spmm} are backed by jt.numpy_code / jt.Function
# bound to scipy.sparse (CPU) / cupyx.scipy.sparse (CUDA, cuSPARSE) -- see sparse.py's
# module docstring for the full design rationale (same pattern as the native complex64/128
# linalg ops). This suite locks: create, to_dense duplicate-summing round-trip, coalesce,
# COO<->CSR conversion, transpose, spmm forward + backward (values AND the dense operand,
# with indices carrying no gradient), float32/64 and complex64/128, empty/zero-nnz,
# non-square, uncoalesced input correctness, and that spmm never densifies (M,N).
#
# Run:  python -m jittor.test.test_sparse
import unittest
import numpy as np
import jittor as jt
from jittor import sparse

_DEVICES = [("cpu", 0)] + ([("cuda", 1)] if jt.has_cuda else [])


def both_devices(fn):
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


def _np_dtype(dtype, rng, n, extra_shape=()):
    shape = (n,) + extra_shape
    if "complex" in dtype:
        return (rng.randn(*shape) + 1j * rng.randn(*shape)).astype(dtype)
    return rng.randn(*shape).astype(dtype)


def _to_sparsevar(row, col, values, shape):
    return sparse.sparse_array(jt.stack([jt.array(row), jt.array(col)], dim=0),
                                jt.array(values), jt.NanoVector(list(shape)))


def _dense_ref(row, col, values, shape):
    A = np.zeros(shape, dtype=values.dtype)
    for e in range(len(values)):
        A[row[e], col[e]] += values[e]
    return A


class TestSparseCreateAndConvert(unittest.TestCase):
    def test_to_dense_sums_duplicates(self):
        # torch.sparse_coo_tensor semantics: uncoalesced duplicate (row,col) entries sum.
        row = np.array([0, 1, 1], dtype="int32")
        col = np.array([2, 0, 0], dtype="int32")
        values = np.array([3.0, 4.0, 5.0], dtype="float32")

        def body(dev):
            sp = _to_sparsevar(row, col, values, (2, 3))
            got = sp.to_dense().numpy()
            ref = _dense_ref(row, col, values, (2, 3))
            np.testing.assert_allclose(got, ref, err_msg=f"to_dense duplicates {dev}")
        both_devices(body)

    def test_transpose(self):
        row = np.array([0, 1, 1], dtype="int32")
        col = np.array([2, 0, 2], dtype="int32")
        values = np.array([3.0, 4.0, 5.0], dtype="float32")

        def body(dev):
            sp = _to_sparsevar(row, col, values, (2, 3))
            t = sp.t()
            self.assertEqual(tuple(t.shape), (3, 2), f"t() shape {dev}")
            np.testing.assert_allclose(t.to_dense().numpy(), sp.to_dense().numpy().T,
                                       err_msg=f"t() value {dev}")
        both_devices(body)

    def test_coalesce_merges_and_sums(self):
        row = np.array([0, 0, 1], dtype="int32")
        col = np.array([1, 1, 0], dtype="int32")
        values = np.array([1.0, 2.0, 5.0], dtype="float32")

        def body(dev):
            sp = _to_sparsevar(row, col, values, (2, 2))
            c = sp.coalesce()
            self.assertEqual(c.nnz, 2, f"coalesce nnz {dev}")
            np.testing.assert_allclose(c.to_dense().numpy(), sp.to_dense().numpy(),
                                       err_msg=f"coalesce value {dev}")
        both_devices(body)

    def test_coalesce_backward_gathers_to_duplicates(self):
        # each original duplicate entry receives the SAME gradient as its merged target.
        row = np.array([0, 0, 1], dtype="int32")
        col = np.array([1, 1, 0], dtype="int32")
        values = np.array([1.0, 2.0, 5.0], dtype="float32")

        def body(dev):
            v = jt.array(values)
            v.requires_grad = True
            sp = sparse.sparse_array(jt.stack([jt.array(row), jt.array(col)], dim=0), v,
                                     jt.NanoVector([2, 2]))
            with jt.enable_grad():
                c = sp.coalesce()
                loss = (c.values * jt.array([10.0, 100.0])).sum()
                g = jt.grad(loss, [v])[0]
            # entries 0,1 both merge into row0/col1 (weight 10); entry 2 -> weight 100.
            np.testing.assert_allclose(g.numpy(), np.array([10.0, 10.0, 100.0], dtype="float32"),
                                       err_msg=f"coalesce backward {dev}")
        both_devices(body)

    def test_to_csr_round_trip_no_dedup(self):
        row = np.array([1, 0, 1, 0], dtype="int32")
        col = np.array([0, 1, 1, 0], dtype="int32")
        values = np.array([10.0, 20.0, 30.0, 40.0], dtype="float32")

        def body(dev):
            sp = _to_sparsevar(row, col, values, (2, 2))
            csr = sp.to_csr()
            self.assertEqual(csr.nnz, 4, f"to_csr preserves nnz (no auto-coalesce) {dev}")
            np.testing.assert_allclose(csr.to_dense().numpy(), sp.to_dense().numpy(),
                                       err_msg=f"coo->csr->dense {dev}")
            back = csr.to_coo()
            np.testing.assert_allclose(back.to_dense().numpy(), sp.to_dense().numpy(),
                                       err_msg=f"coo->csr->coo->dense round trip {dev}")
        both_devices(body)

    def test_csr_transpose_and_coalesce(self):
        row = np.array([0, 0, 1], dtype="int32")
        col = np.array([1, 1, 0], dtype="int32")
        values = np.array([1.0, 2.0, 5.0], dtype="float32")

        def body(dev):
            csr = _to_sparsevar(row, col, values, (2, 2)).to_csr()
            t = csr.t()
            self.assertEqual(tuple(t.shape), (2, 2), f"csr t() shape {dev}")
            np.testing.assert_allclose(t.to_dense().numpy(), csr.to_dense().numpy().T,
                                       err_msg=f"csr t() value {dev}")
            c = csr.coalesce()
            self.assertEqual(c.nnz, 2, f"csr coalesce nnz {dev}")
        both_devices(body)


class TestSpmm(unittest.TestCase):
    def _check_forward(self, dev, dtype, layout, M, N, K, seed):
        rng = np.random.RandomState(seed)
        nnz = min(M * N, 6)
        row = rng.randint(0, M, size=nnz).astype("int32")
        col = rng.randint(0, N, size=nnz).astype("int32")
        values = _np_dtype(dtype, rng, nnz)
        y = _np_dtype(dtype, rng, N, extra_shape=(K,))
        sp = _to_sparsevar(row, col, values, (M, N))
        if layout == "csr":
            sp = sp.to_csr()
        out = sparse.spmm(sp, jt.array(y))
        ref = _dense_ref(row, col, values, (M, N)) @ y
        tol = 1e-6 if "128" in dtype or "64" == dtype[-2:] else 3e-3
        np.testing.assert_allclose(out.numpy(), ref, atol=tol, rtol=tol,
                                   err_msg=f"spmm forward {dev} {dtype} {layout}")

    def test_forward_matrix(self):
        for dtype in ["float32", "float64", "complex64", "complex128"]:
            for layout in ["coo", "csr"]:
                def body(dev, dtype=dtype, layout=layout):
                    self._check_forward(dev, dtype, layout, 5, 6, 3, 0)
                both_devices(body)

    def test_forward_nonsquare(self):
        def body(dev):
            self._check_forward(dev, "float32", "coo", 3, 7, 2, 1)
            self._check_forward(dev, "float32", "coo", 7, 3, 2, 2)
        both_devices(body)

    def test_forward_vector_rhs(self):
        # K=1 (spmv-shaped).
        def body(dev):
            self._check_forward(dev, "float32", "coo", 4, 4, 1, 3)
        both_devices(body)

    def _fd_grad_values(self, row, col, values, y, dtype, eps=1e-4):
        M = int(row.max()) + 1 if len(row) else 1
        def loss_np(v):
            A = _dense_ref(row, col, v, (M, y.shape[0]))
            r = A @ y
            return float(np.real(r).sum()) if "complex" in dtype else float(r.sum())
        g = np.zeros_like(values)
        for e in range(len(values)):
            vp = values.copy(); vp[e] += eps
            vm = values.copy(); vm[e] -= eps
            g[e] = (loss_np(vp) - loss_np(vm)) / (2 * eps)
            if "complex" in dtype:
                vpi = values.copy(); vpi[e] += 1j * eps
                vmi = values.copy(); vmi[e] -= 1j * eps
                g[e] += 1j * (loss_np(vpi) - loss_np(vmi)) / (2 * eps)
        return g

    def _fd_grad_y(self, row, col, values, y, dtype, eps=1e-4):
        M, N = int(row.max()) + 1 if len(row) else 1, y.shape[0]
        A = _dense_ref(row, col, values, (M, N))
        def loss_np(yv):
            r = A @ yv
            return float(np.real(r).sum()) if "complex" in dtype else float(r.sum())
        g = np.zeros_like(y)
        for idx in np.ndindex(*y.shape):
            yp = y.copy(); yp[idx] += eps
            ym = y.copy(); ym[idx] -= eps
            g[idx] = (loss_np(yp) - loss_np(ym)) / (2 * eps)
            if "complex" in dtype:
                ypi = y.copy(); ypi[idx] += 1j * eps
                ymi = y.copy(); ymi[idx] -= 1j * eps
                g[idx] += 1j * (loss_np(ypi) - loss_np(ymi)) / (2 * eps)
        return g

    def _check_backward(self, dev, dtype, layout, seed):
        rng = np.random.RandomState(seed)
        M, N, K = 4, 5, 3
        row = np.array([0, 1, 2, 3, 0], dtype="int32")
        col = np.array([1, 2, 3, 4, 4], dtype="int32")
        values = _np_dtype(dtype, rng, 5)
        y = _np_dtype(dtype, rng, N, extra_shape=(K,))

        jval = jt.array(values); jval.requires_grad = True
        jy = jt.array(y); jy.requires_grad = True
        sp = sparse.sparse_array(jt.stack([jt.array(row), jt.array(col)], dim=0), jval,
                                 jt.NanoVector([M, N]))
        if layout == "csr":
            # re-wrap so jval stays the SAME leaf Var the CSR values array is a view of
            # (to_csr() row-sorts but keeps values traced back to jval via numpy_code).
            sp = sp.to_csr()
        with jt.enable_grad():
            out = sparse.spmm(sp, jy)
            loss = out.real.sum() if "complex" in dtype else out.sum()
            gv, gy = jt.grad(loss, [jval, jy])

        tol = 1e-6 if dtype in ("float64", "complex128") else 3e-2
        # jittor-core-gaps.md §3.6 names, by shape, exactly the regression this
        # locks: to_dense()'s split indices used to keep a (1, nnz) leading
        # axis, so the values gradient came back (1, nnz) instead of (nnz,).
        # assert_allclose alone would silently broadcast that away.
        self.assertEqual(tuple(gv.shape), values.shape,
                          f"spmm backward values grad shape {dev} {dtype} {layout}")
        self.assertEqual(tuple(gy.shape), y.shape,
                          f"spmm backward y grad shape {dev} {dtype} {layout}")
        gv_ref = self._fd_grad_values(row, col, values, y, dtype)
        gy_ref = self._fd_grad_y(row, col, values, y, dtype)
        np.testing.assert_allclose(gv.numpy(), gv_ref, atol=tol, rtol=tol,
                                   err_msg=f"spmm backward values {dev} {dtype} {layout}")
        np.testing.assert_allclose(gy.numpy(), gy_ref, atol=tol, rtol=tol,
                                   err_msg=f"spmm backward y {dev} {dtype} {layout}")

    def test_backward(self):
        for dtype in ["float32", "float64", "complex64", "complex128"]:
            for layout in ["coo", "csr"]:
                def body(dev, dtype=dtype, layout=layout):
                    self._check_backward(dev, dtype, layout, 0)
                both_devices(body)

    def test_indices_carry_no_gradient(self):
        # int-dtype indices must never be asked for a gradient (and must not error if the
        # caller tries jt.grad on values/y only -- the framework itself skips int inputs).
        row = jt.array(np.array([0, 1], dtype="int32"))
        col = jt.array(np.array([1, 0], dtype="int32"))

        def body(dev):
            values = jt.array(np.array([2.0, 3.0], dtype="float32"))
            values.requires_grad = True
            y = jt.array(np.array([[5.0], [7.0]], dtype="float32"))
            sp = sparse.sparse_array(jt.stack([row, col], dim=0), values, jt.NanoVector([2, 2]))
            with jt.enable_grad():
                out = sparse.spmm(sp, y)
                loss = out.sum()
                gv = jt.grad(loss, [values])[0]
            self.assertTrue(np.isfinite(gv.numpy()).all(), f"grad finite {dev}")
        both_devices(body)

    def test_indices_gradient_request_fails_loud(self):
        # jittor-core-gaps.md §3.6 criterion 3 asks for an EXPLICIT
        # stop-gradient contract on indices, not just "values gradient
        # happens to be finite". Indices are int32 -- jt.grad's own target
        # dtype check (grad.cc) rejects a non-float/complex grad target, so
        # requesting d(loss)/d(indices) must raise, not silently return None
        # or a zero/garbage Var.
        row = jt.array(np.array([0, 1], dtype="int32"))
        col = jt.array(np.array([1, 0], dtype="int32"))

        def body(dev):
            values = jt.array(np.array([2.0, 3.0], dtype="float32"))
            values.requires_grad = True
            y = jt.array(np.array([[5.0], [7.0]], dtype="float32"))
            indices = jt.stack([row, col], dim=0)
            sp = sparse.sparse_array(indices, values, jt.NanoVector([2, 2]))
            with jt.enable_grad():
                out = sparse.spmm(sp, y)
                loss = out.sum()
                with self.assertRaises(Exception, msg=f"grad wrt indices should fail loud {dev}"):
                    jt.grad(loss, [indices])
        both_devices(body)

    def test_uncoalesced_csr_still_correct(self):
        # duplicate (row,col) entries in an UNCOALESCED CSR must still sum correctly during
        # spmm (scipy/cupyx sum contributions naturally; coalescing is not required for
        # spmm correctness, only to_dense's dedup semantics need coalesce()).
        row = np.array([0, 0, 1], dtype="int32")
        col = np.array([1, 1, 0], dtype="int32")
        values = np.array([1.0, 2.0, 3.0], dtype="float32")

        def body(dev):
            csr = _to_sparsevar(row, col, values, (2, 2)).to_csr()
            self.assertEqual(csr.nnz, 3, f"uncoalesced csr keeps nnz {dev}")
            y = jt.array(np.random.RandomState(4).randn(2, 2).astype("float32"))
            out = sparse.spmm(csr, y)
            ref = _dense_ref(row, col, values, (2, 2)) @ y.numpy()
            np.testing.assert_allclose(out.numpy(), ref, atol=1e-5,
                                       err_msg=f"uncoalesced csr spmm {dev}")
        both_devices(body)

    def test_empty_nnz(self):
        def body(dev):
            idx = jt.empty((2, 0), "int32")
            values = jt.array(np.array([], dtype="float32"))
            sp = sparse.sparse_array(idx, values, jt.NanoVector([3, 4]))
            dense = sp.to_dense()
            self.assertEqual(tuple(dense.shape), (3, 4), f"empty to_dense shape {dev}")
            self.assertEqual(dense.numpy().sum(), 0.0, f"empty to_dense value {dev}")
            y = jt.array(np.random.randn(4, 2).astype("float32"))
            out = sparse.spmm(sp, y)
            self.assertEqual(tuple(out.shape), (3, 2), f"empty spmm shape {dev}")
            self.assertEqual(out.numpy().sum(), 0.0, f"empty spmm value {dev}")
            t = sp.t()
            self.assertEqual(tuple(t.indices.shape), (2, 0), f"empty t() indices shape {dev}")
            csr = sp.to_csr()
            np.testing.assert_array_equal(csr.crow_indices.numpy(), np.zeros(4, dtype="int32"),
                                          err_msg=f"empty to_csr crow {dev}")
            self.assertEqual(sp.coalesce().nnz, 0, f"empty coalesce nnz {dev}")
        both_devices(body)

    def test_degenerate_empty_shape_matrix(self):
        # genuinely degenerate (M=0 or N=0) shapes, distinct from
        # test_empty_nnz's zero-nnz-but-normal-shape case.
        def body(dev):
            idx = jt.empty((2, 0), "int32")
            values = jt.array(np.array([], dtype="float32"))
            sp = sparse.sparse_array(idx, values, jt.NanoVector([0, 4]))
            dense = sp.to_dense()
            self.assertEqual(tuple(dense.shape), (0, 4), f"(0,N) to_dense shape {dev}")
            y = jt.array(np.random.RandomState(6).randn(4, 2).astype("float32"))
            out = sparse.spmm(sp, y)
            self.assertEqual(tuple(out.shape), (0, 2), f"(0,N) spmm shape {dev}")

            sp2 = sparse.sparse_array(idx, values, jt.NanoVector([4, 0]))
            dense2 = sp2.to_dense()
            self.assertEqual(tuple(dense2.shape), (4, 0), f"(M,0) to_dense shape {dev}")
            y2 = jt.array(np.random.RandomState(7).randn(0, 2).astype("float32"))
            out2 = sparse.spmm(sp2, y2)
            self.assertEqual(tuple(out2.shape), (4, 2), f"(M,0) spmm shape {dev}")
            self.assertEqual(out2.numpy().sum(), 0.0, f"(M,0) spmm value {dev}")
        both_devices(body)

    def test_no_densify_large_shape(self):
        # (M,N) here is far too large to ever materialize as a dense array; spmm/to_dense
        # only touch O(nnz)/O(nnz*K) buffers, so this must complete quickly regardless.
        def body(dev):
            M, N, K, nnz = 200_000, 200_000, 2, 8
            rng = np.random.RandomState(5)
            row = rng.randint(0, M, size=nnz).astype("int32")
            col = rng.randint(0, N, size=nnz).astype("int32")
            values = rng.randn(nnz).astype("float32")
            sp = _to_sparsevar(row, col, values, (M, N))
            y = jt.array(rng.randn(N, K).astype("float32"))
            out = sparse.spmm(sp, y)
            ref = _dense_ref(row, col, values, (M, N)) @ y.numpy()
            np.testing.assert_allclose(out.numpy(), ref, atol=1e-3, rtol=1e-3,
                                       err_msg=f"no-densify large-shape spmm {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)

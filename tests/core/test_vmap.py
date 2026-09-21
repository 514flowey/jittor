# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Jittor Group
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# Native jt.vmap (python/jittor/vmap.py): builds one real batched op graph
# instead of looping over the batch in Python. This repo previously had no
# native vmap at all -- only an approximate, CPU-only, loop/stack-based
# `torch.vmap` compat shim in a completely separate namespace
# (compat/torch/installers/numerical.py).
import unittest

import numpy as np

import jittor as jt


class TestVmap(unittest.TestCase):
    def test_elementwise(self):
        x = jt.array(np.random.RandomState(0).randn(5, 3).astype("float32"))
        out = jt.vmap(lambda v: jt.sin(v) * 2 + 1)(x)
        ref = np.sin(x.numpy()) * 2 + 1
        np.testing.assert_allclose(out.numpy(), ref, atol=1e-5)

    def test_reduce_and_grad_composition(self):
        # vmap(f) where f reduces to a per-example scalar, then jt.grad
        # through the whole batched graph, must give each example its own
        # gradient (not the sum collapsed across the batch).
        x = jt.array(np.random.RandomState(1).randn(4, 3).astype("float32"))
        x.requires_grad = True
        y = jt.vmap(lambda v: (v * v).sum())(x)
        self.assertEqual(tuple(y.shape), (4,))
        g = jt.grad(y.sum(), [x])[0]
        np.testing.assert_allclose(g.numpy(), 2 * x.numpy(), atol=1e-5)

    def test_matmul_with_shared_weight(self):
        # The single most common vmap pattern: an unbatched (shared) weight
        # matrix applied to a batched vector. Plain jt.matmul broadcasting
        # has no notion of "this leading axis is a batch axis I introduced",
        # so this needs the dedicated matmul batching rule (_apply_matmul),
        # not just generic broadcasting.
        w = jt.array(np.random.RandomState(2).randn(3, 3).astype("float32"))
        xb = jt.array(np.random.RandomState(3).randn(6, 3).astype("float32"))
        out = jt.vmap(lambda v: jt.matmul(w, v))(xb)
        ref = np.einsum("ij,bj->bi", w.numpy(), xb.numpy())
        np.testing.assert_allclose(out.numpy(), ref, atol=1e-4)

    def test_nested_vmap(self):
        x = jt.array(np.random.RandomState(4).randn(2, 3, 4).astype("float32"))
        out = jt.vmap(jt.vmap(lambda v: v * 2 + 1))(x)
        np.testing.assert_allclose(out.numpy(), x.numpy() * 2 + 1, atol=1e-5)

    def test_getitem_under_vmap(self):
        y = jt.array(np.random.RandomState(5).randn(5, 6).astype("float32"))
        out = jt.vmap(lambda v: v[1:4])(y)
        np.testing.assert_allclose(out.numpy(), y.numpy()[:, 1:4])

    def test_qr_batching_rule(self):
        # qr/svd/svdvals/eigh/inv are already (..., M, N)-batched numpy_code
        # ops with no op-specific batch-axis handling of their own, so
        # vmap's extra leading physical axis should be accepted as just one
        # more leading batch dim for free (_apply_linalg).
        a = jt.array(np.random.RandomState(6).randn(3, 4, 4).astype("float32"))

        def qrf(x):
            return jt.linalg.qr(x)

        q, r = jt.vmap(qrf)(a)
        self.assertEqual(tuple(q.shape), (3, 4, 4))
        recon = np.einsum("bij,bjk->bik", q.numpy(), r.numpy())
        np.testing.assert_allclose(recon, a.numpy(), atol=1e-3, rtol=1e-3)

    def test_random_different_gives_independent_draws(self):
        r = jt.vmap(lambda v: v + jt.random(v.shape), randomness="different")(
            jt.zeros((8, 3))
        )
        rn = r.numpy()
        self.assertFalse(np.allclose(rn[0], rn[1]))

    def test_random_same_gives_identical_draws(self):
        r = jt.vmap(lambda v: v + jt.random(v.shape), randomness="same")(
            jt.zeros((8, 3))
        )
        rn = r.numpy()
        np.testing.assert_array_equal(rn[0], rn[1])

    def test_unmapped_call_is_unaffected(self):
        # install_batching_patches() monkeypatches jt.Var/jt entry points
        # process-wide the first time jt.vmap() is used; a call with no
        # BatchedVar involved must still route to the same underlying op
        # (not silently get misdispatched) -- allclose, not exact equality,
        # since jt.matmul's own backend (MKL here) legitimately differs from
        # numpy's BLAS at the ~1e-7 float32 ulp level regardless of vmap.
        x = jt.array(np.random.RandomState(7).randn(4, 4).astype("float32"))
        y = jt.array(np.random.RandomState(8).randn(4, 4).astype("float32"))
        np.testing.assert_array_equal((x + y).numpy(), x.numpy() + y.numpy())
        np.testing.assert_allclose(
            jt.matmul(x, y).numpy(), x.numpy() @ y.numpy(), atol=1e-5, rtol=1e-5
        )

    def test_unsupported_op_raises(self):
        with self.assertRaises(NotImplementedError):
            jt.vmap(lambda v: v.cumsum(0))(
                jt.array(np.zeros((3, 4), dtype="float32"))
            )


if __name__ == "__main__":
    unittest.main()

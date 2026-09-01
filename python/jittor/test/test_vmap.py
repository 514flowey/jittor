# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# Tests for jt.vmap (jittor-core-gaps.md section 3.5): a real batching
# transform that builds one batched op graph, instead of the loop-based
# fallback in torch_compat.py's vmap. Core-set scope agreed with the user:
# elementwise unary/binary, reduce, reshape/transpose/broadcast, matmul/
# einsum, getitem/setitem (basic + simple fancy indexing), and random.
import unittest
import numpy as np
import jittor as jt


class TestVmap(unittest.TestCase):
    def test_basic_elementwise(self):
        def f(x):
            return jt.sin(x) * 2 + 1
        x = jt.array(np.arange(12, dtype=np.float32).reshape(4, 3))
        out = jt.vmap(f)(x)
        np.testing.assert_allclose(out.numpy(), np.sin(x.numpy()) * 2 + 1, atol=1e-5)

    def test_install_preserves_plain_transpose_calling_conventions(self):
        x_numpy = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        x = jt.array(x_numpy)
        jt.vmap(lambda a: a)(x).sync()

        np.testing.assert_array_equal(
            x.transpose(-1, -2).numpy(), x_numpy.swapaxes(-1, -2)
        )
        np.testing.assert_array_equal(
            x.transpose((2, 0, 1)).numpy(), x_numpy.transpose((2, 0, 1))
        )

    def test_reduce_dim(self):
        def f(x):
            return x.sum(dim=0)
        x = jt.array(np.arange(12, dtype=np.float32).reshape(4, 3))
        out = jt.vmap(f)(x)
        np.testing.assert_allclose(out.numpy(), x.numpy().sum(axis=1))

    def test_reduce_full_spares_batch_axis(self):
        # .sum() with no dim must reduce only the logical axes, never the
        # batch axis -- a real bug caught during development.
        def f(x):
            return x.sum()
        x = jt.array(np.arange(20, dtype=np.float32).reshape(4, 5))
        out = jt.vmap(f)(x)
        self.assertEqual(list(out.shape), [4])
        np.testing.assert_allclose(out.numpy(), x.numpy().sum(axis=1))

    def test_rank_mismatch_binary_op(self):
        # jt.sin(x)*2 has per-sample rank 1, x.sum(dim=0) has per-sample rank
        # 0 -- combining them requires aligning logical rank before physical
        # broadcasting (another real bug caught during development).
        def f(x):
            return jt.sin(x) * 2 + x.sum(dim=0)
        x = jt.array(np.random.randn(4, 3).astype(np.float32))
        out = jt.vmap(f)(x)
        expected = np.sin(x.numpy()) * 2 + x.numpy().sum(axis=1, keepdims=True)
        np.testing.assert_allclose(out.numpy(), expected, atol=1e-5)

    def test_matmul_broadcasts_unbatched_operand(self):
        def g(x, w):
            return jt.matmul(x, w)
        xb = jt.array(np.random.randn(5, 3, 4).astype(np.float32))
        w = jt.array(np.random.randn(4, 2).astype(np.float32))
        out = jt.vmap(g, in_dims=(0, None))(xb, w)
        np.testing.assert_allclose(out.numpy(), np.matmul(xb.numpy(), w.numpy()), atol=1e-4)

    def test_in_dims_none_and_multi_arg(self):
        def g(x, scale):
            return x * scale
        x = jt.array(np.arange(12, dtype=np.float32).reshape(4, 3))
        scale = jt.array(np.float32(2.0))
        out = jt.vmap(g, in_dims=(0, None))(x, scale)
        np.testing.assert_allclose(out.numpy(), x.numpy() * 2.0)

    def test_out_dims(self):
        def h(x):
            return x * 2
        x = jt.array(np.arange(12, dtype=np.float32).reshape(4, 3))
        out = jt.vmap(h, out_dims=1)(x)
        self.assertEqual(list(out.shape), [3, 4])
        np.testing.assert_allclose(out.numpy(), (x.numpy() * 2).T)

    def test_scalar_per_sample_output(self):
        def scalarize(inp):
            return inp.sum()
        x = jt.array(np.arange(20, dtype=np.float32).reshape(4, 5))
        out = jt.vmap(scalarize)(x)
        np.testing.assert_allclose(out.numpy(), x.numpy().sum(axis=1))

    def test_output_not_depending_on_batched_input(self):
        def const_out(inp):
            return jt.array(np.float32(7.0))
        x = jt.array(np.arange(20, dtype=np.float32).reshape(4, 5))
        out = jt.vmap(const_out)(x)
        np.testing.assert_allclose(out.numpy(), np.full(4, 7.0))

    def test_pytree_dict_input_output(self):
        def combine(d):
            return {"total": d["a"] + d["b"], "doubled": d["a"] * 2}
        A = jt.array(np.arange(4, dtype=np.float32))
        B = jt.array(np.arange(4, dtype=np.float32) * 10)
        out = jt.vmap(combine)({"a": A, "b": B})
        self.assertIsInstance(out, dict)
        np.testing.assert_allclose(out["total"].numpy(), A.numpy() + B.numpy())
        np.testing.assert_allclose(out["doubled"].numpy(), A.numpy() * 2)

    def test_multi_output_tuple(self):
        def multi_out(inp):
            values, _indices = inp.max(dim=0)
            return inp.sum(), values
        x = jt.array(np.arange(20, dtype=np.float32).reshape(4, 5))
        total, mx = jt.vmap(multi_out)(x)
        np.testing.assert_allclose(total.numpy(), x.numpy().sum(axis=1))
        np.testing.assert_allclose(mx.numpy(), x.numpy().max(axis=1))

    def test_getitem_basic_slice(self):
        def f(x):
            return x[1:3]
        x = jt.array(np.arange(20, dtype=np.float32).reshape(4, 5))
        out = jt.vmap(f)(x)
        np.testing.assert_allclose(out.numpy(), x.numpy()[:, 1:3])

    def test_setitem_basic(self):
        def g(x):
            x[0] = 99.0
            return x
        x = jt.array(np.arange(20, dtype=np.float32).reshape(4, 5))
        out = jt.vmap(g)(x.clone())
        expected = x.numpy().copy()
        expected[:, 0] = 99.0
        np.testing.assert_allclose(out.numpy(), expected)

    def test_random_gives_independent_samples_per_batch_element(self):
        def rand_fn(inp):
            return inp + jt.random(inp.shape)
        x = jt.array(np.arange(20, dtype=np.float32).reshape(4, 5))
        out = jt.vmap(rand_fn)(x)
        diffs = out.numpy() - x.numpy()
        self.assertFalse(np.allclose(diffs[0], diffs[1]))

    def test_unsupported_op_fails_loud(self):
        def unsupported(inp):
            return jt.nn.conv2d(inp.reshape(1, 1, 4, 5), jt.random((1, 1, 3, 3)))
        x = jt.array(np.arange(80, dtype=np.float32).reshape(4, 20))
        with self.assertRaises(Exception):
            jt.vmap(unsupported)(x)

    def test_grad_of_sum_of_vmap(self):
        def sq(x):
            return x * x
        xv = jt.array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        xv.requires_grad = True
        out = jt.vmap(sq)(xv)
        g = jt.grad(out.sum(), xv)
        np.testing.assert_allclose(g.numpy(), 2 * xv.numpy())

    def test_vmap_of_grad_per_sample_gradients(self):
        def per_sample_loss(x):
            return (x * x).sum()
        X = jt.array(np.random.randn(4, 3).astype(np.float32))
        X.requires_grad = True
        def grad_fn(x):
            return jt.grad(per_sample_loss(x), x)
        per_sample_grads = jt.vmap(grad_fn)(X)
        np.testing.assert_allclose(per_sample_grads.numpy(), 2 * X.numpy(), atol=1e-5)

    def test_nested_vmap_elementwise(self):
        def add1(x):
            return x + 1
        Y = jt.array(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
        out = jt.vmap(jt.vmap(add1))(Y)
        np.testing.assert_allclose(out.numpy(), Y.numpy() + 1)

    def test_nested_vmap_with_reduce(self):
        # Exercises the nesting-depth-aware axis shift for axis-sensitive ops:
        # a naive "+1 per recursion level" is wrong once ops care about which
        # physical axis is "the batch axis" (see batch_transform.py docstring).
        def rowsum(x):
            return x.sum(dim=-1)
        Y = jt.array(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
        out = jt.vmap(jt.vmap(rowsum))(Y)
        np.testing.assert_allclose(out.numpy(), Y.numpy().sum(axis=-1))

    def test_compile_count_independent_of_batch_size(self):
        def f(x):
            return jt.sin(x) * 2 + x.sum(dim=0)
        with jt.profile_scope() as rep_small:
            jt.vmap(f)(jt.array(np.random.randn(4, 3).astype(np.float32))).sync()
        with jt.profile_scope() as rep_big:
            jt.vmap(f)(jt.array(np.random.randn(400, 3).astype(np.float32))).sync()
        self.assertEqual(len(rep_small), len(rep_big))


@unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
class TestVmapCuda(TestVmap):
    def setUp(self):
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.flags.use_cuda = 0


if __name__ == "__main__":
    unittest.main()

# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# Tests for jt.vmap (jittor-core-gaps.md section 3.5): a real batching
# transform that builds one batched op graph, instead of the loop-based
# fallback in torch_compat.py's vmap. Core-set scope agreed with the user:
# elementwise unary/binary, reduce, cast, reshape/transpose/broadcast,
# matmul/einsum, getitem/setitem (basic + simple fancy indexing), and random.
import unittest
import numpy as np
import jittor as jt
from jittor.gradfunctional import jvp, vjp


class TestVmap(unittest.TestCase):
    def test_basic_elementwise(self):
        def f(x):
            return jt.sin(x) * 2 + 1
        x = jt.array(np.arange(12, dtype=np.float32).reshape(4, 3))
        out = jt.vmap(f)(x)
        np.testing.assert_allclose(out.numpy(), np.sin(x.numpy()) * 2 + 1, atol=1e-5)

    def test_cast_and_astype(self):
        casts = (
            lambda value, dtype: value.cast(dtype),
            lambda value, dtype: value.astype(dtype),
            lambda value, dtype: jt.cast(value, dtype),
        )
        pairs = (
            ("float32", "float64"),
            ("float32", "complex64"),
            ("float64", "complex128"),
            ("complex64", "complex128"),
            ("complex64", "float32"),
            ("int32", "float32"),
            ("bool", "float32"),
        )
        with jt.flag_scope(auto_convert_64_to_32=0):
            for source, target in pairs:
                data = (np.arange(8).reshape(4, 2) / 4 - 0.5).astype(source)
                if "complex" in source:
                    data += 0.25j
                expected = (
                    data.real
                    if "complex" in source and "complex" not in target
                    else data
                )
                expected = expected.astype(target)
                for entry, cast in enumerate(casts):
                    with self.subTest(source=source, target=target, entry=entry):
                        calls = []

                        def f(value):
                            calls.append(tuple(value.shape))
                            return cast(value, target)

                        x = jt.array(data, dtype=source)
                        result = jt.vmap(f)(x)
                        self.assertEqual(calls, [(2,)])
                        self.assertEqual(str(result.dtype), target)
                        np.testing.assert_array_equal(result.numpy(), expected)
                        # Installing batching must not change plain Var casts.
                        plain = cast(x, target)
                        self.assertEqual(str(plain.dtype), target)
                        np.testing.assert_array_equal(plain.numpy(), expected)

    def test_cast_keyword_arguments_after_install(self):
        data = np.arange(6, dtype=np.float32).reshape(2, 3)
        x = jt.array(data)
        jt.vmap(lambda value: value)(x).sync()
        casts = (
            lambda value: value.cast(op="float64"),
            lambda value: value.astype(op="float64"),
            lambda value: jt.cast(value, op="float64"),
            lambda value: jt.Var.cast(value, op="float64"),
        )
        for entry, cast in enumerate(casts):
            with self.subTest(entry=entry):
                for result in (cast(x), jt.vmap(cast)(x)):
                    self.assertEqual(str(result.dtype), "float64")
                    np.testing.assert_array_equal(
                        result.numpy(), data.astype("float64")
                    )

    def test_cast_nested_vmap_and_axes(self):
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 8
        calls = []

        def f(value):
            calls.append(tuple(value.shape))
            return value.astype("complex128")

        inner = jt.vmap(f, in_dims=-1, out_dims=0)
        result = jt.vmap(inner, in_dims=1, out_dims=-1)(jt.array(data))
        self.assertEqual(calls, [(2,)])
        self.assertEqual(str(result.dtype), "complex128")
        np.testing.assert_array_equal(
            result.numpy(), data.transpose(2, 0, 1).astype("complex128")
        )

    def test_cast_gradients_and_vjp(self):
        # Complex-width casts above are forward-only: native UnaryOp does
        # not yet provide their backward. Batching must retain, not extend,
        # the native differentiability contract.
        pairs = (
            ("float32", "float64"),
            ("float32", "complex64"),
            ("float64", "complex128"),
            ("complex64", "float32"),
        )
        with jt.flag_scope(auto_convert_64_to_32=0):
            for source, target in pairs:
                with self.subTest(source=source, target=target):
                    data = (np.arange(6).reshape(2, 3) / 4 - 0.5).astype(source)
                    if "complex" in source:
                        data += 0.25j
                    x = jt.array(data, dtype=source)

                    def loss(value):
                        converted = value.cast(target)
                        if "complex" in target:
                            return (converted.conj() * converted).real.sum()
                        return (converted * converted).sum()

                    expected = (2 * data.real).astype(source)
                    gradient = jt.grad(jt.vmap(loss)(x).sum(), x)
                    self.assertEqual(str(gradient.dtype), source)
                    np.testing.assert_allclose(gradient.numpy(), expected, atol=1e-6)
                    values, gradients = jt.vmap(
                        lambda value: vjp(loss, value, create_graph=True)
                    )(x)
                    self.assertEqual(str(gradients.dtype), source)
                    np.testing.assert_allclose(
                        values.numpy(), (data.real**2).sum(-1), atol=1e-6
                    )
                    np.testing.assert_allclose(gradients.numpy(), expected, atol=1e-6)
                    if "complex" not in source:
                        second = jt.grad(gradients.sum(), x)
                        np.testing.assert_allclose(
                            second.numpy(), np.full_like(data, 2), atol=1e-6
                        )

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
        np.testing.assert_array_equal((x == x).numpy(), np.ones(x.shape, dtype=bool))
        np.testing.assert_array_equal(jt.init.eye(3).numpy(), np.eye(3))

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

    # ----------------------------------------------------------------------
    # jittor-core-gaps.md §3.1: native vmap composing with gradfunctional's
    # jvp/vjp (`vmap(lambda x, v: jvp(f, x, v))` / `vmap(lambda x: vjp(f, x))`).
    # Every case below is checked against an explicit per-sample Python-loop
    # reference, matching the doc's own acceptance criteria wording ("与逐
    # 样本参考一致").
    # ----------------------------------------------------------------------

    def test_vmap_jvp_matches_per_sample(self):
        def f(x):
            return (x * x).sum()
        B, N = 4, 3
        x_np = np.random.randn(B, N).astype("float32")
        v_np = np.random.randn(B, N).astype("float32")

        ref_o, ref_j = [], []
        for i in range(B):
            xi = jt.array(x_np[i]); xi.requires_grad = True
            o, j = jvp(f, xi, v=jt.array(v_np[i]))
            ref_o.append(o.item()); ref_j.append(j.item())

        def batched(x, v):
            return jvp(f, x, v=v)
        got_o, got_j = jt.vmap(batched)(jt.array(x_np), jt.array(v_np))
        np.testing.assert_allclose(got_o.numpy(), ref_o, atol=1e-5)
        np.testing.assert_allclose(got_j.numpy(), ref_j, atol=1e-5)

    def test_vmap_vjp_matches_per_sample(self):
        def f(x):
            return (x * x).sum()
        B, N = 4, 3
        x_np = np.random.randn(B, N).astype("float32")

        ref_o, ref_g = [], []
        for i in range(B):
            o, g = vjp(f, jt.array(x_np[i]))
            ref_o.append(o.item()); ref_g.append(g.numpy())

        def batched(x):
            return vjp(f, x)
        got_o, got_g = jt.vmap(batched)(jt.array(x_np))
        np.testing.assert_allclose(got_o.numpy(), ref_o, atol=1e-5)
        np.testing.assert_allclose(got_g.numpy(), ref_g, atol=1e-5)

    def test_vmap_vjp_multi_input_output(self):
        def f(x, y):
            return (x * y).sum(), (x - y).sum()
        B = 4
        x_np = np.random.randn(B, 3).astype("float32")
        y_np = np.random.randn(B, 3).astype("float32")
        v = (jt.array(1.0), jt.array(1.0))

        ref = [vjp(f, (jt.array(x_np[i]), jt.array(y_np[i])), v=v) for i in range(B)]
        ref_o1 = [r[0][0].item() for r in ref]
        ref_gx = [r[1][0].numpy() for r in ref]

        def batched(x, y):
            (o1, o2), (gx, gy) = vjp(f, (x, y), v=v)
            return o1, gx
        got_o1, got_gx = jt.vmap(batched)(jt.array(x_np), jt.array(y_np))
        np.testing.assert_allclose(got_o1.numpy(), ref_o1, atol=1e-5)
        np.testing.assert_allclose(got_gx.numpy(), ref_gx, atol=1e-5)

    def test_vmap_vjp_shared_param_in_dims_none(self):
        # jittor-core-gaps.md acceptance #4: in_dims=None for a shared
        # (non-mapped) parameter, combined with vjp -- exercises the matmul
        # fix below (a shared weight matrix applied inside the vjp'd
        # function).
        B, N = 4, 3
        x_np = np.random.randn(B, N).astype("float32")
        W_np = (np.random.randn(N, N) * 0.5).astype("float32")

        def f(x, W):
            return (W @ x).sum()

        ref_g = []
        for i in range(B):
            W = jt.array(W_np)
            _, g = vjp(lambda xx: f(xx, W), jt.array(x_np[i]))
            ref_g.append(g.numpy())

        def batched(x, W):
            return vjp(lambda xx: f(xx, W), x)
        _, got_g = jt.vmap(batched, in_dims=(0, None))(jt.array(x_np), jt.array(W_np))
        np.testing.assert_allclose(got_g.numpy(), ref_g, atol=1e-4)

    def test_matmul_shared_weight_batched_vector(self):
        # 2026-08-26-core-gaps-3.1-3.8-verification.md §6.2(a): `U @ x` where
        # U is a plain (unbatched) matrix and x is vmapped used to raise
        # "dimension not match" -- the single most common vmap pattern
        # (shared weight applied per-sample).
        N, B = 3, 4
        U = jt.array((np.eye(N, dtype="float32") * 2))
        x = jt.array(np.random.randn(B, N).astype("float32"))
        out = jt.vmap(lambda xx: U @ xx)(x)
        np.testing.assert_allclose(out.numpy(), (U.numpy() @ x.numpy().T).T, atol=1e-5)
        out2 = jt.vmap(lambda xx: xx @ U)(x)
        np.testing.assert_allclose(out2.numpy(), x.numpy() @ U.numpy(), atol=1e-5)

    # jittor-core-gaps.md 2026-09-05 §3.2: native (non-looping) vmap batching
    # rules for the single-input factorizations qr/svd/svdvals/eigh/inv.
    # These ops are already implemented batch-dim-agnostic (linalg.py wraps
    # np.linalg.* via numpy_code, which treats any leading dims as batch
    # dims), so the batching rule is a pure pass-through with no reshaping,
    # verified here per-sample against plain (unbatched) calls -- not against
    # a Python-loop fallback, so this actually exercises the "one real op
    # graph" native path, not the vmap docstring's own loop-based decoy.
    #
    # `vmap(lambda a: jt.linalg.qr(a))`, not the bare `vmap(jt.linalg.qr)`:
    # see the `func` parameter note on `vmap()`'s own docstring -- a bare op
    # reference is resolved before this module's lazy first-call patching,
    # so on this test file's very first vmap call it would capture the
    # original, BatchedVar-oblivious op and fail with a confusing low-level
    # jt.numpy_code overload-resolution error instead of exercising the
    # batching rule this test is actually meant to check.
    def test_vmap_qr_matches_per_sample(self):
        rng = np.random.RandomState(11)
        X = jt.array(rng.randn(4, 6, 3).astype(np.float32))
        q, r = jt.vmap(lambda a: jt.linalg.qr(a))(X)
        for i in range(4):
            Q, R = jt.linalg.qr(X[i])
            np.testing.assert_allclose(q[i].numpy(), Q.numpy(), atol=1e-4)
            np.testing.assert_allclose(r[i].numpy(), R.numpy(), atol=1e-4)
            np.testing.assert_allclose(q[i].numpy() @ r[i].numpy(), X[i].numpy(), atol=1e-4)

    def test_vmap_eigh_matches_per_sample(self):
        # jt.array of a float64 array is, by default, silently downcast to
        # float32 (jt.flags.auto_convert_64_to_32) -- that's fine for most
        # tests, but comparing "batched eigh" against "4 separate eigh
        # calls" at float32 precision picks up ~1e-6 LAPACK-routine-order
        # noise unrelated to the batching rule itself (see __init__.py's own
        # `with jt.flag_scope(auto_convert_64_to_32=0)` uses for the same
        # reason). Disable it here so this actually tests the batching rule
        # at the float64 precision the inputs were built at.
        with jt.flag_scope(auto_convert_64_to_32=0):
            rng = np.random.RandomState(12)
            A = rng.randn(4, 5, 5).astype(np.float64)
            A = A + A.transpose(0, 2, 1)
            X = jt.array(A)
            w, v = jt.vmap(lambda a: jt.linalg.eigh(a))(X)
            for i in range(4):
                W, V = jt.linalg.eigh(X[i])
                np.testing.assert_allclose(np.sort(w[i].numpy()), np.sort(W.numpy()), atol=1e-9)

    def test_vmap_inv_matches_per_sample(self):
        with jt.flag_scope(auto_convert_64_to_32=0):
            rng = np.random.RandomState(13)
            A = rng.randn(4, 5, 5).astype(np.float64) + 5 * np.eye(5)[None]
            X = jt.array(A)
            out = jt.vmap(lambda a: jt.linalg.inv(a))(X)
            for i in range(4):
                np.testing.assert_allclose(out[i].numpy(), jt.linalg.inv(X[i]).numpy(), atol=1e-9)

    def test_vmap_svdvals_matches_per_sample(self):
        with jt.flag_scope(auto_convert_64_to_32=0):
            rng = np.random.RandomState(14)
            X = jt.array(rng.randn(4, 5, 3).astype(np.float64))
            out = jt.vmap(lambda a: jt.linalg.svdvals(a))(X)
            for i in range(4):
                np.testing.assert_allclose(out[i].numpy(), jt.linalg.svdvals(X[i]).numpy(), atol=1e-9)

    def test_vmap_svd_matches_per_sample(self):
        with jt.flag_scope(auto_convert_64_to_32=0):
            rng = np.random.RandomState(15)
            X = jt.array(rng.randn(4, 5, 3).astype(np.float64))
            u, s, v = jt.vmap(lambda a: jt.linalg.svd(a))(X)
            for i in range(4):
                U, S, V = jt.linalg.svd(X[i])
                np.testing.assert_allclose(s[i].numpy(), S.numpy(), atol=1e-9)
                recon = u[i].numpy() @ np.diag(s[i].numpy()) @ v[i].numpy()
                np.testing.assert_allclose(recon, X[i].numpy(), atol=1e-8)

    def test_complex64_vmap_vjp_expectation_value(self):
        # 2026-08-26-core-gaps-3.1-3.8-verification.md §6.2(b): conj/real/
        # imag were not batchable, so no complex-valued (quantum-state-
        # style) expectation value could flow through vjp under vmap.
        B, n = 4, 3
        state_np = (np.random.randn(B, n) + 1j * np.random.randn(B, n)).astype("complex64")

        def expectation(state):
            return (state.conj() * state).sum().real

        ref_o, ref_g = [], []
        for i in range(B):
            o, g = vjp(expectation, jt.array(state_np[i]))
            ref_o.append(o.item()); ref_g.append(g.numpy())

        def batched(s):
            return vjp(expectation, s)
        got_o, got_g = jt.vmap(batched)(jt.array(state_np))
        np.testing.assert_allclose(got_o.numpy(), ref_o, atol=1e-4)
        np.testing.assert_allclose(got_g.numpy(), ref_g, atol=1e-4)

    def test_nested_vmap_vjp(self):
        def f(x):
            return (x * x).sum()
        B1, B2, N = 3, 4, 2
        x_np = np.random.randn(B1, B2, N).astype("float32")

        ref_o = np.zeros((B1, B2), dtype="float32")
        ref_g = np.zeros((B1, B2, N), dtype="float32")
        for i in range(B1):
            for j in range(B2):
                o, g = vjp(f, jt.array(x_np[i, j]))
                ref_o[i, j] = o.item(); ref_g[i, j] = g.numpy()

        def batched(x):
            return vjp(f, x)
        got_o, got_g = jt.vmap(jt.vmap(batched))(jt.array(x_np))
        np.testing.assert_allclose(got_o.numpy(), ref_o, atol=1e-5)
        np.testing.assert_allclose(got_g.numpy(), ref_g, atol=1e-5)

    def test_vmap_vjp_create_graph_second_order(self):
        def f(x):
            return (x ** 3).sum()
        B, N = 4, 3
        x_np = np.random.randn(B, N).astype("float32")

        ref = np.zeros((B, N), dtype="float32")
        for i in range(B):
            xi = jt.array(x_np[i]); xi.requires_grad = True
            _, g = vjp(f, xi, create_graph=True)
            ref[i] = jt.grad(g.sum(), xi).numpy()

        def batched(x):
            _, g = vjp(f, x, create_graph=True)
            return jt.grad(g.sum(), x)
        got = jt.vmap(batched)(jt.array(x_np))
        np.testing.assert_allclose(got.numpy(), ref, atol=1e-4)

    def test_randomness_same_vs_different(self):
        B = 8
        def rf(_x):
            return jt.random((3,))
        diff_out = jt.vmap(rf, randomness="different")(jt.zeros((B,))).numpy()
        same_out = jt.vmap(rf, randomness="same")(jt.zeros((B,))).numpy()
        self.assertFalse(np.allclose(diff_out[0], diff_out[1]))
        for i in range(1, B):
            np.testing.assert_array_equal(same_out[i], same_out[0])


@unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
class TestVmapCuda(TestVmap):
    def setUp(self):
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.flags.use_cuda = 0


if __name__ == "__main__":
    unittest.main()

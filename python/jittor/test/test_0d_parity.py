"""0-D (rank-0/scalar) tensor parity with NumPy/PyTorch (jittor-core-gaps.md §3.1).

Jittor represents scalars and shape-() results as physical 0-D Vars
(``shape == []``, ``numel() == 1``), distinct from length-one vectors
(``shape == [1]``). This file locks down the acceptance checklist from
agent/workdocs/jittor-core-gaps.md §3.1 and the fuller item list in
agent/workdocs/2026-08-13-0d-tensor-parity-plan.md §three, across CPU and CUDA.

Run::  python -m jittor.test.test_0d_parity
"""
import os
import pickle
import tempfile
import unittest

import numpy as np

import jittor as jt

from jittor.test._internal.common_utils import (
    JittorTestCase, get_all_device_types, use_cuda_for,
)


class Test0DParity(JittorTestCase):

    def _devices(self, body):
        for d in get_all_device_types():
            with self.subTest(device=d):
                with jt.flag_scope(use_cuda=use_cuda_for(d)):
                    body(d)

    def test_scalar_creation_and_numpy(self):
        def body(d):
            x = jt.array(1.0)
            self.assertEqual(x.shape, [])
            self.assertEqual(x.ndim, 0)
            self.assertEqual(x.numpy().shape, ())
            self.assertEqual(x.item(), 1.0)

            y = jt.array(np.float32(2.0))
            self.assertEqual(y.shape, [])
            self.assertEqual(y.numpy().shape, ())

            # a real length-one vector stays (1,) -- must not collapse to 0-D
            z = jt.array([1.0])
            self.assertEqual(z.shape, [1])
        self._devices(body)

    def test_ones_zeros_empty_shape(self):
        def body(d):
            self.assertEqual(jt.ones(()).shape, [])
            self.assertEqual(jt.zeros(()).shape, [])
            self.assertEqual(jt.ones(()).item(), 1.0)
            self.assertEqual(jt.zeros(()).item(), 0.0)
        self._devices(body)

    def test_reduce_index_reshape_transpose(self):
        def body(d):
            x = jt.array([1.0, 2.0, 3.0])
            self.assertEqual(x.sum().shape, [])
            self.assertEqual(x.sum(dims=[0], keepdims=True).shape, [1])
            self.assertEqual(x[0].shape, [])
            self.assertEqual(jt.array([[1, 2], [3, 4]])[0, 1].shape, [])
            self.assertEqual(jt.array(1.0).reshape([]).shape, [])
            self.assertEqual(jt.array(1.0).reshape([1]).shape, [1])
            self.assertEqual(jt.array(1.0).transpose().shape, [])
        self._devices(body)

    def test_argmax_argmin_full_reduce(self):
        # cub_arg_reduce_op.cc had a leftover "empty shape -> push 1" fallback
        # (same pattern as the old Var::numel() default) that only the CUDA
        # fast path hit, so plain jt.arg_reduce() with keepdims=False on a
        # fully-reduced 1-D input silently kept shape [1] on CUDA while CPU
        # already returned [].
        def body(d):
            x = jt.array([1.0, 3.0, 2.0])
            idx, val = jt.arg_reduce(x, "max", 0, False)
            self.assertEqual(idx.shape, [])
            self.assertEqual(val.shape, [])
            self.assertEqual(idx.item(), 1)
            idx_k, val_k = jt.arg_reduce(x, "max", 0, True)
            self.assertEqual(idx_k.shape, [1])
            self.assertEqual(val_k.shape, [1])
            # torch-compat method form: argmax() with no dim flattens to scalar
            self.assertEqual(x.argmax().shape, [])
            self.assertEqual(x.argmin().shape, [])
        self._devices(body)

    def test_squeeze_unsqueeze(self):
        def body(d):
            self.assertEqual(jt.array([[1.0]]).squeeze().shape, [])
            self.assertEqual(jt.array(1.0).unsqueeze(0).shape, [1])
        self._devices(body)

    def test_broadcast_stack_concat(self):
        def body(d):
            self.assertEqual((jt.array(1.0) + jt.ones(3)).shape, [3])
            self.assertEqual(
                jt.stack([jt.array(1.0), jt.array(2.0)]).shape, [2])
            with self.assertRaises(Exception):
                jt.concat([jt.array(1.0)])
        self._devices(body)

    def test_matmul_full_contraction(self):
        def body(d):
            a = jt.array([1.0, 2.0, 3.0])
            b = jt.array([4.0, 5.0, 6.0])
            self.assertEqual(jt.matmul(a, b).shape, [])
            self.assertAlmostEqual(jt.matmul(a, b).item(), 32.0, places=4)
        self._devices(body)

    def test_setitem_scalar_value(self):
        def body(d):
            x = jt.array([1.0, 2.0, 3.0])
            x[1] = jt.array(9.0)
            np.testing.assert_allclose(x.numpy(), [1.0, 9.0, 3.0])
        self._devices(body)

    def test_scalar_protocols(self):
        x = jt.array(1.0)
        with self.assertRaises(TypeError):
            len(x)
        with self.assertRaises(TypeError):
            iter(x)

    def test_pickle_roundtrip(self):
        x = jt.array(np.float32(3.0))
        y = pickle.loads(pickle.dumps(x))
        self.assertEqual(y.shape, [])
        self.assertEqual(y.item(), 3.0)

    def test_save_load_roundtrip(self):
        x = jt.array(3.0)
        p = tempfile.mktemp(suffix=".pkl")
        try:
            jt.save(x, p)
            y = jt.load(p)
        finally:
            if os.path.exists(p):
                os.remove(p)
        self.assertEqual(y.shape, [])
        self.assertEqual(y.item(), 3.0)

    def test_grad_of_0d_loss(self):
        def body(d):
            x = jt.array([1.0, 2.0, 3.0])
            x.requires_grad = True
            y = (x * x).sum()
            self.assertEqual(y.shape, [])
            g = jt.grad(y, [x])[0]
            self.assertEqual(g.shape, x.shape)
            np.testing.assert_allclose(g.numpy(), 2 * x.numpy(), atol=1e-5)
        self._devices(body)

    def test_mse_loss_reduction_shape(self):
        def body(d):
            a = jt.array([1.0, 2.0])
            b = jt.array([1.5, 2.5])
            self.assertEqual(jt.nn.mse_loss(a, b).shape, [])
        self._devices(body)

    def test_weak_scalar_promotion_not_regressed(self):
        # a 0-D float32 Var times a python float must stay float32 (weak
        # scalar promotion), matching pre-0-D behavior.
        def body(d):
            x = jt.array(1.0)
            self.assertEqual(str((x * 2.0).dtype), "float32")
        self._devices(body)


if __name__ == "__main__":
    unittest.main()

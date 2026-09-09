# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guoye Yang <498731903@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
from jittor import nn, Module
from jittor.models import vgg, resnet
import numpy as np
import sys, os
import random
import math
import unittest
from .test_reorder_tuner import simple_parser
from .test_log import find_log_with_re

skip_this_test = False
try:
    jt.dirty_fix_pytorch_runtime_error()
    import torch
except:
    skip_this_test = True


class TestRandomOp(unittest.TestCase):
    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    @jt.flag_scope(use_cuda=1)
    def test(self):
        jt.set_seed(3)
        with jt.log_capture_scope(
            log_silent=1,
            log_v=0, log_vprefix="op.cc=100"
        ) as raw_log:
            t = jt.random([5,5])
            t.data
        logs = find_log_with_re(raw_log, "(Jit op key (not )?found: " + "curand_random" + ".*)")
        assert len(logs)==1

    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    @jt.flag_scope(use_cuda=1)
    def test_float64(self):
        jt.set_seed(3)
        with jt.log_capture_scope(
            log_silent=1,
            log_v=0, log_vprefix="op.cc=100"
        ) as raw_log:
            t = jt.random([5,5], dtype='float64')
            t.data
        logs = find_log_with_re(raw_log, "(Jit op key (not )?found: " + "curand_random" + ".*)")
        assert len(logs)==1

    @unittest.skipIf(skip_this_test, "No Torch Found")
    def test_normal(self):
        from jittor import init
        n = 10000
        r = 0.155
        a = init.gauss([n], "float32", 1, 3)
        data = a.data

        assert (np.abs((data<(1-3)).mean() - r) < 0.1)
        assert (np.abs((data<(1)).mean() - 0.5) < 0.1)
        assert (np.abs((data<(1+3)).mean() - (1-r)) < 0.1)

        np_res = np.random.normal(1, 0.1, (100, 100))
        jt_res = jt.normal(1., 0.1, (100, 100))
        assert (np.abs(np_res.mean() - jt_res.data.mean()) < 0.1)
        assert (np.abs(np_res.std() - jt_res.data.std()) < 0.1)

        np_res = torch.normal(torch.arange(1., 10000.), 1)
        jt_res = jt.normal(jt.arange(1, 10000), 1)
        assert (np.abs(np_res.mean() - jt_res.data.mean()) < 0.1)
        assert (np.abs(np_res.std() - jt_res.data.std()) < 1)

        np_res = np.random.randn(100, 100)
        jt_res = jt.randn(100, 100)
        assert (np.abs(np_res.mean() - jt_res.data.mean()) < 0.1)
        assert (np.abs(np_res.std() - jt_res.data.std()) < 0.1)

        np_res = np.random.rand(100, 100)
        jt_res = jt.rand(100, 100)
        assert (np.abs(np_res.mean() - jt_res.data.mean()) < 0.1)
        assert (np.abs(np_res.std() - jt_res.data.std()) < 0.1)

    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    @jt.flag_scope(use_cuda=1)
    def test_normal_cuda(self):
        self.test_normal()

    def test_random_dtype_not_silently_dropped(self):
        # jittor-core-gaps.md §3.7: jt.random() used to hardcode the sampled
        # dtype to float32 and ignore the caller's `dtype` argument entirely
        # (a `dtype='float64'` request silently came back as float32). The
        # underlying CPU (std::uniform_real_distribution/normal_distribution<T>)
        # and CUDA (curandGenerateUniform/NormalDouble) kernels both natively
        # support float64, so it should be generated directly rather than
        # discarded.
        for type_ in ("uniform", "normal"):
            r32 = jt.random([8, 8], dtype="float32", type=type_)
            r64 = jt.random([8, 8], dtype="float64", type=type_)
            assert str(r32.dtype) == "float32", (type_, r32.dtype)
            assert str(r64.dtype) == "float64", (type_, r64.dtype)
        # dtypes with no native distribution kernel keep the existing
        # sample-in-float32-then-cast fallback.
        r16 = jt.random([8, 8], dtype="float16")
        assert str(r16.dtype) == "float16"

    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    @jt.flag_scope(use_cuda=1)
    def test_random_dtype_not_silently_dropped_cuda(self):
        self.test_random_dtype_not_silently_dropped()

    def test_other_rand(self):
        a = jt.array([1.0,2.0,3.0])
        b = jt.rand_like(a)
        c = jt.randn_like(a)
        assert b.shape == c.shape
        assert b.shape == a.shape
        print(b, c)
        assert jt.randint(10, 20, (2000,)).min() == 10
        assert jt.randint(10, 20, (2000,)).max() == 19
        assert jt.randint(10, shape=(2000,)).max() == 9
        assert jt.randint_like(a, 10).shape == a.shape


class TestGenerator(unittest.TestCase):
    ''' jittor-core-gaps.md §3.6: independent, device-native RNG generators.

    `jt.Generator` (after the default `torch_compat` install every
    `import jittor` performs) wraps a real per-instance RNG state (CPU
    `std::default_random_engine` / CUDA `curandGenerator_t`, see
    `src/misc/random_generator.h`) instead of the old placeholder that only
    stored a `_seed` and reseeded the GLOBAL RNG on every draw.
    '''

    def _check(self, use_cuda):
        with jt.flag_scope(use_cuda=use_cuda):
            # same seed -> identical sequence
            g1 = jt.Generator(); g1.manual_seed(1)
            g2 = jt.Generator(); g2.manual_seed(1)
            a = jt.randn(5, generator=g1).numpy()
            b = jt.randn(5, generator=g2).numpy()
            np.testing.assert_array_equal(a, b)

            # interleaved draws on two independent generators don't perturb
            # each other: the g1 sub-sequence must match a fresh, uninterrupted
            # generator seeded the same way.
            g1 = jt.Generator(); g1.manual_seed(42)
            g2 = jt.Generator(); g2.manual_seed(42)
            seqA = []
            for _ in range(3):
                seqA.append(jt.randn(4, generator=g1).numpy().copy())
                jt.rand(4, generator=g2).numpy()
            g3 = jt.Generator(); g3.manual_seed(42)
            seqA2 = [jt.randn(4, generator=g3).numpy().copy() for _ in range(3)]
            for x, y in zip(seqA, seqA2):
                np.testing.assert_array_equal(x, y)

            # dtype is honored (no silent float32 fallback) alongside generator=
            c = jt.randn(3, dtype="float64", generator=g1)
            assert str(c.dtype) == "float64"

            # state save/restore round-trips, both odd and even draw sizes
            # (odd sizes exercise curand's even-padding on CUDA).
            for n_first, n_next in ((2, 3), (2, 4)):
                g4 = jt.Generator(); g4.manual_seed(7)
                jt.randn(n_first, generator=g4).numpy()
                state = g4.get_state()
                x1 = jt.randn(n_next, generator=g4).numpy().copy()
                g4.set_state(state)
                x2 = jt.randn(n_next, generator=g4).numpy().copy()
                np.testing.assert_array_equal(x1, x2)
            g4u = jt.Generator(); g4u.manual_seed(7)
            jt.rand(3, generator=g4u).numpy()
            state_u = g4u.get_state()
            z1 = jt.rand(5, generator=g4u).numpy().copy()
            g4u.set_state(state_u)
            z2 = jt.rand(5, generator=g4u).numpy().copy()
            np.testing.assert_array_equal(z1, z2)

            # drawing from an explicit generator must not pollute the global
            # (no-generator) RNG stream for other callers.
            jt.set_global_seed(123)
            base = jt.randn(4).numpy().copy()
            jt.set_global_seed(123)
            gX = jt.Generator(); gX.manual_seed(999)
            jt.randn(101, generator=gX).numpy()
            after = jt.randn(4).numpy().copy()
            np.testing.assert_array_equal(base, after)

            # Var.normal_/Var.uniform_ (torch in-place API) honor generator=
            v1 = jt.zeros(5); gA = jt.Generator(); gA.manual_seed(11)
            v1.normal_(0, 1, generator=gA)
            v2 = jt.zeros(5); gB = jt.Generator(); gB.manual_seed(11)
            v2.normal_(0, 1, generator=gB)
            np.testing.assert_array_equal(v1.numpy(), v2.numpy())

            v3 = jt.zeros(5); v3.uniform_(0, 1, generator=gA)
            v4 = jt.zeros(5); v4.uniform_(0, 1, generator=gB)
            np.testing.assert_array_equal(v3.numpy(), v4.numpy())

    def test_generator_cpu(self):
        self._check(use_cuda=0)

    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    def test_generator_cuda(self):
        self._check(use_cuda=1)


if __name__ == "__main__":
    unittest.main()

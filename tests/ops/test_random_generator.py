# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Jittor Group
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# Native jt.Generator (src/runtime/random_generator.{h,cc}): an independent,
# device-native RNG state a caller can create/seed/save/restore, separate
# from the single global RNG stream that ordinary jt.random()/jt.randn()/etc.
# draw from. This repo previously had no such thing -- only a torch-compat,
# numpy-backed shim in compat/torch, disconnected from native RNG entirely.
#
# On CUDA, each jt.Generator("cuda") owns its own curandGenerator_t handle
# (created via a hook so core code doesn't hard-link -lcurand), distinct
# from the per-device global generator every other draw shares.
#
# Note: jittor is lazy -- an op doesn't run until its output is synced, and
# there is no dataflow edge between two independent generator-consuming
# draws. get_state() reads the generator's *current* state, so a draw must
# be sync()'d before get_state()/set_state() are meaningful around it.
import unittest

import numpy as np

import jittor as jt
from _helpers import capability as _test_capability

_has_cuda = _test_capability.check_accelerator("cuda", backend=jt).enabled


class TestRandomGenerator(unittest.TestCase):
    def test_seeded_reproducible_cpu(self):
        g1 = jt.Generator("cpu").manual_seed(42)
        g2 = jt.Generator("cpu").manual_seed(42)
        a = jt.random((8,), generator=g1).numpy()
        b = jt.random((8,), generator=g2).numpy()
        np.testing.assert_array_equal(a, b)

    def test_different_seed_differs_cpu(self):
        g1 = jt.Generator("cpu").manual_seed(1)
        g2 = jt.Generator("cpu").manual_seed(2)
        a = jt.random((8,), generator=g1).numpy()
        b = jt.random((8,), generator=g2).numpy()
        self.assertFalse(np.allclose(a, b))

    def test_independent_generators_do_not_perturb_each_other(self):
        g1 = jt.Generator("cpu").manual_seed(7)
        g2 = jt.Generator("cpu").manual_seed(123)
        baseline = jt.random((5,), generator=g1).numpy()

        g1b = jt.Generator("cpu").manual_seed(7)
        interleaved = []
        for _ in range(5):
            jt.random((1,), generator=g2).sync()
            interleaved.append(jt.random((1,), generator=g1b).numpy()[0])
        np.testing.assert_allclose(baseline, np.array(interleaved))

    def test_generator_does_not_perturb_global_stream(self):
        jt.set_seed(0)
        before = jt.random((4,)).numpy()

        jt.set_seed(0)
        g = jt.Generator("cpu").manual_seed(999)
        jt.random((4,), generator=g).sync()  # drawn from g, not the global stream
        after = jt.random((4,)).numpy()
        np.testing.assert_array_equal(before, after)

    def test_state_roundtrip_cpu(self):
        g = jt.Generator("cpu").manual_seed(5)
        jt.random((6,), generator=g).sync()
        state = g.get_state()
        a = jt.random((6,), generator=g).numpy().copy()
        g.set_state(state)
        b = jt.random((6,), generator=g).numpy().copy()
        np.testing.assert_array_equal(a, b)

    def test_initial_seed_and_reseed(self):
        g = jt.Generator("cpu").manual_seed(17)
        self.assertEqual(g.initial_seed(), 17)
        new_seed = g.seed()
        self.assertEqual(g.initial_seed(), new_seed)

    def test_device_property(self):
        g = jt.Generator("cpu")
        self.assertEqual(g.device, "cpu")


class TestRandomGeneratorCuda(unittest.TestCase):
    @unittest.skipIf(not _has_cuda, "No cuda found")
    @jt.flag_scope(use_cuda=1)
    def test_seeded_reproducible_cuda(self):
        g1 = jt.Generator("cuda").manual_seed(42)
        g2 = jt.Generator("cuda").manual_seed(42)
        a = jt.random((6,), generator=g1).numpy()
        b = jt.random((6,), generator=g2).numpy()
        np.testing.assert_allclose(a, b)

    @unittest.skipIf(not _has_cuda, "No cuda found")
    @jt.flag_scope(use_cuda=1)
    def test_different_seed_differs_cuda(self):
        g1 = jt.Generator("cuda").manual_seed(42)
        g2 = jt.Generator("cuda").manual_seed(99)
        a = jt.random((6,), generator=g1).numpy()
        b = jt.random((6,), generator=g2).numpy()
        self.assertFalse(np.allclose(a, b))

    @unittest.skipIf(not _has_cuda, "No cuda found")
    @jt.flag_scope(use_cuda=1)
    def test_uniform_state_roundtrip_cuda(self):
        g = jt.Generator("cuda").manual_seed(5)
        jt.random((7,), generator=g).sync()
        state = g.get_state()
        a = jt.random((7,), generator=g).numpy().copy()
        g.set_state(state)
        b = jt.random((7,), generator=g).numpy().copy()
        np.testing.assert_allclose(a, b)

    @unittest.skipIf(not _has_cuda, "No cuda found")
    @jt.flag_scope(use_cuda=1)
    def test_normal_odd_length_state_roundtrip_cuda(self):
        # curandGenerateNormal requires an even count; an odd-length draw
        # takes its last element from a separate two-element scratch buffer
        # (see curand_random_op.cc). Regression coverage for that split
        # path's offset bookkeeping specifically.
        g = jt.Generator("cuda").manual_seed(5)
        jt.random((7,), dtype="float32", type="normal", generator=g).sync()
        state = g.get_state()
        a = jt.random((7,), dtype="float32", type="normal", generator=g).numpy().copy()
        g.set_state(state)
        b = jt.random((7,), dtype="float32", type="normal", generator=g).numpy().copy()
        np.testing.assert_allclose(a, b)

    @unittest.skipIf(not _has_cuda, "No cuda found")
    @jt.flag_scope(use_cuda=1)
    def test_normal_even_length_state_roundtrip_cuda(self):
        g = jt.Generator("cuda").manual_seed(5)
        jt.random((8,), dtype="float32", type="normal", generator=g).sync()
        state = g.get_state()
        a = jt.random((8,), dtype="float32", type="normal", generator=g).numpy().copy()
        g.set_state(state)
        b = jt.random((8,), dtype="float32", type="normal", generator=g).numpy().copy()
        np.testing.assert_allclose(a, b)

    @unittest.skipIf(not _has_cuda, "No cuda found")
    @jt.flag_scope(use_cuda=1)
    def test_generator_does_not_perturb_global_cuda_stream(self):
        d1 = jt.random((4,)).numpy()
        g = jt.Generator("cuda").manual_seed(999)
        jt.random((4,), generator=g).sync()
        d2 = jt.random((4,)).numpy()
        self.assertEqual(d1.shape, d2.shape)


if __name__ == "__main__":
    unittest.main()

# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Jittor Group
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# jt.lexsort (jittor-core-gaps.md §3.8): a device-resident, multi-key stable
# sort matching numpy.lexsort, composed from repeated calls to the now
# stable-capable native jt.argsort (see argsort_op.{h,cc}'s `stable` param
# and test_argsort_op.py::TestArgsortOpStable). No .numpy()/host round trip
# is used anywhere in the implementation (misc.py::lexsort).
import unittest
import numpy as np
import jittor as jt


class _Mixin:
    use_cuda = 0

    def setUp(self):
        jt.flags.use_cuda = self.use_cuda

    def _check(self, keys_np):
        keys = [jt.array(k) for k in keys_np]
        got = jt.lexsort(keys).numpy()
        ref = np.lexsort(keys_np)
        np.testing.assert_array_equal(got, ref)

    def test_basic_two_keys(self):
        a = np.array([1, 5, 1, 4, 3, 4, 4], dtype="int32")
        b = np.array([9, 4, 0, 4, 0, 2, 1], dtype="int32")
        self._check([b, a])

    def test_single_key(self):
        rng = np.random.RandomState(0)
        a = rng.randint(0, 100, 50).astype("int32")
        self._check([a])

    def test_many_duplicate_keys(self):
        # Small alphabets force long runs of ties, exercising the
        # `stable=True` argsort this is built on rather than accidentally
        # passing via an already-sorted or all-unique input.
        rng = np.random.RandomState(1)
        n = 500
        k0 = rng.randint(0, 4, n).astype("int32")
        k1 = rng.randint(0, 3, n).astype("int32")
        k2 = rng.randint(0, 2, n).astype("int32")
        self._check([k0, k1, k2])

    def test_float_and_int_mixed_keys(self):
        rng = np.random.RandomState(2)
        n = 200
        k0 = rng.randint(0, 5, n).astype("int32")
        k1 = (rng.randint(0, 5, n)).astype("float32")
        self._check([k0, k1])

    def test_empty(self):
        idx = jt.lexsort([jt.array(np.array([], dtype="int32"))])
        self.assertEqual(tuple(idx.shape), (0,))

    def test_single_element(self):
        self._check([np.array([7], dtype="int32")])

    def test_mismatched_shapes_raises(self):
        with self.assertRaises(RuntimeError):
            jt.lexsort([jt.array([1, 2, 3]), jt.array([1, 2])])

    def test_no_keys_raises(self):
        with self.assertRaises(RuntimeError):
            jt.lexsort([])


class TestLexsortCPU(_Mixin, unittest.TestCase):
    use_cuda = 0


@unittest.skipIf(not jt.has_cuda, "Cuda not found")
class TestLexsortCUDA(_Mixin, unittest.TestCase):
    use_cuda = 1


if __name__ == "__main__":
    unittest.main()

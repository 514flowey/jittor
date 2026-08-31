# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# Tests for DLPack ownership/device/stream semantics (jittor-core-gaps.md
# section 3.7): zero-copy import/export via the classic (unversioned)
# DLManagedTensor, real cross-framework round-trips (NumPy for CPU, CuPy for
# a real GPU when available), single-consumption capsule semantics, deleter
# lifetime, and explicit fail-fast for what's out of the core-set scope
# (non-contiguous tensors, save_mem/swap).
import unittest
import gc
import numpy as np
import jittor as jt

try:
    import cupy as cp
    has_cupy = True
except Exception:
    has_cupy = False

_ALL_DTYPES = [np.bool_, np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float16, np.float32, np.float64,
    np.complex64, np.complex128]


def _sample(dt, n=3):
    if np.issubdtype(dt, np.complexfloating):
        return np.array([complex(i, i + 1) for i in range(n)], dtype=dt)
    if dt == np.bool_:
        return np.array([True, False, True][:n], dtype=dt)
    return np.arange(1, n + 1, dtype=dt)


class TestDLPack(unittest.TestCase):
    def test_export_dtype_coverage_numpy_roundtrip(self):
        for dt in _ALL_DTYPES:
            src = _sample(dt)
            y = jt.from_dlpack(src.copy())
            back = np.from_dlpack(y)
            np.testing.assert_array_equal(back, src)
            self.assertEqual(back.dtype, src.dtype)

    def test_import_from_raw_capsule_and_dunder_alias(self):
        x = jt.array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        cap = x.__dlpack__()
        y = jt.core.from_dlpack_capsule(cap)
        np.testing.assert_allclose(y.numpy(), [1, 2, 3])

    def test_dlpack_device_cpu(self):
        jt.flags.use_cuda = 0
        x = jt.array(np.array([1.0], dtype=np.float32))
        self.assertEqual(x.__dlpack_device__(), (1, 0))  # kDLCPU, 0

    def test_bfloat16_self_roundtrip(self):
        # NumPy has no native bfloat16, so this is a Jittor-only round trip.
        x = jt.array(np.array([1.5, 2.5, 3.5], dtype=np.float32)).cast("bfloat16")
        cap = x.__dlpack__()
        y = jt.core.from_dlpack_capsule(cap)
        self.assertEqual(str(y.dtype), "bfloat16")
        np.testing.assert_allclose(y.float32().numpy(), [1.5, 2.5, 3.5])

    def test_cpu_zero_copy_mutation_visible_on_import(self):
        # jittor-core-gaps.md §3.7 criterion 2 (zero-copy + mutation visible
        # both ways) previously had test coverage ONLY behind the CuPy-gated
        # TestDLPackCupy class -- no CPU/NumPy-only proof existed, so on any
        # machine without CuPy this criterion had zero executed evidence.
        # from_dlpack must alias the NumPy buffer, not copy it: mutating the
        # source array after import must be visible through the Jittor Var.
        jt.flags.use_cuda = 0
        src = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = jt.from_dlpack(src)
        np.testing.assert_allclose(y.numpy(), [1, 2, 3])
        src[1] = 42.0
        np.testing.assert_allclose(y.numpy(), [1, 42, 3])

    def test_non_contiguous_import_rejected(self):
        a = np.arange(20, dtype=np.float32).reshape(4, 5)
        sliced = a[:, ::2]
        self.assertFalse(sliced.flags["C_CONTIGUOUS"])
        with self.assertRaises(Exception):
            jt.from_dlpack(sliced)

    def test_capsule_consumed_exactly_once(self):
        x = jt.array(np.array([1.0], dtype=np.float32))
        cap = x.__dlpack__()
        jt.core.from_dlpack_capsule(cap)
        with self.assertRaises(Exception):
            jt.core.from_dlpack_capsule(cap)

    def test_producer_drops_reference_before_consumer_reads(self):
        def make_capsule():
            v = jt.array(np.array([42.0, 43.0], dtype=np.float32))
            return v.__dlpack__()
        cap = make_capsule()
        gc.collect()
        y = jt.core.from_dlpack_capsule(cap)
        np.testing.assert_allclose(y.numpy(), [42, 43])

    def test_unconsumed_capsule_frees_storage_exactly_once(self):
        n0 = jt.number_of_lived_vars()

        def make_unused_capsule():
            v = jt.array(np.array([1.0, 2.0], dtype=np.float32))
            return v.__dlpack__()  # never imported by anyone
        cap = make_unused_capsule()
        gc.collect()
        n1 = jt.number_of_lived_vars()
        self.assertGreater(n1, n0)  # capsule keeps the storage alive
        del cap
        gc.collect()
        n2 = jt.number_of_lived_vars()
        self.assertEqual(n2, n0)  # freed back to baseline, exactly once

    def test_dlpack_copy_true_rejected(self):
        x = jt.array(np.array([1.0], dtype=np.float32))
        with self.assertRaises(Exception):
            x.dlpack(copy=True)


@unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
class TestDLPackCuda(unittest.TestCase):
    def setUp(self):
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.flags.use_cuda = 0

    def test_dlpack_device_cuda(self):
        x = jt.array(np.array([1.0], dtype=np.float32))
        x.sync()
        dev_type, dev_id = x.__dlpack_device__()
        self.assertEqual(dev_type, 2)  # kDLCUDA
        self.assertEqual(dev_id, x.device_id())

    def test_self_roundtrip_preserves_device(self):
        dev = 0 if jt.get_device_count() < 2 else 1
        x = jt.array(np.array([1.0, 2.0, 3.0], dtype=np.float32)).migrate_to_device(dev)
        cap = x.__dlpack__()
        y = jt.core.from_dlpack_capsule(cap)
        self.assertEqual(y.device_id(), dev)
        np.testing.assert_allclose(y.numpy(), [1, 2, 3])

    def test_migrate_to_device_rejected_on_dlpack_import(self):
        # A DLPack-imported Var's allocator->free() fires the producer's
        # deleter -- migrating it would release memory a NumPy/CuPy/PyTorch
        # array still owns as a side effect of a routine device move.
        x = jt.array(np.array([1.0, 2.0, 3.0], dtype=np.float32)).migrate_to_device(0)
        cap = x.__dlpack__()
        y = jt.core.from_dlpack_capsule(cap)
        with self.assertRaises(Exception):
            y.migrate_to_device(0 if jt.get_device_count() < 2 else 1)
        with self.assertRaises(Exception):
            y.migrate_to_device(-1)


@unittest.skipIf(not (jt.compiler.has_cuda and has_cupy), "No CUDA+CuPy found")
class TestDLPackCupy(unittest.TestCase):
    def setUp(self):
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.flags.use_cuda = 0

    def test_roundtrip_dtype_coverage(self):
        for dt_name in ["float32", "float64", "int32", "int64", "bool_"]:
            dt = getattr(np, dt_name)
            src = _sample(dt)
            c = cp.array(src)
            y = jt.from_dlpack(c)
            self.assertEqual(y.device_id(), 0)
            back = cp.from_dlpack(y)
            np.testing.assert_array_equal(cp.asnumpy(back), src)

    def test_zero_copy_mutation_visible_both_ways(self):
        x = jt.array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        x.sync()
        c = cp.from_dlpack(x)
        c[0] = 99.0
        np.testing.assert_allclose(x.numpy(), [99.0, 2.0, 3.0])

    def test_multi_gpu_device_id_preserved(self):
        if jt.get_device_count() < 2:
            self.skipTest("needs >=2 GPUs")
        with cp.cuda.Device(1):
            c = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        y = jt.from_dlpack(c)
        self.assertEqual(y.device_id(), 1)
        np.testing.assert_allclose(y.numpy(), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()

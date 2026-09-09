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

try:
    import torch
    has_torch = True
except Exception:
    has_torch = False


def _torch_cuda_ready():
    # Importing jittor first initializes CUDA via its own bundled runtime,
    # dlopen'd RTLD_GLOBAL (see compiler.py's import_flags) -- when PyTorch's
    # own *lazy* CUDA init (torch.cuda._lazy_init(), first triggered by e.g.
    # `torch.tensor(..., device='cuda')`) subsequently runs in the same
    # process, it can observe ELF-symbol-interposed CUDA runtime symbols and
    # fail with an AttributeError/NameError deep inside torch.cuda, unrelated
    # to DLPack itself. Warming torch's CUDA context before jittor is ever
    # imported avoids it, but a test process that already imported jittor
    # (as this whole file does) cannot retroactively do that -- so CUDA+torch
    # tests probe readiness here and skip with a clear reason instead of
    # crashing the suite when the hazard is present in this process' import
    # order. See jittor-core-gaps.md section 3.4 for the full writeup.
    if not has_torch or not jt.compiler.has_cuda:
        return False
    try:
        torch.zeros(1, device="cuda")
        return True
    except Exception:
        return False

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

    # jittor-core-gaps.md 2026-09-05 §3.4: non-contiguous import used to be
    # rejected outright; jt.from_dlpack() now materializes a real,
    # correctly-valued contiguous copy instead (see its docstring for the
    # full design -- Jittor's Var still has no stride concept, so this can
    # never be zero-copy, but the *values* are exactly correct for any
    # stride pattern, not just the common "transpose/basic slice" case).
    def test_non_contiguous_import_transpose(self):
        a = np.arange(20, dtype=np.float32).reshape(4, 5)
        t = a.T
        self.assertFalse(t.flags["C_CONTIGUOUS"])
        y = jt.from_dlpack(t)
        np.testing.assert_array_equal(y.numpy(), t)

    def test_non_contiguous_import_step_slice(self):
        a = np.arange(20, dtype=np.float32).reshape(4, 5)
        sliced = a[:, ::2]
        self.assertFalse(sliced.flags["C_CONTIGUOUS"])
        y = jt.from_dlpack(sliced)
        np.testing.assert_array_equal(y.numpy(), sliced)

    def test_non_contiguous_import_negative_stride(self):
        # A reversed view has a genuinely negative stride -- exercises the
        # min/max flat-span computation's negative-offset branch, not just
        # the common "all strides >= 0" case.
        a = np.arange(20, dtype=np.float64).reshape(4, 5)
        rev = a[:, ::-1]
        self.assertLess(rev.strides[1], 0)
        y = jt.from_dlpack(rev)
        np.testing.assert_array_equal(y.numpy(), rev)

    def test_non_contiguous_import_broadcast_stride_zero(self):
        # A broadcast view has stride 0 along the broadcast axis -- every
        # logical element aliases the same source element.
        a = np.arange(4, dtype=np.float32).reshape(4, 1)
        b = np.broadcast_to(a, (4, 6))
        self.assertEqual(b.strides[1], 0)
        y = jt.from_dlpack(b)
        np.testing.assert_array_equal(y.numpy(), b)

    def test_non_contiguous_import_3d_mixed_strides(self):
        a = np.arange(60, dtype=np.float64).reshape(3, 4, 5)
        mixed = a[:, ::2, ::-1]  # one plain axis, one step, one reversed
        y = jt.from_dlpack(mixed)
        np.testing.assert_array_equal(y.numpy(), mixed)

    def test_non_contiguous_import_empty(self):
        a = np.arange(20, dtype=np.float32).reshape(4, 5)
        empty = a[:0, ::2]
        y = jt.from_dlpack(empty)
        self.assertEqual(tuple(y.shape), empty.shape)
        np.testing.assert_array_equal(y.numpy(), empty)

    def test_non_contiguous_import_is_independent_copy(self):
        # Unlike the contiguous fast path (test_cpu_zero_copy_mutation_
        # visible_on_import), a materialized strided import cannot alias the
        # source -- mutating the source afterwards must NOT be visible.
        a = np.arange(20, dtype=np.float32).reshape(4, 5)
        view = a.T
        y = jt.from_dlpack(view)
        a[0, 0] = 999.0
        self.assertNotEqual(y.numpy()[0, 0], 999.0)

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

    def test_dlpack_copy_true_materializes_independent_buffer(self):
        x = jt.array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        cap = x.dlpack(copy=True)
        y = jt.core.from_dlpack_capsule(cap)
        np.testing.assert_allclose(y.numpy(), [1, 2, 3])
        # mutating the original after export must NOT be visible through the
        # copy -- unlike the default zero-copy export, copy=True must own
        # independent memory.
        x[0] = 999.0
        x.sync()
        np.testing.assert_allclose(y.numpy(), [1, 2, 3])

    def test_dlpack_copy_true_zero_size(self):
        z = jt.array(np.array([], dtype=np.float32))
        cap = z.dlpack(copy=True)
        y = jt.core.from_dlpack_capsule(cap)
        self.assertEqual(tuple(y.shape), (0,))
        np.testing.assert_allclose(y.numpy(), [])

    def test_dlpack_versioned_capsule_roundtrip(self):
        # DLPack>=0.8's DLManagedTensorVersioned, requested the way real
        # consumers (NumPy>=2/CuPy/PyTorch's own __dlpack__) do.
        x = jt.array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        cap = x.dlpack(max_version=(1, 0))
        import ctypes
        name = ctypes.pythonapi.PyCapsule_GetName
        name.restype = ctypes.c_char_p
        name.argtypes = [ctypes.py_object]
        self.assertEqual(name(cap), b"dltensor_versioned")
        y = jt.core.from_dlpack_capsule(cap)
        np.testing.assert_allclose(y.numpy(), [1, 2, 3])

    def test_dlpack_versioned_capsule_single_consumption(self):
        x = jt.array(np.array([1.0], dtype=np.float32))
        cap = x.dlpack(max_version=(1, 0))
        jt.core.from_dlpack_capsule(cap)
        with self.assertRaises(Exception):
            jt.core.from_dlpack_capsule(cap)

    def test_dlpack_old_max_version_falls_back_to_classic(self):
        x = jt.array(np.array([1.0], dtype=np.float32))
        cap = x.dlpack(max_version=(0, 8))
        import ctypes
        name = ctypes.pythonapi.PyCapsule_GetName
        name.restype = ctypes.c_char_p
        name.argtypes = [ctypes.py_object]
        self.assertEqual(name(cap), b"dltensor")

    def test_from_dlpack_prefers_versioned_capsule_from_numpy(self):
        # jt.from_dlpack() requests max_version=(1,0); NumPy>=2's __dlpack__
        # honors it and hands back a versioned capsule -- exercise the real
        # negotiated path, not just a directly-built jittor->jittor capsule.
        src = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = jt.from_dlpack(src)
        np.testing.assert_allclose(y.numpy(), [1, 2, 3])
        src[0] = 42.0
        np.testing.assert_allclose(y.numpy(), [42, 2, 3])


@unittest.skipIf(not has_torch, "No PyTorch found")
class TestDLPackTorch(unittest.TestCase):
    def test_cpu_roundtrip_dtype_coverage_via_torch_from_dlpack(self):
        for dt in [torch.float32, torch.float64, torch.int32, torch.int64, torch.bool]:
            src = torch.tensor([True, False, True], dtype=dt) if dt is torch.bool \
                else torch.arange(1, 4, dtype=dt)
            y = jt.from_dlpack(src)
            np.testing.assert_array_equal(y.numpy(), src.numpy())

    def test_jittor_to_torch_zero_copy_mutation_visible(self):
        jt.flags.use_cuda = 0
        x = jt.array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        t = torch.from_dlpack(x)
        t[0] = 42.0
        np.testing.assert_allclose(x.numpy(), [42.0, 2.0, 3.0])

    # jittor-core-gaps.md 2026-09-05 §3.4: a real, non-toy producer of
    # non-contiguous DLPack tensors -- torch's own __dlpack__() happily
    # exports a transposed/permuted/step-sliced tensor with real, non-null
    # strides (unlike, say, a framework that refuses to export those at
    # all), which is exactly the "supported on the wire, unsupported on
    # import" gap this fix closes.
    def test_non_contiguous_torch_transpose(self):
        jt.flags.use_cuda = 0
        t = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        tt = t.T
        self.assertFalse(tt.is_contiguous())
        y = jt.from_dlpack(tt)
        np.testing.assert_array_equal(y.numpy(), tt.numpy())

    def test_non_contiguous_torch_step_slice(self):
        jt.flags.use_cuda = 0
        t = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        s = t[:, ::2]
        y = jt.from_dlpack(s)
        np.testing.assert_array_equal(y.numpy(), s.numpy())

    def test_non_contiguous_torch_permute_3d(self):
        jt.flags.use_cuda = 0
        t = torch.arange(60, dtype=torch.float64).reshape(3, 4, 5).permute(2, 0, 1)
        self.assertFalse(t.is_contiguous())
        y = jt.from_dlpack(t)
        np.testing.assert_array_equal(y.numpy(), t.numpy())

    def test_non_contiguous_torch_expand_stride_zero(self):
        jt.flags.use_cuda = 0
        t = torch.arange(4, dtype=torch.float32).reshape(4, 1).expand(4, 6)
        self.assertEqual(t.stride(1), 0)
        y = jt.from_dlpack(t)
        np.testing.assert_array_equal(y.numpy(), t.numpy())

    def test_torch_to_jittor_zero_copy_mutation_visible(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        y = jt.from_dlpack(t)
        t[0] = 77.0
        np.testing.assert_allclose(y.numpy(), [77.0, 2.0, 3.0])


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


class TestDLPackTorchCuda(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _torch_cuda_ready():
            raise unittest.SkipTest(
                "torch CUDA lazy-init failed in this process -- see "
                "jittor-core-gaps.md section 3.4 (RTLD_GLOBAL/ELF symbol "
                "interposition hazard between jittor_core and a co-loaded "
                "PyTorch CUDA runtime when jittor is imported first)")

    def setUp(self):
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.flags.use_cuda = 0

    def test_roundtrip_dtype_coverage(self):
        for dt in [torch.float32, torch.float64, torch.int32, torch.int64,
                   torch.bool, torch.complex64, torch.complex128]:
            if dt is torch.bool:
                src = torch.tensor([True, False, True], device="cuda")
            elif dt in (torch.complex64, torch.complex128):
                src = torch.tensor([1 + 2j, 3 + 4j], dtype=dt, device="cuda")
            else:
                src = torch.arange(1, 4, dtype=dt, device="cuda")
            y = jt.from_dlpack(src)
            self.assertEqual(y.device_id(), 0)
            np.testing.assert_array_equal(y.numpy(), src.cpu().numpy())

    def test_zero_copy_mutation_visible_both_ways(self):
        x = jt.array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        x.sync()
        t = torch.from_dlpack(x)
        self.assertEqual(t.device.type, "cuda")
        t[0] = 99.0
        np.testing.assert_allclose(x.numpy(), [99.0, 2.0, 3.0])

    # jittor-core-gaps.md 2026-09-05 §3.4: the strided-import fallback's
    # device-side gather must actually run on CUDA (not silently fall back
    # to a host round-trip) for a CUDA producer -- checked both by asserting
    # correct values AND that the resulting Var's own location is "device".
    def test_non_contiguous_torch_cuda_transpose(self):
        t = torch.arange(24, dtype=torch.float32, device="cuda").reshape(4, 6)
        tt = t.T
        y = jt.from_dlpack(tt)
        y.sync()
        self.assertEqual(y.location(), "device")
        self.assertEqual(y.device_id(), 0)
        np.testing.assert_array_equal(y.numpy(), tt.cpu().numpy())

    def test_non_contiguous_torch_cuda_permute_3d(self):
        t = torch.arange(60, dtype=torch.float64, device="cuda").reshape(3, 4, 5).permute(2, 0, 1)
        y = jt.from_dlpack(t)
        np.testing.assert_array_equal(y.numpy(), t.cpu().numpy())

    def test_non_contiguous_torch_cuda_step_slice(self):
        t = torch.arange(24, dtype=torch.float32, device="cuda").reshape(4, 6)
        s = t[:, ::2]
        y = jt.from_dlpack(s)
        np.testing.assert_array_equal(y.numpy(), s.cpu().numpy())

    def test_multi_gpu_device_id_preserved(self):
        if jt.get_device_count() < 2:
            self.skipTest("needs >=2 GPUs")
        # torch.Tensor.__dlpack__() itself refuses to export unless
        # torch.cuda.current_device() matches the tensor's device (a
        # torch-side restriction, not a jittor one) -- keep device 1 current
        # for both the allocation and the export call.
        with torch.cuda.device(1):
            t = torch.tensor([1.0, 2.0, 3.0], device="cuda")
            y = jt.from_dlpack(t)
        self.assertEqual(y.device_id(), 1)
        np.testing.assert_allclose(y.numpy(), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()

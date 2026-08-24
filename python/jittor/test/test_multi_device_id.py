# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# Tests for explicit per-tensor device id (jittor-core-gaps.md section 3.8):
# a Var can be pinned to a specific physical GPU, always reports its real
# residency via device_id()/location(), migrates correctly across CPU/GPUs,
# and ops combining Vars pinned to different devices fail loud instead of
# silently running on the wrong CUDA context.
#
# Every test branches on jt.get_device_count(): with >=2 GPUs it exercises a
# genuine cross-GPU scenario; with exactly 1 GPU it substitutes a CPU-vs-GPU0
# pair so the same code paths (pin, mismatch detection, migration) are still
# exercised instead of being skipped outright.
import unittest
import numpy as np
import jittor as jt


@unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
class TestMultiDeviceId(unittest.TestCase):
    def setUp(self):
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.flags.use_cuda = 0

    def _two_distinct_devices(self):
        """Return (dev_a, dev_b), two distinct device ids to pin Vars to.
        Real (gpu, gpu) pair when >=2 GPUs are visible, else (cpu, gpu0)."""
        if jt.get_device_count() >= 2:
            return 0, 1
        return -1, 0

    def test_device_id_query_cpu(self):
        jt.flags.use_cuda = 0
        a = jt.array([1.0, 2.0, 3.0])
        a.sync()
        self.assertEqual(a.device_id(), -1)
        self.assertEqual(a.location(), "cpu")

    def test_device_id_explicit_gpu(self):
        a = jt.array([1.0, 2.0, 3.0], device="cuda:0")
        a.sync()
        self.assertEqual(a.device_id(), 0)
        self.assertEqual(a.location(), "device")

    def test_migrate_to_device_round_trip(self):
        dev_a, dev_b = self._two_distinct_devices()
        data = np.arange(6, dtype=np.float32).reshape(2, 3)
        a = jt.array(data).migrate_to_device(dev_a)
        self.assertEqual(a.device_id(), dev_a)
        np.testing.assert_allclose(a.numpy(), data)

        a2 = jt.array(data).migrate_to_device(dev_a).migrate_to_device(dev_b)
        self.assertEqual(a2.device_id(), dev_b)
        np.testing.assert_allclose(a2.numpy(), data)

        a3 = jt.array(data).migrate_to_device(dev_b).migrate_to_device(dev_a)
        self.assertEqual(a3.device_id(), dev_a)
        np.testing.assert_allclose(a3.numpy(), data)

    def test_pinned_op_executes_on_pinned_device(self):
        dev_a, _ = self._two_distinct_devices()
        a = jt.array(np.array([1.0, 2.0, 3.0], dtype=np.float32)).migrate_to_device(dev_a)
        b = jt.array(np.array([10.0, 20.0, 30.0], dtype=np.float32)).migrate_to_device(dev_a)
        c = a + b
        c.sync()
        did = c.device_id()
        vals = c.numpy()
        self.assertEqual(did, dev_a)
        np.testing.assert_allclose(vals, [11, 22, 33])

    def test_pinned_var_resident_on_cpu_migrates_to_its_pin(self):
        # Regression: a Var pinned to a specific GPU can still legitimately be
        # CPU-resident at op-dispatch time (e.g. after an explicit
        # migrate_to_cpu() that doesn't clear the pin). The op's CPU->GPU
        # migration branch must send it to the PINNED device, not whatever
        # get_allocator()'s ambient ("ambient" == last-set jt.flags.device_id
        # / current CUDA context) ends up being.
        dev_a, _ = self._two_distinct_devices()
        b = jt.array(np.array([10.0, 20.0, 30.0], dtype=np.float32)).migrate_to_device(dev_a)
        b.migrate_to_cpu()
        self.assertEqual(b.location(), "cpu")
        c = b + 1
        c.sync()
        did = c.device_id()
        vals = c.numpy()
        self.assertEqual(did, dev_a)
        np.testing.assert_allclose(vals, [11, 21, 31])

    def test_mixed_device_op_fails_loud(self):
        dev_a, dev_b = self._two_distinct_devices()
        a = jt.array(np.array([1.0], dtype=np.float32)).migrate_to_device(dev_a)
        b = jt.array(np.array([2.0], dtype=np.float32)).migrate_to_device(dev_b)
        with self.assertRaises(Exception):
            c = a + b
            c.sync()

    def test_grad_preserves_device(self):
        # A lone CPU pin (no conflicting GPU pin on the same op) is legally
        # promoted to GPU under global use_cuda=1 -- same pre-existing
        # semantics as any CPU array participating in GPU computation.
        # Device preservation is only meaningful to assert for a real GPU
        # pin, so this always pins to GPU 0 (guaranteed to exist: the whole
        # class is skipped when there is no CUDA device at all).
        dev_a = 0
        x = jt.array(np.array([1.0, 2.0, 3.0], dtype=np.float32)).migrate_to_device(dev_a)
        x.requires_grad = True
        y = (x * 2).sum()
        gx = jt.grad(y, x)
        gx.sync()
        self.assertEqual(gx.device_id(), dev_a)
        np.testing.assert_allclose(gx.numpy(), [2, 2, 2])

    def test_out_of_range_device_rejected(self):
        with self.assertRaises(Exception):
            jt.array([1.0], device=f"cuda:{jt.get_device_count() + 10}")


@unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
class TestMultiDeviceIdTorchCompat(unittest.TestCase):
    def setUp(self):
        jt.use_torch = True
        import torch
        self.torch = torch
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.flags.use_cuda = 0

    def test_tensor_device_index_roundtrip(self):
        torch = self.torch
        dev = 0
        t = torch.tensor([1.0, 2.0, 3.0], device=f"cuda:{dev}")
        self.assertEqual(str(t.device), f"cuda:{dev}")
        self.assertEqual(t.get_device(), dev)

    def test_to_moves_between_devices(self):
        torch = self.torch
        if jt.get_device_count() < 2:
            self.skipTest("needs >=2 GPUs for a real cross-device .to()")
        t = torch.tensor([1.0, 2.0, 3.0], device="cuda:0")
        t2 = t.to("cuda:1")
        self.assertEqual(str(t2.device), "cuda:1")
        np.testing.assert_allclose(t2.numpy(), [1, 2, 3])

    def test_cpu_device_reports_minus_one(self):
        torch = self.torch
        t = torch.tensor([1.0], device="cpu")
        self.assertEqual(str(t.device), "cpu")
        self.assertEqual(t.get_device(), -1)


if __name__ == "__main__":
    unittest.main()

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
import signal
import subprocess
import sys
import textwrap
import unittest
import numpy as np
import jittor as jt


def _run_subprocess_ignoring_jittor_sigchld(args, **kwargs):
    """subprocess.run(), but temporarily resets SIGCHLD to its default
    disposition around the call.

    jittor installs a process-wide SIGCHLD handler (src/utils/log.cc,
    register_sigaction) that treats ANY non-clean child-process death as its
    own compiler-worker pool running out of memory and quick_exit()s the
    current process -- this test file imports jittor at module level, so
    that handler is already installed in the *test runner* process itself.
    Spawning a child that we expect to crash (as the regression tests below
    deliberately do) would otherwise take the whole test run down with it.
    Resetting SIGCHLD to SIG_DFL for the duration of the call is safe here:
    Python's subprocess module reaps children via waitpid() regardless of
    the signal disposition, and jittor's own internal worker pool isn't in
    use in this process to begin with.
    """
    old_handler = signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    try:
        return subprocess.run(args, **kwargs)
    finally:
        signal.signal(signal.SIGCHLD, old_handler)


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

    def test_three_way_cpu_gpu0_gpu1_coexistence(self):
        # §3.8 criterion 1 literally: CPU + GPU0 + GPU1 alive simultaneously
        # in one process, each reporting its own real device_id(). The
        # existing _two_distinct_devices() helper only ever holds two of the
        # three at once (it drops CPU whenever >=2 GPUs are visible).
        if jt.get_device_count() < 2:
            self.skipTest("needs >=2 GPUs for a genuine 3-way CPU/GPU0/GPU1 test")
        cpu_t = jt.array(np.array([1.0, 2.0], dtype=np.float32), device="cpu")
        gpu0_t = jt.array(np.array([3.0, 4.0], dtype=np.float32), device="cuda:0")
        gpu1_t = jt.array(np.array([5.0, 6.0], dtype=np.float32), device="cuda:1")
        cpu_t.sync(); gpu0_t.sync(); gpu1_t.sync()
        self.assertEqual(cpu_t.device_id(), -1)
        self.assertEqual(gpu0_t.device_id(), 0)
        self.assertEqual(gpu1_t.device_id(), 1)
        np.testing.assert_allclose(cpu_t.numpy(), [1, 2])
        np.testing.assert_allclose(gpu0_t.numpy(), [3, 4])
        np.testing.assert_allclose(gpu1_t.numpy(), [5, 6])

    def test_no_process_restart_on_device_migration(self):
        # §3.8 criterion 5: ordinary per-tensor device moves must not restart
        # the process (unlike the legacy global jt.flags.device_id path,
        # which execvp-restarts -- src/misc/cuda_flags.cc setter_device_id).
        import os
        pid_before = os.getpid()
        dev_a, dev_b = self._two_distinct_devices()
        a = jt.array(np.array([1.0, 2.0], dtype=np.float32)).migrate_to_device(dev_a)
        a.migrate_to_device(dev_b)
        a.migrate_to_cpu()
        a.migrate_to_device(dev_a)
        self.assertEqual(os.getpid(), pid_before)

    def test_save_load_preserves_device_id(self):
        # jt.save/jt.load used to drop device identity entirely (both the
        # native Var.__reduce__ pickle path and the torch-compat
        # _to_portable/_from_portable checkpoint path only carried a numpy
        # array + dtype); a GPU1-pinned Var silently came back on GPU0/CPU.
        dev_a, _ = self._two_distinct_devices()
        if dev_a < 0:
            self.skipTest("needs a real GPU pin (no second device to distinguish from CPU)")
        import tempfile, os
        p = jt.array(np.array([1.0, 2.0, 3.0], dtype=np.float32)).migrate_to_device(dev_a)
        path = tempfile.mktemp(suffix=".pkl")
        try:
            jt.save({"p": p}, path)
            loaded = jt.load(path)
            self.assertEqual(loaded["p"].device_id(), dev_a)
            np.testing.assert_allclose(loaded["p"].numpy(), p.numpy())
        finally:
            if os.path.exists(path):
                os.remove(path)

    # -- Regression for jittor-core-gaps.md §9.4-bis --------------------
    # NodeFlags::_device_tag (Var-only, bits 13-18, encodes an explicit
    # device pin: 0=unset, 1=CPU, n+2=GPU n) numerically aliases
    # NodeFlags::_has_gopt and four other op-only flags at the very same
    # bit positions (see src/node.h): a Var pinned to CPU (pin_value=1) or
    # to an odd-numbered GPU (pin_value=device_id+2, odd) has bit13
    # spuriously set. Executor::run_sync's graph-optimize BFS pass
    # (executor.cc, the `while(1)` loop that scans bfs_q for _has_gopt)
    # used to read that bit without first checking is_var(), so such a Var
    # -- reachable in the same run_sync call as any genuinely
    # _has_gopt-flagged Op (setitem/getitem are the only ops that set it)
    # -- got misidentified as an Op needing graph_optimize() and crashed
    # via an invalid virtual call through a Var's vtable (jump through a
    # null function pointer). Fixed by guarding all three _has_gopt reads
    # in executor.cc with `!node->is_var()`.
    #
    # Both tests below run the repro in a subprocess with a timeout: if
    # this regresses, the subprocess crashes (or -- a naive single-line
    # fix that only guards the crash-site read but not the two need_opt
    # accumulation reads would make it hang forever instead, since the
    # spurious bit is then never cleared -- either way the subprocess
    # fails loudly as an assertion/timeout instead of segfaulting or
    # hanging the whole test run.
    def _run_pinned_gopt_repro(self, pin_device_expr):
        script = textwrap.dedent(f"""
            import jittor as jt
            jt.flags.use_cuda = 1
            # A Var carrying the pin at the moment it's swept into
            # run_sync's bfs_q must not yet be finished/synced (an eagerly
            # migrate_to_device()'d Var is already finished and gets
            # skipped by run_sync's seed loop) -- .update() replaces the
            # underlying Var with a freshly computed, unfinished one and
            # (via assign_var, var_holder.cc) carries the old pin onto it,
            # exactly like every jt.optim optimizer's `p.update(...)` /
            # `m.update(...)` does on every step.
            p = jt.array([1.0, 2.0, 3.0]).migrate_to_device({pin_device_expr})
            p2 = p + 0
            p.update(p2)
            a = jt.zeros((3,))
            a[0] = jt.array(5.0)  # setitem_op: the only op that legitimately sets _has_gopt
            jt.sync([p, a])
            print("OK", p.numpy().tolist(), a.numpy().tolist())
        """)
        result = _run_subprocess_ignoring_jittor_sigchld(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0,
            f"pinned-Var + graph-optimize regression (§9.4-bis): "
            f"stdout={result.stdout!r} stderr={result.stderr[-4000:]!r}")
        self.assertIn("OK [1.0, 2.0, 3.0] [5.0, 0.0, 0.0]", result.stdout)

    def test_pinned_var_survives_graph_optimize_pass_cpu(self):
        # CPU pin -> pin_value=1, always odd, always reproducible regardless
        # of how many GPUs are visible.
        self._run_pinned_gopt_repro("-1")

    def test_pinned_var_survives_graph_optimize_pass_odd_gpu(self):
        # GPU 1 -> pin_value=3, odd. Needs a second GPU to pin to.
        if jt.get_device_count() < 2:
            self.skipTest("needs >=2 GPUs for an odd-numbered GPU pin")
        self._run_pinned_gopt_repro("1")

    # Note: an end-to-end "real jt.optim.Adam optimizer over pinned params"
    # black-box variant of this regression (matching how §9.4 was originally
    # reported) was deliberately left out here. Unlike the two targeted
    # tests above, whether that scenario actually crashes on the unfixed
    # code turned out to depend on incidental lazy-execution batching/op-
    # fusion timing (confirmed non-deterministic across otherwise-identical
    # process runs in this environment) -- it is not a reliable check and
    # would give false confidence. The two tests above pin down the exact,
    # deterministic mechanism instead: an explicitly pinned Var reachable in
    # the same run_sync call as a genuinely _has_gopt-flagged op
    # (setitem/getitem).


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

    def test_to_does_not_mutate_source(self):
        # Var._to's device branch called out.migrate_to_device(idx) after
        # `out = self` (copy=False, no dtype change), and migrate_to_device
        # mutates in place -- so t.to(other_device) used to move the SOURCE
        # tensor too (t is t2, t.device became the new device). torch's
        # .to(other_device) must return a distinct tensor and leave the
        # source untouched.
        torch = self.torch
        if jt.get_device_count() < 2:
            self.skipTest("needs >=2 GPUs for a real cross-device .to()")
        t = torch.tensor([1.0, 2.0, 3.0], device="cuda:0")
        t2 = t.to("cuda:1")
        self.assertIsNot(t, t2)
        self.assertEqual(str(t.device), "cuda:0")
        self.assertEqual(str(t2.device), "cuda:1")
        np.testing.assert_allclose(t.numpy(), [1, 2, 3])
        np.testing.assert_allclose(t2.numpy(), [1, 2, 3])

    def test_to_same_device_returns_self(self):
        # the no-op case (torch semantics: .to() on an already-matching
        # device/dtype with copy=False returns the same object).
        torch = self.torch
        t = torch.tensor([1.0, 2.0, 3.0], device="cuda:0")
        t2 = t.to("cuda:0")
        self.assertIs(t, t2)

    def test_cpu_device_reports_minus_one(self):
        torch = self.torch
        t = torch.tensor([1.0], device="cpu")
        self.assertEqual(str(t.device), "cpu")
        self.assertEqual(t.get_device(), -1)


if __name__ == "__main__":
    unittest.main()

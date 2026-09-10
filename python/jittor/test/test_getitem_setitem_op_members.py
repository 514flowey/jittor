# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Dun Liang <randonlang@gmail.com>.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# Regression tests for agent/workdocs/2026-09-05-core-gaps.md sec 3.1 (P0):
# GetitemOp/SetitemOp used to read their primary input exclusively via
# inputs().front()/input(1), which the generic executor's per-op _inputs
# edge-list bookkeeping can legitimately reset once it considers an op
# "finished" -- even though the op can still be revisited for jit_prepare
# later (its output shared by a later, separate sync). That produced a
# 100%-reproducible SIGABRT/segfault (inputs().front() on an empty list)
# for CUDA MPSCircuit-style lazy graphs (getitem/setitem with a tensor
# index feeding further linalg, repeatedly materialized via .item()/
# .numpy() across many partial syncs).
#
# The fix caches the primary input(s) as named Var* members (x for
# GetitemOp, x/y for SetitemOp) populated once at construction, matching
# the idiom every other op (UnaryOp, BinaryOp, ...) already uses, so
# jit_prepare/jit_run never depend on the mutable _inputs edge list.
# This closes the specific "_inputs.size()==0 -> inputs().front() UB"
# crash class and the CUDA compile_optimize regression that fixing it
# naively triggers (see report), but does NOT close a deeper, separately
# documented use-after-free that the same investigation surfaced: see
# agent/workdocs/2026-09-05-p0-getitem-lazy-relay-fix.md sec "residual gap"
# and agent/repros/2026-09-05-p0-mpscircuit-sample-crash.py for the
# still-crashing repro (now failing later, inside jit_run/jit_compile,
# instead of unconditionally at the first jit_prepare call).
import unittest
import numpy as np
import jittor as jt


class TestGetitemSetitemOpMembers(unittest.TestCase):
    def _check_dynamic_getitem_setitem(self):
        a = jt.array([1.0, 2.0, 3.0, 4.0, 5.0])
        idx = jt.array(2)
        self.assertEqual(a[idx].item(), 3.0)

        b = jt.zeros(5)
        b[idx] = 9.0
        np.testing.assert_allclose(b.numpy(), [0, 0, 9, 0, 0])

        c = jt.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c[jt.array(1)] = jt.array([7.0, 8.0])
        np.testing.assert_allclose(c.numpy(), [[1, 2], [7, 8], [5, 6]])

        # matmul/backward composed with a dynamic-index getitem, matching
        # the CUDA compile_optimize path (slice_func) that a naive fix
        # (renaming inputs().front() to a bare member name) breaks: the
        # hoisting heuristic in GetitemOp::_compile_optimize must still
        # recognize `x`/`y` as host-only accessors to extract into kernel
        # arguments, not clone into the free-standing __global__ slice_func.
        w = jt.array(np.random.RandomState(0).randn(3, 3).astype("float32"))
        row = w[idx % 3]
        loss = (row * row).sum()
        gw = jt.grad(loss, w)
        self.assertTrue(np.isfinite(gw.numpy()).all())

    def test_cpu(self):
        jt.flags.use_cuda = 0
        self._check_dynamic_getitem_setitem()

    @unittest.skipIf(not jt.has_cuda, "no cuda found")
    def test_cuda(self):
        jt.flags.use_cuda = 1
        self._check_dynamic_getitem_setitem()
        jt.flags.use_cuda = 0


if __name__ == "__main__":
    unittest.main()

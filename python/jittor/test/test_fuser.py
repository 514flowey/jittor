# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np


class TestFuser(unittest.TestCase):
    def test_wrong_fuse(self):
        a = jt.array([1])
        b = jt.random([10,])
        c = (a * b).sum() + (a + 1)
        print(c)

    def test_wrong_fuse2(self):
        a = jt.array([1])
        b = jt.random([10,])
        c = jt.random([100,])
        bb = a*b
        cc = a*c
        jt.sync([bb,cc])
        np.testing.assert_allclose(b.data, bb.data)
        np.testing.assert_allclose(c.data, cc.data)

    def test_for_fuse(self):
        arr = []
        x = 0
        for i in range(100):
            arr.append(jt.array(1))
            x += arr[-1]
        x.sync()
        for i in range(100):
            # print(arr[i].debug_msg())
            assert ",0)" not in arr[i].debug_msg()

    def test_array_bc(self):
        # a = jt.array(1)
        with jt.profile_scope() as rep:
            b = jt.array(1).broadcast([10])
            b.sync()
        assert len(rep) == 2

    def test_stop_fuse_splits_kernel(self):
        # A chain that fuses into a single kernel by default...
        with jt.profile_scope() as rep:
            x = jt.array([1.0, 2.0, 3.0])
            y = x + 1
            z = y * 2
            w = z - 3
            w.sync()
        baseline = len(rep)
        # ...must split into (at least) two once an explicit fusion boundary
        # is set on an intermediate Var. This is the mechanism explicit
        # per-device Var pinning (VarHolder.migrate_to_device, see
        # test_multi_device_id.py) relies on to guarantee a device-pinned
        # Var is never silently fused across a device boundary, since the
        # op-merge algorithm itself (fuser.cc) isn't directly editable.
        with jt.profile_scope() as rep2:
            x = jt.array([1.0, 2.0, 3.0])
            y = x + 1
            y.stop_fuse()
            z = y * 2
            w = z - 3
            w.sync()
        assert len(rep2) > baseline

    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    def test_device_pin_splits_kernel(self):
        # migrate_to_device() must have the same fusion-splitting effect as
        # stop_fuse(), since it sets that flag as a side effect.
        jt.flags.use_cuda = 1
        try:
            with jt.profile_scope() as rep:
                x = jt.array([1.0, 2.0, 3.0])
                y = x + 1
                z = y * 2
                w = z - 3
                w.sync()
            baseline = len(rep)
            with jt.profile_scope() as rep2:
                x = jt.array([1.0, 2.0, 3.0])
                y = (x + 1).migrate_to_device(0)
                z = y * 2
                w = z - 3
                w.sync()
            assert len(rep2) > baseline
        finally:
            jt.flags.use_cuda = 0


if __name__ == "__main__":
    unittest.main()
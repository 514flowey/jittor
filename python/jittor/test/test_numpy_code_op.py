# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
from jittor import Function
import jittor as jt
import numpy
import ctypes
import sys

try:
    import cupy
except:
    pass

class TestCodeOp(unittest.TestCase):
    def test_func(self):
        class Func(Function):
            def forward_code(self, np, data):
                a = data["inputs"][0]
                b = data["outputs"][0]
                if (jt.flags.use_cuda==0):
                    assert isinstance(a,numpy.ndarray)
                else:
                    assert isinstance(a,cupy.ndarray)
                np.add(a,a,out=b)

            def backward_code(self, np, data):
                a, dout = data["inputs"]
                out = data["outputs"][0]
                np.copyto(out, dout*2.0)

            def execute(self, a):
                self.save_vars = a
                return jt.numpy_code(
                    a.shape,
                    a.dtype,
                    [a],
                    self.forward_code,
                )

            def grad(self, grad_a):
                a = self.save_vars
                return jt.numpy_code(
                    a.shape,
                    a.dtype,
                    [a, grad_a],
                    self.backward_code,
                )

        def check():
            a = jt.random((5,1))
            func = Func()
            b = func(a)
            assert numpy.allclose(b.data,(a+a).data)
            da = jt.grad(b,a)
            one=numpy.ones(a.shape)
            assert numpy.allclose(da.data,one*2.0)

        if jt.has_cuda:
            with jt.flag_scope(use_cuda=1):
                check()
        check()

    def test(self):
        def forward_code(np, data):
            a = data["inputs"][0]
            b = data["outputs"][0]
            if (jt.flags.use_cuda==0):
                assert isinstance(a,numpy.ndarray)
            else:
                assert isinstance(a,cupy.ndarray)
            np.add(a,a,out=b)

        def backward_code(np, data):
            dout = data["dout"]
            out = data["outputs"][0]
            np.copyto(out, dout*2.0)

        def check():
            a = jt.random((5,1))
            b = jt.numpy_code(
                a.shape,
                a.dtype,
                [a],
                forward_code,
                [backward_code],
            )
            assert numpy.allclose(b.data,(a+a).data)
            da = jt.grad(b,a)
            one=numpy.ones(a.shape)
            assert numpy.allclose(da.data,one*2.0)

        if jt.has_cuda:
            with jt.flag_scope(use_cuda=1):
                check()
        check()

    def test_multi_input(self):
        def forward_code(np, data):
            a,b = data["inputs"]
            c,d = data["outputs"]
            np.add(a,b,out=c)
            np.subtract(a,b,out=d)

        def backward_code1(np, data):
            dout = data["dout"]
            out = data["outputs"][0]
            np.copyto(out, dout)

        def backward_code2(np, data):
            dout = data["dout"]
            out_index = data["out_index"]
            out = data["outputs"][0]
            if out_index==0:
                np.copyto(out, dout)
            else:
                np.negative(dout, out)

        def check():
            a = jt.random((5,1))
            b = jt.random((5,1))
            c, d = jt.numpy_code(
                [a.shape, a.shape],
                [a.dtype, a.dtype],
                [a, b],
                forward_code,
                [backward_code1,backward_code2],
            )
            assert numpy.allclose(c.data,(a+b).data)
            assert numpy.allclose(d.data,(a-b).data)
            dca, dcb = jt.grad(c,[a,b])
            dda, ddb = jt.grad(d,[a,b])
            one=numpy.ones(a.shape)
            mone=one*-1.0
            assert numpy.allclose(dca.data,one)
            assert numpy.allclose(dcb.data,one)
            assert numpy.allclose(dda.data,one)
            assert numpy.allclose(ddb.data,mone)
        
        if jt.has_cuda:
            with jt.flag_scope(use_cuda=1):
                check()
        check()

    @unittest.skipIf(True, "Memory leak testing is not in progress, Skip")
    def test_memory_leak(self):
        def forward_code(np, data):
            a,b = data["inputs"]
            c,d = data["outputs"]
            np.add(a,b,out=c)
            np.subtract(a,b,out=d)

        def backward_code1(np, data):
            dout = data["dout"]
            out = data["outputs"][0]
            np.copyto(out, dout)

        def backward_code2(np, data):
            dout = data["dout"]
            out_index = data["out_index"]
            out = data["outputs"][0]
            if out_index==0:
                np.copyto(out, dout)
            else:
                np.negative(dout, out)

        for i in range(1000000):
            a = jt.random((10000,1))
            b = jt.random((10000,1))
            c, d = jt.numpy_code(
                [a.shape, a.shape],
                [a.dtype, a.dtype],
                [a, b],
                forward_code,
                [backward_code1,backward_code2],
            )
            assert numpy.allclose(c.data,(a+b).data)
            assert numpy.allclose(d.data,(a-b).data)
            dca, dcb = jt.grad(c,[a,b])
            dda, ddb = jt.grad(d,[a,b])
            one=numpy.ones(a.shape)
            mone=one*-1.0
            assert numpy.allclose(dca.data,one)
            assert numpy.allclose(dcb.data,one)
            assert numpy.allclose(dda.data,one)
            assert numpy.allclose(ddb.data,mone)

    def test_second_order_grad_fails_loud(self):
        # jittor-core-gaps.md §3.3: a numpy_code op's analytic backward is
        # itself built as a *new* numpy_code op with no backward of its own
        # (NumpyCodeOp::grad uses the NumpyResult-only constructor, which
        # leaves `backward` empty). Requesting a second derivative through
        # such a chain used to index that empty `backward` vector out of
        # bounds and invoke a garbage NumpyFunc, segfaulting the whole
        # process instead of raising a catchable error. This is tested here
        # directly against a raw jt.numpy_code call with no registered
        # backward-of-backward (see below for `jt.linalg.inv`/`det`/`solve`/
        # `cumsum`, which now DO support real 2nd-order AD and are no longer
        # examples of this fail-loud path -- see test_*_gradgradcheck).
        def forward_code(np, data):
            a = data["inputs"][0]
            b = data["outputs"][0]
            np.multiply(a, a, out=b)

        def backward_code(np, data):
            dout = data["dout"]
            a = data["inputs"][0]
            out = data["outputs"][0]
            np.copyto(out, dout * 2.0 * a)

        x = jt.array([[2.0, 3.0], [1.0, 2.0]])
        x.requires_grad = True
        y = jt.numpy_code(x.shape, x.dtype, [x], forward_code, [backward_code]).sum()
        g1 = jt.grad(y, x)
        with self.assertRaises(RuntimeError):
            jt.grad(g1.sum(), x)

    def _gradgradcheck(self, f, x0, eps=1e-4, atol=1e-3, seed=None):
        # jittor-core-gaps.md §3.3 acceptance #1: gradgradcheck each op that
        # now claims real 2nd-order AD, against an independent (not reusing
        # any jittor internals) central-difference oracle. Uses float64 (via
        # auto_convert_64_to_32=0) since the finite-difference reference
        # itself needs headroom well above the analytic result's own
        # rounding error to be a meaningful check.
        with jt.flag_scope(auto_convert_64_to_32=0):
            x = jt.array(x0.copy())
            x.requires_grad = True
            y = f(x)
            loss = (y * y).sum() if seed is None else (y * jt.array(seed)).sum()
            g1 = jt.grad(loss, x, retain_graph=True)
            g2 = jt.grad(g1.sum(), x)

            def g1sum_of(xv):
                xx = jt.array(xv)
                xx.requires_grad = True
                yy = f(xx)
                ll = (yy * yy).sum() if seed is None else (yy * jt.array(seed)).sum()
                return jt.grad(ll, xx).sum().item()

            flat = x0.flatten().copy()
            num_g2 = numpy.zeros_like(flat)
            for i in range(len(flat)):
                p = flat.copy(); p[i] += eps
                m = flat.copy(); m[i] -= eps
                num_g2[i] = (g1sum_of(p.reshape(x0.shape)) - g1sum_of(m.reshape(x0.shape))) / (2 * eps)
            numpy.testing.assert_allclose(g2.numpy().flatten(), num_g2, atol=atol)

    # jittor-core-gaps.md §3.3: this sandbox's jt.flags.use_cuda DEFAULTS TO 1
    # (a GPU is present), so a bare `check()` with no explicit device flag is
    # NOT a CPU check here -- it silently runs on CUDA. That matters because
    # jt.matmul on CUDA has been found to lose float64 precision (~1e-7
    # relative, verified directly against a plain 50x50 jt.matmul vs numpy --
    # a separate, pre-existing Jittor limitation, not something this fix
    # introduces) -- for a bare `eps=1e-4` central-difference gradgradcheck,
    # that ~1e-7 noise in each sample gets divided by `2*eps=2e-4`, i.e.
    # amplified by ~5000x, which is enough to fail a tight `atol=1e-3` (this
    # was caught by test_qr_gradgradcheck failing flakily depending on how
    # the test was invoked, since it happened to compound the matmul noise
    # across a longer op chain than inv/det/solve/cumsum's simpler formulas
    # do). Every gradgradcheck below therefore explicitly pins `use_cuda=0`
    # for the exact-precision CPU check, and uses a larger `eps`/looser
    # `atol` for the separate, explicit CUDA check (larger eps reduces the
    # cancellation-noise amplification at the cost of a bit more finite-
    # difference truncation error, which is fine at looser atol).
    def test_inv_gradgradcheck(self):
        # jittor-core-gaps.md §3.3: jt.linalg.inv's backward is now expressed
        # with ordinary differentiable jittor ops (matmul/transpose) and a
        # live recursive call to `inv`, instead of an opaque numpy_code
        # backward-of-backward -- so jt.grad(jt.grad(loss, x), x) computes a
        # real second derivative rather than raising.
        def check(eps=1e-4, atol=1e-3):
            x0 = numpy.array([[4.0, 1.0], [2.0, 3.0]])
            self._gradgradcheck(jt.linalg.inv, x0, eps=eps, atol=atol)
            xb0 = numpy.stack([x0, x0 * 1.5 + numpy.eye(2)])
            self._gradgradcheck(jt.linalg.inv, xb0, eps=eps, atol=atol)
        with jt.flag_scope(use_cuda=0):
            check()
        if jt.has_cuda:
            with jt.flag_scope(use_cuda=1):
                check(eps=1e-2, atol=3e-2)

    def test_det_gradgradcheck(self):
        def check(eps=1e-4, atol=1e-3):
            x0 = numpy.array([[4.0, 1.0], [2.0, 3.0]])
            self._gradgradcheck(jt.linalg.det, x0, eps=eps, atol=atol)
            xb0 = numpy.stack([x0, x0 * 1.5 + numpy.eye(2)])
            self._gradgradcheck(jt.linalg.det, xb0, eps=eps, atol=atol)
        with jt.flag_scope(use_cuda=0):
            check()
        if jt.has_cuda:
            with jt.flag_scope(use_cuda=1):
                check(eps=1e-2, atol=3e-2)

    def test_solve_gradgradcheck(self):
        # differentiate wrt `a` with `b` held fixed (a closure, matching how
        # the doc's acceptance criteria frame per-argument gradgradcheck).
        def check(eps=1e-4, atol=1e-3):
            a0 = numpy.array([[4.0, 1.0], [2.0, 3.0]])
            b0 = jt.array(numpy.array([1.0, 2.0]))
            self._gradgradcheck(lambda a: jt.linalg.solve(a, b0), a0, eps=eps, atol=atol)
        with jt.flag_scope(use_cuda=0):
            check()
        if jt.has_cuda:
            with jt.flag_scope(use_cuda=1):
                check(eps=1e-2, atol=3e-2)

    def test_cumsum_gradgradcheck(self):
        # cumsum ITSELF is linear (its Jacobian doesn't depend on x, unlike
        # inv/det/solve's), so this exercises the "no live-x reconnection
        # needed" half of the fix rather than the "_reconnect trick" half --
        # the loss below is sum(cumsum(x)^2), which is quadratic in x (via
        # the squaring) and so has a real, generally nonzero constant
        # Hessian; this checks that Hessian against finite differences.
        def check(eps=1e-4, atol=1e-3):
            c0 = numpy.array([1.0, 2.0, 3.0, 4.0])
            self._gradgradcheck(lambda x: jt.misc.numpy_cumsum(x, 0), c0, eps=eps, atol=atol)
        with jt.flag_scope(use_cuda=0):
            check()
        if jt.has_cuda:
            with jt.flag_scope(use_cuda=1):
                check(eps=1e-2, atol=3e-2)

    def _gradgradcheck_multi(self, f, x0, eps=1e-4, atol=1e-3):
        # Like _gradgradcheck, but for a multi-output function (qr/rq
        # return a 2-tuple); loss = sum(y0^2) + 0.5*sum(y1^2) so BOTH
        # outputs' cotangents are simultaneously nonzero -- this is what
        # exercises the `gq`/`gr` combined-formula path in `_QR.grad`
        # (see qr's own docstring) rather than only one `out_index` branch.
        with jt.flag_scope(auto_convert_64_to_32=0):
            def loss_of(xv):
                xx = jt.array(xv)
                xx.requires_grad = True
                y0, y1 = f(xx)
                loss = (y0 * y0).sum() + (y1 * y1).sum() * 0.5
                return xx, jt.grad(loss, xx, retain_graph=True)
            xx, g1 = loss_of(x0)
            g2 = jt.grad(g1.sum(), xx)

            def g1sum_of(xv):
                return loss_of(xv)[1].sum().item()

            flat = x0.flatten().copy()
            num_g2 = numpy.zeros_like(flat)
            for i in range(len(flat)):
                p = flat.copy(); p[i] += eps
                m = flat.copy(); m[i] -= eps
                num_g2[i] = (g1sum_of(p.reshape(x0.shape)) - g1sum_of(m.reshape(x0.shape))) / (2 * eps)
            numpy.testing.assert_allclose(g2.numpy().flatten(), num_g2, atol=atol)

    def test_qr_gradgradcheck(self):
        # jittor-core-gaps.md §3.2/§3.3: qr's backward is now expressed with
        # ordinary differentiable jittor ops (solve/matmul/transpose/tril)
        # and a live recursive call to `qr`, covering square, tall (M>=N),
        # and wide (M<N) shapes (the wide case exercises the R1/R2-split
        # branch of `_QR.grad`).
        def check(eps=1e-4, atol=1e-3):
            rng = numpy.random.RandomState(2)
            self._gradgradcheck_multi(jt.linalg.qr, rng.randn(3, 3), eps=eps, atol=atol)
            self._gradgradcheck_multi(jt.linalg.qr, rng.randn(4, 3), eps=eps, atol=atol)
            self._gradgradcheck_multi(jt.linalg.qr, rng.randn(3, 5), eps=eps, atol=atol)
        with jt.flag_scope(use_cuda=0):
            check()
        if jt.has_cuda:
            with jt.flag_scope(use_cuda=1):
                check(eps=1e-2, atol=5e-2)

    def test_rq_gradgradcheck(self):
        # rq is implemented purely as a composition of flip/transpose/qr
        # (linalg.py::rq), so it inherits real higher-order AD "for free"
        # once qr itself has it -- no rq-specific backward code exists to
        # get wrong.
        def check(eps=1e-4, atol=1e-3):
            rng = numpy.random.RandomState(3)
            self._gradgradcheck_multi(jt.linalg.rq, rng.randn(4, 3), eps=eps, atol=atol)
        with jt.flag_scope(use_cuda=0):
            check()
        if jt.has_cuda:
            with jt.flag_scope(use_cuda=1):
                check(eps=1e-2, atol=5e-2)

    def test_inv_third_order_grad(self):
        # jittor-core-gaps.md §3.3: the fix recurses through the same
        # (now higher-order-differentiable) public function at every level,
        # so it is not hardcoded to exactly 2 orders -- verify a 3rd
        # derivative too, against a scalar closed-form 1/x case where the
        # analytic 3rd derivative is trivial to state independently
        # (d^3(1/x)/dx^3 = -6/x^4).
        with jt.flag_scope(auto_convert_64_to_32=0):
            x = jt.array(2.0)
            x.requires_grad = True
            y = jt.linalg.inv(x.reshape(1, 1)).reshape(())
            g1 = jt.grad(y, x, retain_graph=True)
            g2 = jt.grad(g1, x, retain_graph=True)
            g3 = jt.grad(g2, x)
            self.assertAlmostEqual(g1.item(), -1 / 2.0**2, places=6)
            self.assertAlmostEqual(g2.item(), 2 / 2.0**3, places=6)
            self.assertAlmostEqual(g3.item(), -6 / 2.0**4, places=6)


if __name__ == "__main__":
    unittest.main()
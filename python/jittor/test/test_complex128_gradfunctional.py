"""Native complex128 through gradfunctional, mirroring
test_complex64_gradfunctional.py at double precision (jittor-core-gaps.md
SS3.2 acceptance #5: "一阶 VJP 和第 3.3 节的二阶/JVP 测试同时覆盖 complex64、
complex128"). Locks that jittor.gradfunctional.{vjp, jvp} and native
double-backward correctly differentiate the native complex128 dtype, not just
complex64 -- same double-backward machinery, same Wirtinger/conjugate grad
convention, on BOTH CPU and CUDA.

No real torch in this env: numpy is the oracle. Tolerances are tighter than
the complex64 suite (float64 double-backward carries ~1e-10 relative
precision through the same graph).

Run:  python -m jittor.test.test_complex128_gradfunctional
"""
import unittest
import numpy as np
import jittor as jt
from jittor.gradfunctional import vjp, jvp

_DEVICES = [("cpu", 0)] + ([("cuda", 1)] if jt.has_cuda else [])


def both_devices(fn):
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


def _to_complex(g):
    return np.asarray(g.numpy())


def _np_complex(rng, shape):
    return (rng.randn(*shape) + 1j * rng.randn(*shape)).astype("complex128")


def _fd_vjp(fnp, a, v, eps=1e-6):
    """Finite-difference of the real seeded loss L(z) = sum(Re f(z)*Re v + Im f(z)*Im v),
    w.r.t. real and imag parts of a, returned as a complex grad (the vjp value)."""
    a = a.astype("complex128")
    zr0 = a.real.astype(np.float64)
    zi0 = a.imag.astype(np.float64)

    def loss(zr, zi):
        fo = fnp((zr + 1j * zi).astype("complex128"))
        return float(np.sum(fo.real * v.real + fo.imag * v.imag))

    gr = np.zeros(a.shape, dtype=np.float64)
    gi = np.zeros(a.shape, dtype=np.float64)
    it = np.ndindex(*a.shape)
    for idx in it:
        zr = zr0.copy(); zr[idx] += eps
        zrm = zr0.copy(); zrm[idx] -= eps
        gr[idx] = (loss(zr, zi0) - loss(zrm, zi0)) / (2 * eps)
        zi = zi0.copy(); zi[idx] += eps
        zim = zi0.copy(); zim[idx] -= eps
        gi[idx] = (loss(zr0, zi) - loss(zr0, zim)) / (2 * eps)
    return gr + 1j * gi


class TestComplex128GradFunctional(unittest.TestCase):
    # -------------------------------------------------------------- vjp: exp().sum(1)
    def test_vjp_exp_complex_input(self):
        rng = np.random.RandomState(0)
        s = (5, 6)
        a = _np_complex(rng, s)
        v = _np_complex(rng, (5,))

        def f(x):
            return x.exp().sum(1)

        def fnp(z):
            return np.exp(z).sum(1)

        fd = _fd_vjp(fnp, a, v)

        def body(dev):
            out, g = vjp(f, jt.array(a), jt.array(v), create_graph=True)
            self.assertEqual(str(out.dtype), "complex128", f"out dtype {dev}")
            self.assertEqual(str(g.dtype), "complex128", f"vjp dtype {dev}")
            self.assertEqual(tuple(g.shape), s, f"vjp shape {dev}")
            gnp = _to_complex(g)
            self.assertTrue(np.isfinite(gnp).all(), f"vjp finite {dev}")
            np.testing.assert_allclose(gnp, fd, atol=1e-6, rtol=1e-6,
                                       err_msg=f"vjp vs finite-diff {dev}")
        both_devices(body)

    # -------------------------------------------------------- vjp: closed form (z*z)
    def test_vjp_square_closed_form(self):
        rng = np.random.RandomState(4)
        s = (4, 3)
        a = _np_complex(rng, s)
        v = _np_complex(rng, s)
        closed = np.conj(2 * a) * v        # Wirtinger conj convention, see complex64 test

        def f(x):
            return x * x

        def body(dev):
            out, g = vjp(f, jt.array(a), jt.array(v), create_graph=True)
            self.assertEqual(str(out.dtype), "complex128", f"out dtype {dev}")
            gnp = _to_complex(g)
            np.testing.assert_allclose(gnp, closed, atol=1e-8, rtol=1e-8,
                                       err_msg=f"vjp(z*z) vs closed form {dev}")
        both_devices(body)

    # ------------------------------------------------------------ vjp: two complex inputs
    def test_vjp_two_complex_inputs(self):
        rng = np.random.RandomState(2)
        s = (4, 5)
        x = _np_complex(rng, s)
        y = _np_complex(rng, s)
        v = _np_complex(rng, s)
        w1 = np.array(0.7 + 0.4j, dtype="complex128")
        w2 = np.array(-0.3 + 0.9j, dtype="complex128")

        def f(a, b):
            return jt.array(w1) * a + jt.array(w2) * b

        def body(dev):
            out, (gx, gy) = vjp(f, (jt.array(x), jt.array(y)), jt.array(v),
                                create_graph=True)
            self.assertEqual(str(out.dtype), "complex128", f"out dtype {dev}")
            self.assertEqual(str(gx.dtype), "complex128", f"gx dtype {dev}")
            self.assertEqual(str(gy.dtype), "complex128", f"gy dtype {dev}")
            gxnp, gynp = _to_complex(gx), _to_complex(gy)
            np.testing.assert_allclose(gxnp, np.conj(w1) * v, atol=1e-8, rtol=1e-8,
                                       err_msg=f"two-complex grad x {dev}")
            np.testing.assert_allclose(gynp, np.conj(w2) * v, atol=1e-8, rtol=1e-8,
                                       err_msg=f"two-complex grad y {dev}")
        both_devices(body)

    # --------------------------------------------- jvp: native complex128 -> numeric pass
    def _fd_jvp(self, fnp, z, v, eps=1e-6):
        return (fnp(z + eps * v) - fnp(z - eps * v)) / (2 * eps)

    def test_jvp_native_complex_holomorphic(self):
        rng = np.random.RandomState(3)
        s = (5, 6)
        a = _np_complex(rng, s)
        vin = _np_complex(rng, s)

        def f(x):
            return x.exp().sum(1)

        ref = self._fd_jvp(lambda z: np.exp(z).sum(1), a, vin)

        def body(dev):
            out, j = jvp(f, jt.array(a), jt.array(vin), create_graph=True)
            self.assertEqual(str(out.dtype), "complex128", f"out dtype {dev}")
            self.assertEqual(str(j.dtype), "complex128", f"jvp dtype {dev}")
            self.assertEqual(tuple(j.shape), (5,), f"jvp shape {dev}")
            got = np.asarray(j.numpy())
            self.assertTrue(np.isfinite(got).all(), f"jvp finite {dev}")
            np.testing.assert_allclose(got, ref, atol=1e-5, rtol=1e-5,
                                       err_msg=f"jvp(exp) vs finite-diff {dev}")
        both_devices(body)

    def test_jvp_native_complex_nonholomorphic(self):
        # conj(z) is the canonical non-holomorphic example (df/dz = 0, df/dz* = 1).
        rng = np.random.RandomState(7)
        s = (4, 5)
        a = _np_complex(rng, s)
        vin = _np_complex(rng, s)

        def f(x):
            return x.conj().sum()          # full reduction -> real 0-d complex128 output

        ref = self._fd_jvp(lambda z: np.conj(z).sum(), a, vin)

        def body(dev):
            out, j = jvp(f, jt.array(a), jt.array(vin), create_graph=True)
            self.assertEqual(tuple(out.shape), (), f"out shape (0-d) {dev}")
            self.assertEqual(tuple(j.shape), (), f"jvp shape (0-d) {dev}")
            got = complex(j.item())
            self.assertTrue(np.isfinite(got), f"jvp finite {dev}")
            np.testing.assert_allclose(got, ref, atol=1e-5, rtol=1e-5,
                                       err_msg=f"jvp(conj) vs finite-diff {dev}")
        both_devices(body)

    def test_jvp_native_multi_input(self):
        rng = np.random.RandomState(11)
        s = (3, 4)
        x1 = _np_complex(rng, s)
        x2 = _np_complex(rng, s)
        v1 = _np_complex(rng, s)
        v2 = _np_complex(rng, s)

        def f(a, b):
            return (a * b).sum()

        ref = (self._fd_jvp(lambda a: (a * x2).sum(), x1, v1)
               + self._fd_jvp(lambda b: (x1 * b).sum(), x2, v2))

        def body(dev):
            out, j = jvp(f, (jt.array(x1), jt.array(x2)), (jt.array(v1), jt.array(v2)),
                        create_graph=True)
            got = complex(j.item())
            np.testing.assert_allclose(got, ref, atol=1e-5, rtol=1e-5,
                                       err_msg=f"multi-input jvp {dev}")
        both_devices(body)

    def test_jvp_native_zero_tangent_is_zero(self):
        rng = np.random.RandomState(17)
        s = (3, 3)
        a = _np_complex(rng, s)
        vzero = np.zeros(s, dtype="complex128")

        def f(x):
            return x.exp().sum()

        def body(dev):
            _, j = jvp(f, jt.array(a), jt.array(vzero), create_graph=True)
            got = complex(j.item())
            self.assertEqual(got, 0j, f"zero tangent -> zero jvp {dev}")
        both_devices(body)

    def test_jvp_native_unused_input_is_zero(self):
        rng = np.random.RandomState(19)
        s = (3,)
        x1 = _np_complex(rng, s)
        x2 = _np_complex(rng, s)
        v1 = _np_complex(rng, s)
        v2 = _np_complex(rng, s)

        def f(a, b):
            return (a * a).sum()           # does not depend on b

        def body(dev):
            out, j = jvp(f, (jt.array(x1), jt.array(x2)), (jt.array(v1), jt.array(v2)),
                        create_graph=True)
            ref = self._fd_jvp(lambda a: (a * a).sum(), x1, v1)
            got = complex(j.item())
            np.testing.assert_allclose(got, ref, atol=1e-5, rtol=1e-5,
                                       err_msg=f"unused-input jvp {dev}")
        both_devices(body)

    def test_jvp_native_create_graph_false_matches_true(self):
        rng = np.random.RandomState(13)
        s = (4, 5)
        a = _np_complex(rng, s)
        vin = _np_complex(rng, s)

        def f(x):
            return (x * x).sum(1)

        def body(dev):
            _, j_true = jvp(f, jt.array(a), jt.array(vin), create_graph=True)
            _, j_false = jvp(f, jt.array(a), jt.array(vin), create_graph=False)
            np.testing.assert_allclose(
                np.asarray(j_true.numpy()), np.asarray(j_false.numpy()),
                atol=1e-10, rtol=1e-10,
                err_msg=f"create_graph=True/False value mismatch {dev}")
        both_devices(body)

    # -------------------------------------------- second-order via nested jt.grad
    def test_hessian_of_real_complex_loss_via_nested_grad(self):
        # L(z) = |z|^2 = z*conj(z); z.grad = 2*dL/dconj(z) = 2*z (PyTorch convention).
        # Differentiating (g1.real+g1.imag).sum() a second time reproduces
        # 2*(1+1j) per element -- same closed form as the complex64 test, at
        # double precision, exercising the SAME double-backward machinery
        # jvp's double-backward trick relies on (jittor-core-gaps.md SS3.3).
        rng = np.random.RandomState(23)
        s = (3, 4)
        a = _np_complex(rng, s)

        def body(dev):
            x = jt.array(a)
            x.requires_grad = True
            with jt.enable_grad():
                loss = (x * x.conj()).real.sum()
                g1 = jt.grad(loss, [x], retain_graph=True)[0]
                proxy = (g1.real + g1.imag).sum()
                g2 = jt.grad(proxy, [x], retain_graph=True)[0]
            self.assertEqual(str(g1.dtype), "complex128", f"g1 dtype {dev}")
            self.assertEqual(str(g2.dtype), "complex128", f"g2 dtype {dev}")
            g1np = np.asarray(g1.numpy())
            g2np = np.asarray(g2.numpy())
            self.assertTrue(np.isfinite(g1np).all(), f"1st-order grad finite {dev}")
            self.assertTrue(np.isfinite(g2np).all(), f"2nd-order grad finite {dev}")
            np.testing.assert_allclose(g1np, 2 * a, atol=1e-10, rtol=1e-10,
                                       err_msg=f"d|z|^2/dz == 2*z {dev}")
            np.testing.assert_allclose(
                g2np, np.full(s, 2 + 2j, dtype="complex128"), atol=1e-10, rtol=1e-10,
                err_msg=f"2nd-order grad closed form {dev}")
        both_devices(body)

    def test_jvp_native_real_input_complex_output(self):
        # complex appears only in the OUTPUT: a real float64 Var times a
        # complex128 constant promotes to complex128.
        rng = np.random.RandomState(3)
        s = (5, 6)
        cone = jt.array(np.array(0.6 + 0.8j, dtype="complex128"))
        cone_np = complex(cone.item())

        def g(x):
            return (x * cone).sum(1)

        xr = rng.randn(*s).astype("float64")
        vr = rng.randn(*s).astype("float64")
        ref = self._fd_jvp(lambda z: (z * cone_np).sum(1), xr.astype("complex128"),
                           vr.astype("complex128"))

        def body(dev):
            out, j = jvp(g, jt.array(xr, dtype="float64"), jt.array(vr, dtype="float64"),
                        create_graph=True)
            self.assertEqual(str(out.dtype), "complex128", f"out dtype {dev}")
            self.assertEqual(tuple(j.shape), (5,), f"jvp shape {dev}")
            got = np.asarray(j.numpy())
            np.testing.assert_allclose(got, ref, atol=1e-5, rtol=1e-5,
                                       err_msg=f"real-input jvp vs finite-diff {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)

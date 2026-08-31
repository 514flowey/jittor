"""Native complex64 through gradfunctional (Phase 6 P5).

Locks that jittor.gradfunctional.{vjp, jvp} accept and correctly differentiate the NEW
native complex64 dtype (a first-class differentiable Var), not just the legacy
jt.nn.ComplexNumber real/imag-pair simulation.

What is checked (CPU + CUDA), per jittor-core-gaps.md §3.3:
  - vjp on a complex->complex function with a native complex64 input and a complex
    grad-output v: result matches a numpy finite-difference oracle of the SAME real
    seeded loss the implementation uses, L = Re(sum(out * conj(v))) over the (real,imag)
    representation. (This is torch's conjugate/Wirtinger input-grad convention.)
  - vjp result is numerically identical to the legacy ComplexNumber path (polymorphism).
  - jvp on native complex64 (holomorphic and non-holomorphic, elementwise and full
    reduction to a 0-d output, real-input/complex-output, multi-input pytree,
    create_graph True/False, zero tangent, unused input) matches a numpy
    finite-difference oracle of the real-linear directional derivative
    (f(z+eps*v)-f(z-eps*v))/(2*eps) -- well-defined for non-holomorphic f too.
  - A real-valued loss of a complex variable differentiated TWICE (native complex
    double-backward, the same machinery jvp's double-backward trick uses) is finite and
    matches the closed form.
  - jvp on the legacy ComplexNumber path STILL works (regression guard).
  - jvp on native REAL Vars still works.

No real torch in this env: numpy is the oracle.

History (native complex64 jvp used to raise NotImplementedError here, believed to need
an unimplemented complex64->float32 cast-backward): the double-backward machinery itself
was already correct. Two unrelated real bugs made it LOOK broken, both now fixed at the
source (not worked around in gradfunctional):
  1. nn.py::_real2_to_complex64_raw used a stale ``list(shape[:-1]) or [1]`` fallback
     (pre-dating §3.1's real 0-d Var support) that silently turned a genuinely 0-d
     complex64 gradient into shape (1,) -- this broke ANY backward (not just jvp) through
     .real/.imag of a full-reduction complex64 result.
  2. py_converter.h's ItemData->PyObject conversion had no complex64 case, so ``.item()``
     on a complex64 scalar silently reinterpreted the raw bytes as an int64.

Run:  python -m jittor.test.test_complex64_gradfunctional
"""
import unittest
import numpy as np
import jittor as jt
from jittor.gradfunctional import vjp, jvp
from jittor.nn import ComplexNumber

_DEVICES = [("cpu", 0)] + ([("cuda", 1)] if jt.has_cuda else [])


def both_devices(fn):
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


def _to_complex(g):
    # numpy complex array from a native complex64 Var.
    return np.asarray(g.numpy())


def _cn_to_complex(cn):
    # numpy complex array from a legacy ComplexNumber.
    v = np.asarray(cn.value.numpy())
    return v[..., 0] + 1j * v[..., 1]


def _np_complex(rng, shape):
    return (rng.randn(*shape) + 1j * rng.randn(*shape)).astype("complex64")


def _fd_vjp(fnp, a, v, eps=1e-3):
    """Finite-difference of the real seeded loss L(z) = sum(Re f(z)*Re v + Im f(z)*Im v),
    w.r.t. real and imag parts of a, returned as a complex grad (the vjp value)."""
    a = a.astype("complex64")
    zr0 = a.real.astype(np.float64)
    zi0 = a.imag.astype(np.float64)

    def loss(zr, zi):
        fo = fnp((zr + 1j * zi).astype("complex64"))
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


class TestComplex64GradFunctional(unittest.TestCase):
    # -------------------------------------------------------------- vjp: exp().sum(1)
    def test_vjp_exp_complex_input(self):
        rng = np.random.RandomState(0)
        s = (5, 6)
        a = _np_complex(rng, s)
        v = _np_complex(rng, (5,))          # matches the (5,) output of exp().sum(1)

        def f(x):
            return x.exp().sum(1)

        def fnp(z):
            return np.exp(z).sum(1)

        fd = _fd_vjp(fnp, a, v)

        def body(dev):
            out, g = vjp(f, jt.array(a), jt.array(v), create_graph=True)
            self.assertEqual(str(out.dtype), "complex64", f"out dtype {dev}")
            self.assertEqual(str(g.dtype), "complex64", f"vjp dtype {dev}")
            self.assertEqual(tuple(g.shape), s, f"vjp shape {dev}")
            gnp = _to_complex(g)
            self.assertTrue(np.isfinite(gnp).all(), f"vjp finite {dev}")
            np.testing.assert_allclose(gnp, fd, atol=2e-3, rtol=2e-3,
                                       err_msg=f"vjp vs finite-diff {dev}")
        both_devices(body)

    # -------------------------------------------------------- vjp: closed form (z*z)
    def test_vjp_square_closed_form(self):
        # f(z) = z*z (elementwise, holomorphic). For the real seeded loss
        # L = Re(sum(f * conj(v))) the input grad is conj(df/dz)^T applied to v:
        # since df/dz = 2z and the map is diagonal, grad = conj(2z) * v ... but the
        # implementation differentiates the REAL loss, whose grad equals 2*conj(z)*v
        # (Wirtinger conj convention). We assert against that closed form AND the
        # finite-diff oracle agrees.
        rng = np.random.RandomState(4)
        s = (4, 3)
        a = _np_complex(rng, s)
        v = _np_complex(rng, s)
        closed = np.conj(2 * a) * v        # == 2*conj(a)*v

        def f(x):
            return x * x

        def fnp(z):
            return z * z

        fd = _fd_vjp(fnp, a, v)
        np.testing.assert_allclose(fd, closed, atol=2e-3, rtol=2e-3,
                                   err_msg="finite-diff vs closed form (sanity)")

        def body(dev):
            out, g = vjp(f, jt.array(a), jt.array(v), create_graph=True)
            gnp = _to_complex(g)
            np.testing.assert_allclose(gnp, closed, atol=2e-3, rtol=2e-3,
                                       err_msg=f"vjp(z*z) vs closed form {dev}")
        both_devices(body)

    # ------------------------------------- vjp: native complex64 == legacy ComplexNumber
    def test_vjp_native_matches_complexnumber(self):
        rng = np.random.RandomState(1)
        s = (5, 6)
        a = _np_complex(rng, s)
        v = _np_complex(rng, (5,))

        def f(x):
            return x.exp().sum(1)

        def body(dev):
            # native complex64
            _, g_native = vjp(f, jt.array(a), jt.array(v), create_graph=True)
            gn = _to_complex(g_native)
            # legacy ComplexNumber (real/imag stacked value)
            cn_a = ComplexNumber(jt.array(np.stack([a.real, a.imag], -1)),
                                 is_concat_value=True)
            cn_v = ComplexNumber(jt.array(np.stack([v.real, v.imag], -1)),
                                 is_concat_value=True)
            _, g_cn = vjp(f, cn_a, cn_v, create_graph=True)
            gc = _cn_to_complex(g_cn)
            np.testing.assert_allclose(gn, gc, atol=1e-4, rtol=1e-4,
                                       err_msg=f"native vjp != ComplexNumber vjp {dev}")
        both_devices(body)

    # ------------------------------------------ vjp: tuple of two native complex inputs
    def test_vjp_two_complex_inputs(self):
        # adder(x, y) = w1*x + w2*y, both native complex64 -> complex output, two grads.
        rng = np.random.RandomState(2)
        s = (4, 5)
        x = _np_complex(rng, s)
        y = _np_complex(rng, s)
        v = _np_complex(rng, s)
        w1 = np.array(0.7 + 0.4j, dtype="complex64")
        w2 = np.array(-0.3 + 0.9j, dtype="complex64")

        def f(a, b):
            return jt.array(w1) * a + jt.array(w2) * b

        def body(dev):
            out, (gx, gy) = vjp(f, (jt.array(x), jt.array(y)), jt.array(v),
                                create_graph=True)
            self.assertEqual(str(out.dtype), "complex64", f"out dtype {dev}")
            self.assertEqual(str(gx.dtype), "complex64", f"gx dtype {dev}")
            self.assertEqual(str(gy.dtype), "complex64", f"gy dtype {dev}")
            gxnp, gynp = _to_complex(gx), _to_complex(gy)
            # d/da of Re(sum((w1 a + w2 b) conj(v))) = conj(w1) * v  (Wirtinger conj conv.)
            np.testing.assert_allclose(gxnp, np.conj(w1) * v, atol=2e-3, rtol=2e-3,
                                       err_msg=f"two-complex grad x {dev}")
            np.testing.assert_allclose(gynp, np.conj(w2) * v, atol=2e-3, rtol=2e-3,
                                       err_msg=f"two-complex grad y {dev}")
        both_devices(body)

    # --------------------------------------------- jvp: native complex64 -> numeric pass
    # jittor-core-gaps.md §3.3. Previously native complex64 jvp raised NotImplementedError
    # (the double-backward trick was believed to need an unimplemented complex64->float32
    # cast-backward). It turns out the double-backward machinery itself was already
    # correct; two unrelated real bugs made it LOOK broken and are now fixed at the
    # source (not worked around here):
    #   1. nn.py::_real2_to_complex64_raw used a stale ``list(x.shape[:-1]) or [1]``
    #      fallback (pre-dating jittor-core-gaps.md §3.1's real 0-d Var support), which
    #      silently turned a genuinely 0-d complex64 gradient into shape (1,) -- this
    #      broke ANY backward (not just jvp's double-backward) through .real/.imag of a
    #      full-reduction (e.g. ``x.sum()``) complex64 result.
    #   2. py_converter.h's ItemData->PyObject conversion had no complex64 case, so
    #      ``.item()`` on a complex64 scalar silently reinterpreted the raw 8-byte
    #      (float32,float32) pair as an int64 bit pattern (returning a huge garbage int
    #      instead of a python complex).
    # A finite-difference oracle (numpy, real-linear directional derivative
    # ``(f(z+eps*v) - f(z-eps*v)) / (2*eps)``) is used throughout: this is well-defined
    # for holomorphic AND non-holomorphic f alike, unlike a closed-form Wirtinger formula.
    def _fd_jvp(self, fnp, z, v, eps=1e-3):
        return (fnp(z + eps * v) - fnp(z - eps * v)) / (2 * eps)

    def test_jvp_native_complex_holomorphic(self):
        # f(x) = exp(x).sum(1): elementwise holomorphic, reduced over one axis.
        rng = np.random.RandomState(3)
        s = (5, 6)
        a = _np_complex(rng, s)
        vin = _np_complex(rng, s)

        def f(x):
            return x.exp().sum(1)

        ref = self._fd_jvp(lambda z: np.exp(z).sum(1), a, vin)

        def body(dev):
            out, j = jvp(f, jt.array(a), jt.array(vin), create_graph=True)
            self.assertEqual(str(out.dtype), "complex64", f"out dtype {dev}")
            self.assertEqual(str(j.dtype), "complex64", f"jvp dtype {dev}")
            self.assertEqual(tuple(j.shape), (5,), f"jvp shape {dev}")
            got = np.asarray(j.numpy())
            self.assertTrue(np.isfinite(got).all(), f"jvp finite {dev}")
            np.testing.assert_allclose(got, ref, atol=5e-2, rtol=5e-2,
                                       err_msg=f"jvp(exp) vs finite-diff {dev}")
        both_devices(body)

    def test_jvp_native_complex_nonholomorphic(self):
        # conj(z) is the canonical non-holomorphic example (df/dz = 0, df/dz* = 1).
        rng = np.random.RandomState(7)
        s = (4, 5)
        a = _np_complex(rng, s)
        vin = _np_complex(rng, s)

        def f(x):
            return x.conj().sum()          # full reduction -> real 0-d complex64 output

        ref = self._fd_jvp(lambda z: np.conj(z).sum(), a, vin)

        def body(dev):
            out, j = jvp(f, jt.array(a), jt.array(vin), create_graph=True)
            self.assertEqual(tuple(out.shape), (), f"out shape (0-d) {dev}")
            self.assertEqual(tuple(j.shape), (), f"jvp shape (0-d) {dev}")
            got = complex(j.item())
            self.assertTrue(np.isfinite(got), f"jvp finite {dev}")
            np.testing.assert_allclose(got, ref, atol=5e-2, rtol=5e-2,
                                       err_msg=f"jvp(conj) vs finite-diff {dev}")
        both_devices(body)

    def test_jvp_native_real_input_complex_output(self):
        # complex appears only in the OUTPUT: a real Var times a complex64 constant
        # promotes to complex64 (this is the case the old guard special-cased and
        # raised on; it now numerically passes like everything else).
        rng = np.random.RandomState(3)
        s = (5, 6)
        cone = jt.array(np.array(0.6 + 0.8j, dtype="complex64"))
        cone_np = complex(cone.item())

        def g(x):
            return (x * cone).sum(1)

        xr = rng.randn(*s).astype("float32")
        vr = rng.randn(*s).astype("float32")
        ref = self._fd_jvp(lambda z: (z * cone_np).sum(1), xr.astype("complex64"),
                           vr.astype("complex64"))

        def body(dev):
            out, j = jvp(g, jt.array(xr), jt.array(vr), create_graph=True)
            self.assertEqual(str(out.dtype), "complex64", f"out dtype {dev}")
            self.assertEqual(tuple(j.shape), (5,), f"jvp shape {dev}")
            got = np.asarray(j.numpy())
            np.testing.assert_allclose(got, ref, atol=5e-2, rtol=5e-2,
                                       err_msg=f"real-input jvp vs finite-diff {dev}")
        both_devices(body)

    def test_jvp_native_multi_input(self):
        # pytree (tuple) inputs: f(a,b) = (a*b).sum(); jvp = sum(da*b + a*db) at (v1,v2).
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
            # jt.gradfunctional.jvp sums contributions across a tuple of inputs into a
            # SINGLE jvp result per output (matches torch.autograd.functional.jvp).
            out, j = jvp(f, (jt.array(x1), jt.array(x2)), (jt.array(v1), jt.array(v2)),
                        create_graph=True)
            got = complex(j.item())
            np.testing.assert_allclose(got, ref, atol=5e-2, rtol=5e-2,
                                       err_msg=f"multi-input jvp vs finite-diff {dev}")
        both_devices(body)

    def test_jvp_native_multi_output(self):
        # jittor-core-gaps.md §3.3 criterion 2 asks for multi-output jvp
        # coverage; previously only multi-INPUT was tested. f returns two
        # outputs of different shape -- each must get its own jvp entry.
        rng = np.random.RandomState(17)
        s = (3, 4)
        a = _np_complex(rng, s)
        vin = _np_complex(rng, s)

        def f(x):
            return x.exp().sum(1), (x * x).sum()

        ref0 = self._fd_jvp(lambda z: np.exp(z).sum(1), a, vin)
        ref1 = self._fd_jvp(lambda z: (z * z).sum(), a, vin)

        def body(dev):
            out, j = jvp(f, jt.array(a), jt.array(vin), create_graph=True)
            self.assertIsInstance(j, tuple)
            self.assertEqual(len(j), 2)
            np.testing.assert_allclose(np.asarray(j[0].numpy()), ref0, atol=5e-2, rtol=5e-2,
                                       err_msg=f"multi-output jvp[0] {dev}")
            np.testing.assert_allclose(complex(j[1].item()), ref1, atol=5e-2, rtol=5e-2,
                                       err_msg=f"multi-output jvp[1] {dev}")
        both_devices(body)

    def test_jvp_native_create_graph_false_matches_true(self):
        rng = np.random.RandomState(13)
        s = (4, 5)
        a = _np_complex(rng, s)
        vin = _np_complex(rng, s)

        def f(x):
            return (x * x).sum(1)          # holomorphic, non-trivial (not just linear)

        def body(dev):
            _, j_true = jvp(f, jt.array(a), jt.array(vin), create_graph=True)
            _, j_false = jvp(f, jt.array(a), jt.array(vin), create_graph=False)
            np.testing.assert_allclose(
                np.asarray(j_true.numpy()), np.asarray(j_false.numpy()),
                atol=1e-5, rtol=1e-5,
                err_msg=f"create_graph=True/False value mismatch {dev}")
        both_devices(body)

    def test_jvp_native_zero_tangent_is_zero(self):
        rng = np.random.RandomState(17)
        s = (3, 3)
        a = _np_complex(rng, s)
        vzero = np.zeros(s, dtype="complex64")

        def f(x):
            return x.exp().sum()

        def body(dev):
            _, j = jvp(f, jt.array(a), jt.array(vzero), create_graph=True)
            got = complex(j.item())
            self.assertEqual(got, 0j, f"zero tangent -> zero jvp {dev}")
        both_devices(body)

    def test_jvp_native_unused_input_is_zero(self):
        # strict=False (default): an input the output doesn't depend on gets a zero jvp
        # instead of raising.
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
            np.testing.assert_allclose(got, ref, atol=5e-2, rtol=5e-2,
                                       err_msg=f"unused-input jvp {dev}")
        both_devices(body)

    def test_hessian_of_real_complex_loss_via_nested_grad(self):
        # A real-valued loss of a complex variable, L(z) = |z|^2 = z * conj(z). jittor's
        # native jt.grad follows the same convention as PyTorch: for a real L,
        # z.grad = 2 * dL/dconj(z); holding z fixed, dL/dconj(z) = z, so grad = 2*z.
        # Verify the SECOND jt.grad call (double backward through the native complex
        # graph, the same machinery jvp's double-backward trick relies on) is finite
        # and matches that closed form applied twice: differentiating
        # (g1.real + g1.imag).sum() = 2*sum(Re z + Im z) a second time w.r.t. z
        # reproduces 2*(1+1j) per element (the same "grad = 2 * d(.)/dconj(z)" rule).
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
            g1np = np.asarray(g1.numpy())
            g2np = np.asarray(g2.numpy())
            self.assertTrue(np.isfinite(g1np).all(), f"1st-order grad finite {dev}")
            self.assertTrue(np.isfinite(g2np).all(), f"2nd-order grad finite {dev}")
            np.testing.assert_allclose(g1np, 2 * a, atol=2e-3, rtol=2e-3,
                                       err_msg=f"d|z|^2/dz == 2*z {dev}")
            np.testing.assert_allclose(
                g2np, np.full(s, 2 + 2j, dtype="complex64"), atol=2e-3, rtol=2e-3,
                err_msg=f"2nd-order grad closed form {dev}")
        both_devices(body)

    # ----------------------------------------------- jvp: native REAL still works
    def test_jvp_native_real_ok(self):
        rng = np.random.RandomState(5)
        s = (5, 6)
        x = rng.randn(*s).astype("float32")
        vin = rng.randn(*s).astype("float32")

        def f(a):
            return (a * a).sum(1)       # out (5,)

        # jvp = J @ vin ; for f=sum_j a_ij^2, (J v)_i = sum_j 2 a_ij v_ij
        ref = (2 * x * vin).sum(1)

        def body(dev):
            out, j = jvp(f, jt.array(x), jt.array(vin), create_graph=True)
            jnp = np.asarray(j.numpy())
            self.assertTrue(np.isfinite(jnp).all(), f"real jvp finite {dev}")
            np.testing.assert_allclose(jnp, ref, atol=2e-3, rtol=2e-3,
                                       err_msg=f"real jvp vs closed form {dev}")
        both_devices(body)

    # --------------------------------------------- jvp: legacy ComplexNumber still works
    def test_jvp_complexnumber_still_works(self):
        rng = np.random.RandomState(6)
        s = (5, 6)
        a = _np_complex(rng, s)
        vin = _np_complex(rng, s)

        def f(x):
            return x.exp().sum(1)

        def body(dev):
            cn_a = ComplexNumber(jt.array(np.stack([a.real, a.imag], -1)),
                                 is_concat_value=True)
            cn_v = ComplexNumber(jt.array(np.stack([vin.real, vin.imag], -1)),
                                 is_concat_value=True)
            out, j = jvp(f, cn_a, cn_v, create_graph=True)
            self.assertIsInstance(j, ComplexNumber, f"jvp returns ComplexNumber {dev}")
            jnp = _cn_to_complex(j)
            self.assertTrue(np.isfinite(jnp).all(), f"CN jvp finite {dev}")
            self.assertEqual(jnp.shape, (5,), f"CN jvp shape {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)

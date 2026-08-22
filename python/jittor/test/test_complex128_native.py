"""Native complex128 dtype (jittor-core-gaps.md §3.2) — create / numpy round-trip /
arithmetic / dtype promotion vs numpy, mirroring test_complex64_native.py at double
precision. Locks: registered (dsize 16, is_complex), lossless numpy complex128 import/
export (not silently downcast to complex64), elementwise arithmetic, transcendentals,
reduce, matmul, backward, and the complex64<->complex128/float64 promotion & cast rules.
CPU+CUDA.

Run:  python -m jittor.test.test_complex128_native
"""
import unittest
import numpy as np
import jittor as jt

_DEVICES = [("cpu", 0)] + ([("cuda", 1)] if jt.has_cuda else [])


def both_devices(fn):
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


class TestComplex128Native(unittest.TestCase):
    def test_dtype_props(self):
        ns = jt.NanoString("complex128")
        self.assertEqual(ns.dsize(), 16)
        self.assertTrue(ns.is_complex())
        self.assertFalse(ns.is_floating_point())
        self.assertFalse(ns.is_int())

    def test_create_roundtrip_bit_precision(self):
        # gaps.md §3.2 acceptance #1: numpy complex128 import/export must NOT be
        # silently downcast to complex64 -- assert bit-exact equality (not just
        # allclose), which a float32 round-trip would fail.
        a = np.array([1 + 2j, 3 - 4j, 0 + 1j, -2 - 3j,
                      1.0 + 1.1102230246251565e-16j],  # needs full float64 mantissa
                     dtype="complex128")
        def body(dev):
            v = jt.array(a)
            self.assertEqual(str(v.dtype), "complex128", f"dtype {dev}")
            got = np.asarray(v.numpy())
            self.assertEqual(got.dtype, np.complex128, f"numpy() dtype {dev}")
            np.testing.assert_array_equal(got, a, err_msg=f"bit-exact roundtrip {dev}")
        both_devices(body)

    def test_zeros_ones(self):
        def body(dev):
            z = jt.zeros((3,), "complex128")
            self.assertEqual(str(z.dtype), "complex128", f"zeros dtype {dev}")
            np.testing.assert_array_equal(np.asarray(z.numpy()), np.zeros(3, "complex128"))
            o = jt.ones((3,), "complex128")
            np.testing.assert_array_equal(np.asarray(o.numpy()), np.ones(3, "complex128"))
        both_devices(body)

    def test_arithmetic(self):
        rng = np.random.RandomState(0)
        a = (rng.randn(8) + 1j * rng.randn(8)).astype("complex128")
        b = (rng.randn(8) + 1j * rng.randn(8)).astype("complex128")
        def body(dev):
            va, vb = jt.array(a), jt.array(b)
            for nm, jr, nr in [("add", va + vb, a + b), ("sub", va - vb, a - b),
                               ("mul", va * vb, a * b), ("div", va / vb, a / b),
                               ("neg", -va, -a)]:
                self.assertEqual(str(jr.dtype), "complex128", f"{nm} dtype {dev}")
                np.testing.assert_allclose(np.asarray(jr.numpy()), nr, atol=1e-12, rtol=1e-12,
                                           err_msg=f"{nm} {dev}")
        both_devices(body)

    def test_transcendentals(self):
        rng = np.random.RandomState(11)
        a = (rng.randn(8) + 1j * rng.randn(8)).astype("complex128")
        ops = [("exp", np.exp), ("log", np.log), ("sin", np.sin),
               ("cos", np.cos), ("sqrt", np.sqrt)]
        def body(dev):
            for nm, npf in ops:
                r = np.asarray(getattr(jt, nm)(jt.array(a)).numpy())
                self.assertEqual(r.dtype.name, "complex128", f"{nm} dtype {dev}")
                np.testing.assert_allclose(r, npf(a), atol=1e-10, rtol=1e-10,
                                           err_msg=f"{nm} {dev}")
        both_devices(body)

    def test_conj_abs(self):
        rng = np.random.RandomState(3)
        a = (rng.randn(6) + 1j * rng.randn(6)).astype("complex128")
        def body(dev):
            cj = jt.array(a).conj()
            self.assertEqual(str(cj.dtype), "complex128", f"conj dtype {dev}")
            np.testing.assert_allclose(np.asarray(cj.numpy()), a.conj(), atol=1e-12)
            ab = jt.array(a).abs()
            self.assertEqual(str(ab.dtype), "float64", f"abs dtype {dev}")
            np.testing.assert_allclose(np.asarray(ab.numpy()), np.abs(a), atol=1e-12)
        both_devices(body)

    def test_matmul_and_reduce(self):
        rng = np.random.RandomState(1)
        A = (rng.randn(3, 4) + 1j * rng.randn(3, 4)).astype("complex128")
        B = (rng.randn(4, 5) + 1j * rng.randn(4, 5)).astype("complex128")
        ref = A @ B
        def body(dev):
            r = np.asarray(jt.matmul(jt.array(A), jt.array(B)).numpy())
            self.assertEqual(r.dtype.name, "complex128", f"matmul dtype {dev}")
            np.testing.assert_allclose(r, ref, atol=1e-10, rtol=1e-10, err_msg=f"matmul {dev}")
            a = (rng.randn(6) + 1j * rng.randn(6)).astype("complex128")
            s = jt.array(a).sum()
            self.assertEqual(tuple(s.shape), (), f"sum shape (0-d) {dev}")
            np.testing.assert_allclose(complex(s.item()), a.sum(), atol=1e-10)
        both_devices(body)

    def test_grad(self):
        rng = np.random.RandomState(11)
        a = (rng.randn(4) + 1j * rng.randn(4)).astype("complex128")
        def body(dev):
            z = jt.array(a)
            z.requires_grad = True
            with jt.enable_grad():
                loss = (z * z.conj()).real.sum()
                g = jt.grad(loss, [z])[0]
            self.assertEqual(str(g.dtype), "complex128", f"grad dtype {dev}")
            np.testing.assert_allclose(np.asarray(g.numpy()), 2 * a, atol=1e-10,
                                       err_msg=f"d|z|^2/dz==2z {dev}")
        both_devices(body)

    def test_view_bridge(self):
        # complex128 <-> float64[...,2], mirroring the complex64<->float32 bridge.
        rng = np.random.RandomState(5)
        a = (rng.randn(3, 4) + 1j * rng.randn(3, 4)).astype("complex128")
        def body(dev):
            z = jt.array(a)
            vr = jt.nn.view_as_real(z)
            self.assertEqual(str(vr.dtype), "float64", f"view_as_real dtype {dev}")
            self.assertEqual(tuple(vr.shape), (3, 4, 2), f"view_as_real shape {dev}")
            np.testing.assert_allclose(np.asarray(vr.numpy()),
                                       np.stack([a.real, a.imag], axis=-1), atol=1e-12)
            zc = jt.nn.view_as_complex(vr)
            self.assertEqual(str(zc.dtype), "complex128", f"view_as_complex dtype {dev}")
            np.testing.assert_array_equal(np.asarray(zc.numpy()), a,
                                          err_msg=f"bridge roundtrip {dev}")
        both_devices(body)

    def test_accessors(self):
        rng = np.random.RandomState(6)
        re = rng.randn(3, 4).astype("float64")
        im = rng.randn(3, 4).astype("float64")
        a = (re + 1j * im).astype("complex128")
        def body(dev):
            z = jt.array(a)
            self.assertEqual(str(z.real.dtype), "float64", f"real dtype {dev}")
            self.assertEqual(str(z.imag.dtype), "float64", f"imag dtype {dev}")
            np.testing.assert_allclose(np.asarray(z.real.numpy()), re, atol=1e-12)
            np.testing.assert_allclose(np.asarray(z.imag.numpy()), im, atol=1e-12)
            np.testing.assert_allclose(np.asarray(z.angle().numpy()), np.angle(a), atol=1e-9)
        both_devices(body)

    # ------------------------------------------------------------ dtype promotion
    def test_promotion_complex64_complex128(self):
        def body(dev):
            c64 = jt.array(np.array(1 + 2j, dtype="complex64"))
            c128 = jt.array(np.array(3 + 4j, dtype="complex128"))
            self.assertEqual(str((c64 + c128).dtype), "complex128", f"c64+c128 {dev}")
            self.assertEqual(str((c128 + c64).dtype), "complex128", f"c128+c64 {dev}")
        both_devices(body)

    def test_promotion_complex_with_real(self):
        # complex64's value type is float32; combining with a double-precision-class
        # real operand (float64, or an int64-class integer) widens to complex128 to
        # avoid losing precision -- matches numpy/torch. complex128 always stays
        # complex128 (it's already the widest).
        def body(dev):
            # jt.array narrows any bare 8-byte-dtype numpy array to 32-bit by default
            # (float64->float32, int64->int32) unless complex or given an explicit
            # dtype= override -- pass dtype= explicitly to actually get float64 here.
            c64 = jt.array(np.array(1 + 2j, dtype="complex64"))
            f32 = jt.array(np.array(1.0), dtype="float32")
            f64 = jt.array(np.array(1.0), dtype="float64")
            i32 = jt.array(np.array(1, dtype="int32"))
            self.assertEqual(str((c64 + f32).dtype), "complex64", f"c64+f32 {dev}")
            self.assertEqual(str((c64 + f64).dtype), "complex128", f"c64+f64 {dev}")
            self.assertEqual(str((c64 + i32).dtype), "complex64", f"c64+i32 {dev}")
            c128 = jt.array(np.array(1 + 2j, dtype="complex128"))
            self.assertEqual(str((c128 + f32).dtype), "complex128", f"c128+f32 {dev}")
            self.assertEqual(str((c128 + f64).dtype), "complex128", f"c128+f64 {dev}")
        both_devices(body)

    def test_cast_precision_roundtrip(self):
        rng = np.random.RandomState(2)
        a = (rng.randn(5) + 1j * rng.randn(5)).astype("complex128")
        def body(dev):
            x128 = jt.array(a)
            x64 = x128.cast("complex64")
            self.assertEqual(str(x64.dtype), "complex64", f"cast down dtype {dev}")
            np.testing.assert_allclose(np.asarray(x64.numpy()), a.astype("complex64"),
                                       atol=1e-6, err_msg=f"cast c128->c64 {dev}")
            back = x64.cast("complex128")
            self.assertEqual(str(back.dtype), "complex128", f"cast up dtype {dev}")
            # widening a narrowed value must round-trip EXACTLY (no further precision loss)
            np.testing.assert_array_equal(np.asarray(back.numpy()), a.astype("complex64").astype("complex128"),
                                          err_msg=f"cast c64->c128 exact {dev}")
            # complex -> real -> complex drops imag (documented behavior, same as complex64)
            r = x128.cast("float64")
            self.assertEqual(str(r.dtype), "float64", f"complex->real cast dtype {dev}")
            np.testing.assert_allclose(np.asarray(r.numpy()), a.real, atol=1e-12)
        both_devices(body)

    def test_cast_backward(self):
        def body(dev):
            seed = jt.array(np.array([2.0, -0.5], dtype="float64"))
            z = jt.array(np.array([1 + 2j, -3 + 4j], dtype="complex128"))
            g = jt.grad((z.cast("float64") * seed).sum(), z)
            g = g[0] if isinstance(g, (list, tuple)) else g
            np.testing.assert_array_equal(np.asarray(g.numpy()),
                                          np.array([2 + 0j, -0.5 + 0j], dtype="complex128"),
                                          err_msg=f"complex->real cast backward {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)

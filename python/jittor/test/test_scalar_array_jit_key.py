"""Shared scalar constants must retain dtype-specific compiled kernels."""

import unittest

import jittor as jt
import numpy as np


class TestScalarArrayJitKey(unittest.TestCase):
    def test_multi_target_gradient_seed_dtypes(self):
        for _ in range(2):
            for dtype in ("float32", "float64", "complex64", "complex128"):
                with self.subTest(dtype=dtype):
                    data = np.array([0.4, -0.7], dtype=dtype)
                    if "complex" in dtype:
                        data += np.array([0.2j, -0.3j], dtype=dtype)
                    x = jt.array(data, dtype=dtype)
                    a, b = x.clone(), x.clone()

                    def norm(value):
                        product = value * jt.conj(value)
                        if "complex" in dtype:
                            product = jt.real(product)
                        return product.sum()

                    loss = norm(a) + 3 * norm(b)
                    first, second = jt.grad(loss, [a, b])
                    np.testing.assert_allclose(
                        first.numpy(), 2 * data, rtol=2e-6, atol=2e-6
                    )
                    np.testing.assert_allclose(
                        second.numpy(), 6 * data, rtol=2e-6, atol=2e-6
                    )


@unittest.skipUnless(jt.compiler.has_cuda, "CUDA is unavailable")
class TestScalarArrayJitKeyCUDA(TestScalarArrayJitKey):
    def setUp(self):
        self.scope = jt.flag_scope(use_cuda=1)
        self.scope.__enter__()

    def tearDown(self):
        self.scope.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()

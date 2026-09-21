"""Native complex64 spectral adjoints against independent NumPy differences.

The real and imaginary input coordinates are perturbed independently. Vector
losses use weighted squared magnitudes, so arbitrary eigenvector/SVD phases do
not turn a valid finite-difference check into a comparison of different gauges.
"""

import json

import numpy as np
import pytest

import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks
from _helpers import capability
from _helpers.cupy_bridge import cuda_numpy_code_available


@pytest.fixture(params=("cpu", "cuda"))
def spectral_device(request):
    device = request.param
    if device == "cuda":
        if not capability.check_accelerator("cuda", backend=jt).enabled:
            pytest.skip("CUDA is unavailable")
        if not cuda_numpy_code_available():
            pytest.skip("CUDA numpy-code requires CuPy")
    with jt.runtime.scope(use_cuda=int(device == "cuda")):
        with forbid_backend_fallbacks():
            yield device
            jt.sync_all()


def _weights(shape):
    return (np.arange(np.prod(shape)).reshape(shape) / np.prod(shape) + 0.3).astype(
        "float32"
    )


def _input(shape):
    rng = np.random.RandomState(73)
    value = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    # Pin the oracle to exactly the native complex64 input, then difference in
    # complex128 so rounding of the forward does not dominate the derivative.
    return value.astype("complex64").astype("complex128")


def _numerical_gradient(loss, value, eps=1e-5):
    gradient = np.zeros_like(value)
    for index in np.ndindex(value.shape):
        for direction in (1.0, 1.0j):
            plus, minus = value.copy(), value.copy()
            plus[index] += eps * direction
            minus[index] -= eps * direction
            gradient[index] += direction * (loss(plus) - loss(minus)) / (2 * eps)
    return gradient


def _squared_loss(value, native):
    weights = _weights(tuple(value.shape))
    if native:
        return ((value * value.conj()).real * jt.array(weights)).sum()
    return np.sum(np.abs(value) ** 2 * weights)


def _spectral_loss(value, native):
    weights = _weights(tuple(value.shape))
    return (value.real * (jt.array(weights) if native else weights)).sum()


def _svd_loss(outputs, component, native):
    u, s, vh = outputs
    if component == "s":
        return _spectral_loss(s, native)
    if component == "u":
        return _squared_loss(u, native)
    if component == "vh":
        return _squared_loss(vh, native)
    return (
        _squared_loss(u, native) + _spectral_loss(s, native) + _squared_loss(vh, native)
    )


def _check_gradient(value, native_loss, numpy_loss, device):
    x = jt.array(value.astype("complex64"), dtype="complex64")
    x.start_grad()
    loss = native_loss(x)
    gradient = jt.grad(loss, x)
    gradient.sync()
    # Check the physical device before numpy() fetches the result to the host.
    assert gradient.location() == ("device" if device == "cuda" else "cpu")
    actual = gradient.numpy().copy()
    reference = _numerical_gradient(numpy_loss, value)
    np.testing.assert_allclose(actual, reference, atol=5e-4, rtol=2e-3)
    np.testing.assert_allclose(loss.numpy(), numpy_loss(value), atol=2e-4, rtol=2e-4)
    return actual


@pytest.mark.parametrize(
    "shape", [(3, 3), (4, 2), (2, 4), (2, 3, 3), (2, 4, 2), (2, 2, 4)]
)
@pytest.mark.parametrize("component", ["u", "s", "vh", "all"])
def test_svd_spectral_and_phase_invariant_vector_gradients(
    spectral_device, shape, component
):
    value = _input(shape)
    singular_values = np.linalg.svd(value, compute_uv=False)
    assert np.all(singular_values > 0.1)
    assert np.all(np.abs(np.diff(singular_values, axis=-1)) > 0.1)
    _check_gradient(
        value,
        lambda x: _svd_loss(jt.linalg.svd(x), component, True),
        lambda x: _svd_loss(np.linalg.svd(x, full_matrices=False), component, False),
        spectral_device,
    )


def _eigh_loss(outputs, component, native):
    w, v = outputs
    if component == "w":
        return _spectral_loss(w, native)
    if component == "v":
        return _squared_loss(v, native)
    return _spectral_loss(w, native) + _squared_loss(v, native)


@pytest.mark.parametrize("shape", [(3, 3), (2, 3, 3)])
@pytest.mark.parametrize("component", ["w", "v", "all"])
def test_eigh_spectral_and_phase_invariant_vector_gradients(
    spectral_device, shape, component
):
    # Deliberately not Hermitian: UPLO='L' ignores the upper triangle and the
    # imaginary diagonal. This checks the lower-triangle folding, not merely
    # the projected Hermitian gradient expected by a downstream adapter.
    value = _input(shape)
    eigenvalues = np.linalg.eigh(value, UPLO="L")[0]
    assert np.all(np.abs(np.diff(eigenvalues, axis=-1)) > 0.1)
    gradient = _check_gradient(
        value,
        lambda x: _eigh_loss(jt.linalg.eigh(x), component, True),
        lambda x: _eigh_loss(np.linalg.eigh(x, UPLO="L"), component, False),
        spectral_device,
    )
    np.testing.assert_array_equal(np.triu(gradient, 1), np.zeros_like(gradient))
    diagonal = np.diagonal(gradient, axis1=-2, axis2=-1).imag
    np.testing.assert_array_equal(diagonal, np.zeros_like(diagonal))


@pytest.mark.parametrize("decomposition", ["svd", "eigh"])
def test_spectral_imaginary_output_has_zero_gradient(spectral_device, decomposition):
    value = _input((2, 3, 3))

    def loss(x):
        spectrum = (
            jt.linalg.svd(x)[1] if decomposition == "svd" else jt.linalg.eigh(x)[0]
        )
        return spectrum.imag.sum()

    gradient = _check_gradient(value, loss, lambda x: 0.0, spectral_device)
    np.testing.assert_array_equal(gradient, np.zeros_like(gradient))


def svd_reconstruction_gradient_probe(device="cpu"):
    """JSON-safe evidence for a jointly phase-invariant SVD reconstruction loss.

    This callable is also usable independently of pytest through runpy. Setup,
    forward and ordinary complex AD errors remain RuntimeError, so the test's
    narrow expected AssertionError cannot accidentally classify them as the
    missing SVD phase-coupling contribution.
    """
    value = np.array([[3 + 1j, 0.2 + 0.1j], [0.3 - 0.2j, 1 + 0.5j]], "complex64")
    weight = np.array([[0.7 + 0.4j, 0.8 - 0.1j], [0.9 + 0.3j, 1.2 + 0.6j]], "complex64")
    with jt.runtime.scope(use_cuda=int(device == "cuda")):
        with forbid_backend_fallbacks():
            x = jt.array(value, dtype="complex64")
            x.start_grad()
            p = jt.array(weight, dtype="complex64")
            u, s, vh = jt.linalg.svd(x)
            reconstruction = jt.matmul(u * s.reshape((1, -1)), vh)
            loss = (reconstruction * p.conj()).real.sum()
            gradient = jt.grad(loss, x)
            gradient.sync()
            location = gradient.location()
            if location != ("device" if device == "cuda" else "cpu"):
                raise RuntimeError("SVD gradient ran on the wrong device: " + location)
            actual = gradient.numpy().copy()
            reconstructed = reconstruction.numpy().copy()
            if not np.allclose(reconstructed, value, atol=2e-5, rtol=2e-5):
                raise RuntimeError("SVD forward reconstruction failed")
            # A direct linear loss is an analytic oracle: Re< X, P > has
            # complex gradient P, with no SVD, gauge choice or finite difference.
            direct_x = jt.array(value, dtype="complex64")
            direct_x.start_grad()
            direct = jt.grad((direct_x * p.conj()).real.sum(), direct_x)
            direct.sync()
            direct_array = direct.numpy().copy()
            if not np.allclose(direct_array, weight, atol=2e-5, rtol=2e-5):
                raise RuntimeError("The direct complex linear-loss gradient failed")
            return {
                "device": device,
                "gradient_location": location,
                "forward_max_absolute_error": float(
                    np.max(np.abs(reconstructed - value))
                ),
                "direct_gradient_max_absolute_error": float(
                    np.max(np.abs(direct_array - weight))
                ),
                "gradient_max_absolute_error": float(np.max(np.abs(actual - weight))),
                "actual_real": actual.real.tolist(),
                "actual_imag": actual.imag.tolist(),
                "expected_real": weight.real.tolist(),
                "expected_imag": weight.imag.tolist(),
            }


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "SVD adjoint drops the U/V diagonal phase coupling for jointly "
        "phase-invariant losses; confirmed by native CPU/CUDA regressions"
    ),
)
def test_svd_joint_phase_invariant_reconstruction_gradient(spectral_device):
    result = svd_reconstruction_gradient_probe(spectral_device)
    print(json.dumps(result, sort_keys=True))
    actual = np.asarray(result["actual_real"]) + 1j * np.asarray(result["actual_imag"])
    expected = np.asarray(result["expected_real"]) + 1j * np.asarray(
        result["expected_imag"]
    )
    np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=2e-3)

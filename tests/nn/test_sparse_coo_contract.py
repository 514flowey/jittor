"""COO metadata and construction checks must not require tensor execution."""

import numpy as np
import pytest
import jittor as jt

from _helpers import capability


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda":
        capability.require_accelerator("cuda")
    with jt.flag_scope(use_cuda=int(request.param == "cuda")):
        yield request.param


def make_coo(indices, values, shape):
    return jt.sparse.sparse_array(indices, values, jt.NanoVector(shape))


@pytest.mark.parametrize("index_dtype", ["int32", "int64", "uint8"])
@pytest.mark.parametrize("value_dtype", ["float32", "float64", "complex64"])
def test_metadata_preserves_input_identity_and_dtype(device, index_dtype, value_dtype):
    indices = jt.array([[0, 1, 1], [2, 0, 2]], dtype=index_dtype)
    values = jt.array([1, 2, 3], dtype=value_dtype)
    sparse = make_coo(indices, values, [2, 3])
    assert sparse.nnz == 3
    assert sparse.dtype == values.dtype
    assert str(sparse.dtype) == value_dtype
    assert sparse.is_sparse() is True
    assert sparse.ndim == 2
    assert tuple(sparse.shape) == (2, 3)
    assert sparse._indices() is indices
    assert sparse._values() is values


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("nnz", [0, 3])
def test_constructor_keeps_ndimensional_and_zero_nnz_storage(device, ndim, nnz):
    indices = jt.array(np.zeros((ndim, nnz), dtype=np.int32))
    values = jt.array(np.arange(nnz, dtype=np.float32))
    sparse = make_coo(indices, values, [2] * ndim)
    assert sparse.nnz == nnz
    assert sparse.ndim == ndim
    assert sparse._indices() is indices
    assert sparse._values() is values


def test_constructor_keeps_values_with_trailing_dimensions(device):
    # The existing storage contract only requires a leading nnz dimension;
    # this does not claim that matrix-only sparse operations accept it.
    indices = jt.array([[0, 1, 1], [2, 0, 2]], dtype="int32")
    values = jt.array(np.arange(6, dtype=np.float32).reshape(3, 2))
    sparse = make_coo(indices, values, [2, 3])
    assert sparse.nnz == 3
    assert sparse._values() is values


@pytest.mark.parametrize("index_shape", [(), (3,), (2, 1, 3)])
def test_constructor_rejects_indices_rank(device, index_shape):
    indices = jt.array(np.zeros(index_shape, dtype=np.int32))
    values = jt.array([1, 2, 3], dtype="float32")
    # Stop at construction: malformed coordinates must never reach a kernel.
    with pytest.raises(ValueError, match="indices.*shape"):
        make_coo(indices, values, [2, 3])


@pytest.mark.parametrize("coordinate_dims", [1, 3])
def test_constructor_rejects_coordinate_dimension_mismatch(device, coordinate_dims):
    indices = jt.array(np.zeros((coordinate_dims, 3), dtype=np.int32))
    values = jt.array([1, 2, 3], dtype="float32")
    with pytest.raises(ValueError, match="indices.*shape"):
        make_coo(indices, values, [2, 3])


@pytest.mark.parametrize("value_shape", [(), (0,), (2,), (4,)])
def test_constructor_rejects_values_nnz_mismatch(device, value_shape):
    indices = jt.array([[0, 1, 1], [2, 0, 2]], dtype="int32")
    values = jt.array(np.zeros(value_shape, dtype=np.float32))
    with pytest.raises(ValueError, match="values.*nnz"):
        make_coo(indices, values, [2, 3])


@pytest.mark.parametrize("index_dtype", ["bool", "float32", "float64", "complex64"])
def test_constructor_rejects_noninteger_indices(device, index_dtype):
    indices = jt.array([[0, 1, 1], [2, 0, 2]], dtype=index_dtype)
    values = jt.array([1, 2, 3], dtype="float32")
    with pytest.raises(TypeError, match="indices.*integer"):
        make_coo(indices, values, [2, 3])


@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_valid_coo_roundtrip_and_value_gradient_execute_on_device(device, index_dtype):
    indices = jt.array([[0, 1, 1], [2, 0, 2]], dtype=index_dtype)
    values = jt.array([1, 2, 3], dtype="float32")
    sparse = make_coo(indices, values, [2, 3])
    dense = sparse.to_dense()
    gradient = jt.grad(dense.sum(), values)
    roundtrip = sparse.to_csr().to_coo()
    restored = roundtrip.to_dense()
    assert roundtrip.nnz == 3
    assert roundtrip.dtype == values.dtype
    assert str(roundtrip._indices().dtype) == index_dtype
    assert roundtrip.is_sparse() is True
    # numpy() fetches device storage to CPU; check placement before any fetch.
    results = (dense, gradient, restored)
    for result in results:
        result.sync()
    for result in results:
        assert result.location() == ("device" if device == "cuda" else "cpu")
    expected = np.array([[0, 0, 1], [2, 0, 3]], dtype=np.float32)
    np.testing.assert_array_equal(dense.numpy(), expected)
    np.testing.assert_array_equal(restored.numpy(), expected)
    np.testing.assert_array_equal(gradient.numpy(), np.ones(3, dtype=np.float32))

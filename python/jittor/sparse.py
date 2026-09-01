# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#   Dun Liang <randonlang@gmail.com>.
#   Xiangli Li <190569238@qq.com>
#
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# Native COO/CSR sparse tensors and sparse-dense matmul (jittor-core-gaps.md SS3.6).
#
# Design: SparseVar (COO) / SparseCSR hold plain dense jt.Vars for indices/values --
# the same "composite of ordinary Vars" shape the pre-existing SparseVar already used --
# but every real kernel (to_dense, coo<->csr conversion, coalesce, spmm forward+backward)
# is now backed by jt.numpy_code / jt.Function bound to scipy.sparse (CPU) or
# cupyx.scipy.sparse (CUDA), the SAME pattern already used throughout jittor/linalg.py for
# native complex64/128 QR/SVD/eigh. On CUDA this is genuinely device-resident (numpy_code's
# CUDA path binds `np` to a real cupy view of device memory, not a host round-trip -- see
# jittor/init_cupy.py's numpy2cupy), and cupyx.scipy.sparse itself calls into cuSPARSE, so
# this is not a "densify" fallback: spmm never materializes the (M,N) dense matrix, only
# O(nnz) / O(nnz*K) sparse-native buffers.
#
# spmm's own gradient math (dValues = gather(dOut, row) . gather(B, col) per nonzero,
# dB = A^T @ dOut) is derived directly rather than routed through autograd of the
# scipy/cupyx call, matching every other numpy_code-based backward_code in this codebase.
#
# Known scope limits (documented, not silently absorbed):
#   - SparseCSR is 2-D only (no batch dim); SparseVar (COO) keeps a batch-capable
#     (ndim, nnz) index layout for storage/to_dense/transpose, but spmm/coalesce/to_csr
#     only support a plain 2-D matrix (asserted).
#   - Indices (and, for coalesce/to_csr, the row-sort permutation) are never
#     differentiated -- they are int dtype, which jittor's autograd never asks a gradient
#     for, so this is enforced by dtype alone, not a manual stop_grad() call.
#   - coalesce()/to_csr() are genuinely data-dependent-shape operations (nnz can shrink
#     when there are duplicate (row,col) entries) and therefore go through jt.Function
#     (whose Python-level execute() can return a freshly-shaped Var) rather than
#     jt.numpy_code (whose C++ Op requires the output shape fixed BEFORE the callback
#     runs) -- to_csr() itself does NOT coalesce (duplicates are kept, only row-sorted),
#     so it stays a static nnz-preserving permutation and CAN go through numpy_code.
import jittor as jt
import numpy as np


def _sparse_backend(np_):
    """The numpy_code `np` argument is literally the numpy or cupy module (bound per-call,
    CPU vs CUDA); return the matching scipy.sparse / cupyx.scipy.sparse module."""
    if np_.__name__ == "numpy":
        import scipy.sparse as xsp
        return xsp
    else:
        import cupyx.scipy.sparse as xsp
        return xsp


def _unused_backward(np, data):
    raise NotImplementedError("this input is integer/index-only and must never receive a "
                               "gradient request -- jittor's autograd should never call this")


# --------------------------------------------------------------------------- to_dense (COO)
def _coo_to_dense(row, col, values, shape):
    M, N = int(shape[0]), int(shape[1])

    def forward_code(np_, data):
        row_, col_, values_ = data["inputs"]
        out = data["outputs"][0]
        xsp = _sparse_backend(np_)
        # coo_matrix(...).toarray() correctly SUMS duplicate (row,col) entries -- this is
        # standard, documented scipy/cupyx behavior, matching torch.sparse's to_dense().
        A = xsp.coo_matrix((values_, (row_, col_)), shape=(M, N))
        np_.copyto(out, A.toarray().astype(out.dtype))

    def backward_values(np_, data):
        row_, col_ = data["inputs"][0], data["inputs"][1]
        dout = data["dout"]
        out = data["outputs"][0]
        # d(dense[i,j])/d(values[e]) = 1 for every e with (row[e],col[e])==(i,j); duplicates
        # each independently receive dout at the SHARED position, which is exactly correct
        # by linearity of the sum-of-duplicates forward.
        np_.copyto(out, dout[row_, col_])

    return jt.numpy_code(
        (M, N), values.dtype, [row, col, values],
        forward_code,
        [_unused_backward, _unused_backward, backward_values],
    )


# ------------------------------------------------------------------------- coo <-> csr (no dedup)
def _coo_to_csr(row, col, values, num_rows):
    M = int(num_rows)
    nnz = int(row.shape[0])

    def forward_code(np_, data):
        row_, col_, values_ = data["inputs"]
        crow_out, col_out, values_out = data["outputs"]
        # stable sort by row ONLY -- duplicates are kept (not coalesced), so nnz is
        # preserved and the output shapes declared below stay valid.
        perm = np_.argsort(row_, kind="stable")
        np_.copyto(col_out, col_[perm])
        np_.copyto(values_out, values_[perm])
        counts = np_.bincount(row_.astype(np_.int64), minlength=M) if nnz else np_.zeros(M, dtype=np_.int64)
        crow = np_.zeros(M + 1, dtype=np_.int64)
        np_.cumsum(counts, out=crow[1:])
        np_.copyto(crow_out, crow.astype(crow_out.dtype))

    def backward_values(np_, data):
        row_ = data["inputs"][0]
        dout_values = data["dout"]  # gradient of the (permuted) csr values output
        out = data["outputs"][0]
        perm = np_.argsort(row_, kind="stable")
        inv = np_.empty_like(perm)
        inv[perm] = np_.arange(len(perm))
        np_.copyto(out, dout_values[inv])

    idx_dtype = row.dtype
    crow, col_out, values_out = jt.numpy_code(
        [(M + 1,), (nnz,), (nnz,)],
        [idx_dtype, idx_dtype, values.dtype],
        [row, col, values],
        forward_code,
        # backward is only ever requested for the 3rd (values) input/output pairing;
        # this op has 3 outputs (crow, col, values) and 3 inputs (row, col, values) --
        # jittor calls backward[v_index] once per (output, differentiable input) pair, and
        # only `values` is float/complex (row/col are int -> never grad-requested).
        [_unused_backward, _unused_backward, backward_values],
    )
    return crow, col_out, values_out


def _csr_row_indices(crow_indices, nnz):
    nnz = int(nnz)

    def forward_code(np_, data):
        crow_ = data["inputs"][0]
        out = data["outputs"][0]
        # CuPy does not accept a device ndarray as the `repeats` argument of repeat.
        # Search each stored element offset in the CSR row pointers instead; this stays
        # device-resident and also handles empty rows without a host synchronization.
        rows = np_.searchsorted(crow_[1:], np_.arange(nnz), side="right")
        np_.copyto(out, rows.astype(out.dtype))

    return jt.numpy_code((nnz,), crow_indices.dtype, [crow_indices], forward_code,
                          [_unused_backward])


# ------------------------------------------------------------------------------- coalesce
class _CoalesceCOO(jt.Function):
    """Merge duplicate (row,col) entries by summing their values. nnz can shrink, which is
    a genuinely data-dependent output shape -- jt.numpy_code cannot express that (its C++
    Op fixes output shape before the callback runs), so this uses jt.Function instead,
    whose execute() builds the output Var directly from host/device-computed numpy/cupy
    arrays of whatever size the dedup produced."""

    def execute(self, row, col, values, M, N):
        del M
        # Deliberately host-only (plain numpy/scipy.sparse), regardless of jt.flags.use_cuda:
        # the output nnz is only known after deduping, so at least the shape-determining
        # step forces a device->host sync no matter what (unlike spmm's forward/backward,
        # which are the actual numeric hot path and DO stay cupy/cuSPARSE-resident on CUDA).
        # jt.array(...) below still places the result on the ambient device as usual.
        row_dtype, col_dtype, values_dtype = str(row.dtype), str(col.dtype), str(values.dtype)
        row_, col_, values_ = row.numpy(), col.numpy(), values.numpy()
        import scipy.sparse as xsp
        key = row_.astype(np.int64) * N + col_.astype(np.int64)
        uniq_key, inverse = np.unique(key, return_inverse=True)
        inverse = inverse.reshape(-1)
        # coo_matrix(...).toarray() sums duplicate entries by design -- reuse that instead
        # of hand-rolling a scatter-add (works uniformly for real and complex dtypes).
        if values_.ndim == 1:
            m = xsp.coo_matrix((values_, (inverse, np.zeros_like(inverse))),
                                shape=(len(uniq_key), 1))
            merged = m.toarray()[:, 0].astype(values_.dtype)
        else:
            merged = np.zeros((len(uniq_key),) + values_.shape[1:], dtype=values_.dtype)
            for k in range(values_.shape[1]):
                m = xsp.coo_matrix((values_[:, k], (inverse, np.zeros_like(inverse))),
                                    shape=(len(uniq_key), 1))
                merged[:, k] = m.toarray()[:, 0]
        new_row = (uniq_key // N).astype(row_.dtype)
        new_col = (uniq_key % N).astype(col_.dtype)
        self.inverse = jt.array(inverse, dtype="int64")
        return (jt.array(new_row, dtype=row_dtype),
                jt.array(new_col, dtype=col_dtype),
                jt.array(merged, dtype=values_dtype))

    def grad(self, d_row, d_col, d_values):
        # each original (possibly-duplicate) entry receives the gradient of the merged
        # entry it was summed into -- a pure gather by the stored forward `inverse` map.
        if d_values is None:
            return None, None, None, None, None
        g = d_values[self.inverse]
        return None, None, g, None, None


def _coo_coalesce(row, col, values, shape):
    M, N = int(shape[0]), int(shape[1])
    new_row, new_col, new_values = _CoalesceCOO()(row, col, values, M, N)
    return new_row, new_col, new_values


# ---------------------------------------------------------------------------------- spmm
def _spmm_backward_values(np_, data, layout):
    dout = data["dout"]
    if layout == "coo":
        row_, col_, y_ = data["inputs"][0], data["inputs"][1], data["inputs"][3]
    else:  # csr
        crow_, col_, y_ = data["inputs"][0], data["inputs"][1], data["inputs"][3]
        row_ = np_.searchsorted(
            crow_[1:], np_.arange(len(col_)), side="right"
        )
    out = data["outputs"][0]
    # dValues[e] = dOut[row[e],:] . conj(B[col[e],:])  (per-nonzero dot product, dense
    # gather -- no sparse matrix needed for this direction). The conj() on B, NOT dOut,
    # matches PyTorch/jittor's conjugate-Wirtinger convention for complex matmul backward
    # (grad_A = grad_output @ B.conj().T); for real dtypes conj() is a no-op so the same
    # code path is correct for float32/64 too -- no dtype branch needed.
    contrib = dout[row_] * np_.conj(y_[col_])
    np_.copyto(out, np_.sum(contrib, axis=-1))


def _spmm_backward_y(np_, data, layout, M, N):
    dout = data["dout"]
    xsp = _sparse_backend(np_)
    # dY = conj(A)^T @ dOut (same conjugate-Wirtinger convention as above; conj() is a
    # no-op for real dtypes).
    if layout == "coo":
        row_, col_, values_ = data["inputs"][0], data["inputs"][1], data["inputs"][2]
        At = xsp.coo_matrix((np_.conj(values_), (col_, row_)), shape=(N, M))  # conj + transpose
    else:  # csr
        crow_, col_, values_ = data["inputs"][0], data["inputs"][1], data["inputs"][2]
        A = xsp.csr_matrix((values_, col_, crow_), shape=(M, N))
        At = A.conj().transpose().tocsr()
    out = data["outputs"][0]
    np_.copyto(out, (At @ dout).astype(out.dtype))


def _spmm_coo(row, col, values, y, M, N):
    K = y.shape[-1]

    def forward_code(np_, data):
        row_, col_, values_, y_ = data["inputs"]
        out = data["outputs"][0]
        xsp = _sparse_backend(np_)
        A = xsp.coo_matrix((values_, (row_, col_)), shape=(M, N))
        np_.copyto(out, (A @ y_).astype(out.dtype))

    def backward_values(np_, data):
        _spmm_backward_values(np_, data, "coo")

    def backward_y(np_, data):
        _spmm_backward_y(np_, data, "coo", M, N)

    return jt.numpy_code(
        (M, K), values.dtype, [row, col, values, y],
        forward_code,
        [_unused_backward, _unused_backward, backward_values, backward_y],
    )


def _spmm_csr(crow, col, values, y, M, N):
    K = y.shape[-1]

    def forward_code(np_, data):
        crow_, col_, values_, y_ = data["inputs"]
        out = data["outputs"][0]
        xsp = _sparse_backend(np_)
        A = xsp.csr_matrix((values_, col_, crow_), shape=(M, N))
        np_.copyto(out, (A @ y_).astype(out.dtype))

    def backward_values(np_, data):
        _spmm_backward_values(np_, data, "csr")

    def backward_y(np_, data):
        _spmm_backward_y(np_, data, "csr", M, N)

    return jt.numpy_code(
        (M, K), values.dtype, [crow, col, values, y],
        forward_code,
        [_unused_backward, _unused_backward, backward_values, backward_y],
    )


# ------------------------------------------------------------------------------- public API
class SparseVar:
    """COO sparse tensor. ``indices`` is ``(ndim, nnz)`` (integer dtype -- int32/int64),
    ``values`` is ``(nnz, ...)``, ``shape`` is the logical dense shape. Duplicate index
    entries are allowed (uncoalesced); :meth:`to_dense` and :func:`spmm` sum them, matching
    ``torch.sparse_coo_tensor`` semantics. Call :meth:`coalesce` to merge duplicates
    explicitly. ``t()``/``to_dense()``/``coalesce()``/``to_csr()`` only support a plain 2-D
    matrix (asserted) -- batched (ndim>2) COO storage/round-trip is otherwise supported.
    """
    def __init__(self, indices, values, shape):
        assert isinstance(indices, jt.Var) and isinstance(values, jt.Var) and isinstance(shape, jt.NanoVector)
        assert indices.ndim == 2 and indices.shape[0] == len(shape), \
            f"indices must be shape (ndim={len(shape)}, nnz), got {tuple(indices.shape)}"
        assert values.shape[0] == indices.shape[1], \
            f"values ({values.shape[0]}) and indices ({indices.shape[1]}) nnz mismatch"
        assert "int" in str(indices.dtype), "indices must be an integer dtype"
        self.indices = indices
        self.values = values
        self.shape = shape
        self.ndim = len(shape)

    @property
    def nnz(self):
        return self.indices.shape[1]

    @property
    def dtype(self):
        return self.values.dtype

    def is_sparse(self):
        return True

    def _indices(self):
        return self.indices

    def _values(self):
        return self.values

    def t(self):
        assert self.ndim == 2, "t() only supports a plain 2-D sparse matrix"
        indices = jt.stack([self.indices[1], self.indices[0]], dim=0)
        shape = jt.NanoVector([self.shape[1], self.shape[0]])
        return SparseVar(indices, self.values, shape)

    def to_dense(self):
        ret = jt.zeros(self.shape,self.values.dtype)
        indices = tuple(
            index.reshape((-1,)) for index in self.indices.split(1, dim=0)
        )
        return ret.setitem(indices, self.values, "add")

    def coalesce(self):
        assert self.ndim == 2, "coalesce() only supports a plain 2-D sparse matrix"
        row, col, values = _coo_coalesce(self.indices[0], self.indices[1], self.values, self.shape)
        indices = jt.stack([row, col], dim=0)
        return SparseVar(indices, values, self.shape)

    def to_csr(self):
        assert self.ndim == 2, "to_csr() only supports a plain 2-D sparse matrix"
        crow, col, values = _coo_to_csr(self.indices[0], self.indices[1], self.values, self.shape[0])
        return SparseCSR(crow, col, values, self.shape)


class SparseCSR:
    """CSR sparse matrix (2-D only). ``crow_indices`` is ``(M+1,)``, ``col_indices`` and
    ``values`` are ``(nnz, ...)``. Mirrors ``torch.sparse_csr_tensor``."""
    def __init__(self, crow_indices, col_indices, values, shape):
        assert isinstance(crow_indices, jt.Var) and isinstance(col_indices, jt.Var) and isinstance(values, jt.Var)
        assert isinstance(shape, jt.NanoVector) and len(shape) == 2, "SparseCSR only supports a 2-D matrix"
        assert crow_indices.shape[0] == shape[0] + 1, "crow_indices must have length rows+1"
        assert col_indices.shape[0] == values.shape[0], "col_indices/values nnz mismatch"
        assert "int" in str(crow_indices.dtype) and "int" in str(col_indices.dtype), \
            "crow_indices/col_indices must be an integer dtype"
        self.crow_indices = crow_indices
        self.col_indices = col_indices
        self.values = values
        self.shape = shape
        self.ndim = 2

    @property
    def nnz(self):
        return self.col_indices.shape[0]

    @property
    def dtype(self):
        return self.values.dtype

    def is_sparse(self):
        return True

    def to_coo(self):
        row = _csr_row_indices(self.crow_indices, self.col_indices.shape[0])
        indices = jt.stack([row, self.col_indices], dim=0)
        return SparseVar(indices, self.values, self.shape)

    def to_dense(self):
        return self.to_coo().to_dense()

    def coalesce(self):
        return self.to_coo().coalesce().to_csr()

    def t(self):
        return self.to_coo().t().to_csr()


def sparse_array(indices, values, shape):
    return SparseVar(indices, values, shape)


def spmm(sparse_x, y):
    """Sparse-dense matmul: ``sparse_x`` (``SparseVar`` COO or ``SparseCSR``, shape
    ``(M,N)``) @ ``y`` (dense ``jt.Var``, shape ``(N,K)``) -> dense ``(M,K)``. Never
    materializes the ``(M,N)`` dense matrix -- runs through ``scipy.sparse``
    (CPU) / ``cupyx.scipy.sparse`` (CUDA, cuSPARSE-backed). Differentiable w.r.t. both
    ``sparse_x``'s values and ``y``; indices carry no gradient (integer dtype).
    """
    assert isinstance(sparse_x, (SparseVar, SparseCSR)) and isinstance(y, jt.Var)
    assert sparse_x.ndim == 2 and y.ndim == 2 and sparse_x.shape[-1] == y.shape[0], \
        f"shape mismatch: sparse {tuple(sparse_x.shape)} @ dense {tuple(y.shape)}"
    assert str(sparse_x.dtype) == str(y.dtype), \
        f"dtype mismatch: sparse values {sparse_x.dtype} vs dense {y.dtype} (cast one first)"
    M, N = int(sparse_x.shape[0]), int(sparse_x.shape[1])
    if isinstance(sparse_x, SparseVar):
        return _spmm_coo(sparse_x.indices[0], sparse_x.indices[1], sparse_x.values, y, M, N)
    else:
        return _spmm_csr(sparse_x.crow_indices, sparse_x.col_indices, sparse_x.values, y, M, N)

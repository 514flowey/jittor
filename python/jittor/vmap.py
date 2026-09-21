# A real batching transform (vmap): a native jt.vmap, independent of
# compat/torch's own approximate `torch.vmap` shim (a CPU-only, loop/stack-
# based fallback -- see compat/torch/installers/numerical.py and
# refactor-wip/architecture/vmap-owner-plan.md, which scopes extending that
# shell but does not plan a native entry point). This module is that native
# entry point: `jt.vmap`, usable directly on plain jt.Var, with no torch-
# compat dependency.
#
# Design: a BatchedVar wraps a real jt.Var whose physical axis 0 is a batch
# dimension the user's function never sees. `vmap(f)` wraps mapped args into
# BatchedVar and calls `f` once; every op `f` performs is intercepted (either
# directly on BatchedVar, or via a small set of patched jt/Var entry points)
# and re-dispatched with the batch dimension threaded through, producing ONE
# real op graph with a genuine batch dimension baked in -- not a per-sample
# Python loop. Because it is a real op graph, jt.grad/vjp/jvp differentiate
# through it with no extra code (Jittor's autograd is graph-driven, see
# src/grad.cc): vmap(grad(f)) and grad(sum(vmap(f))) work for free.
#
# Nested vmap(vmap(f)): BatchedVar.value may itself be a lower-level
# BatchedVar, so a value can carry several stacked batch axes (added
# outermost-first, so physically axis 0 is the first (outermost) vmap's batch
# dim, axis 1 the second's, etc.). Elementwise ops (unary/binary/matmul/
# einsum) don't care about specific axis positions, so they simply peel one
# level, recurse, and rewrap -- correct regardless of nesting depth. Axis-
# sensitive ops (reduce/transpose/reshape/unsqueeze/getitem/setitem) instead
# work on the fully-unwrapped physical Var directly, shifting the user's axis
# by the *total* nesting depth (see `_nesting_depth`/`_physical`/`_rewrap_like`)
# -- a naive "+1 per recursion level" is WRONG here because a level's own
# batch axis is not always at physical position 0 once other levels are
# nested inside it.
#
# Scope ("core set"): elementwise unary/binary (incl. conj), real/imag,
# reduce, reshape/transpose/broadcast, matmul/einsum, getitem/setitem (basic
# + simple fancy indexing), detach/start_grad/stop_grad/requires_grad,
# random (both randomness="different" and "same"), and the single-input
# factorizations qr/svd/svdvals/eigh/inv (these are already implemented as
# `(..., M, N)`-batched `numpy_code` ops with NO op-specific batch-axis
# handling of their own, so vmap's extra leading physical axis is just one
# more leading batch dim they already accept -- see `_apply_linalg`).
# conv/pool and other nn layers, sparse ops, two-operand linalg (`solve` --
# batch-shape broadcasting between the two operands is a separate,
# not-yet-implemented rule), custom numpy_code ops, and fancy-index corner
# cases (batched index arrays, nested-depth mismatches in setitem) are
# explicitly out of scope and fail loud rather than silently looping or
# producing wrong results.
#
# Composing with jvp/vjp: python/jittor/autograd/functional.py's vjp/jvp are
# themselves plain Python functions built entirely out of ops in the scope
# above (grad/sum/detach/start_grad/stop_grad/requires_grad/reshape/
# real/imag/...), so once those ops have BatchedVar batching rules,
# `vmap(lambda x, v: jvp(f, x, v))` / `vmap(lambda x, v: vjp(f, x, v))`
# compose with no vjp/jvp-specific code at all -- the same "it's just a real
# op graph" argument as vmap(grad(f)) above, just one level higher. The two
# concrete gaps that would otherwise break this are (a)
# detach/start_grad/stop_grad/requires_grad simply not existing as
# BatchedVar operations (jvp/vjp's preprocessing calls them on every
# input/output), and (b) `.real`/`.imag`/`.conj()` not being batchable
# either, so no complex-valued (quantum-state-style) computation could flow
# through vjp/jvp under vmap. Both are covered above/below. Nested
# vmap(vmap(...)) composes for the same structural reason (every op in
# scope is recursion-correct at any nesting depth, per the "peel one level"
# argument above); create_graph=True (higher-order jvp/vjp) likewise needs
# no special-casing, since it only changes whether `.detach()`/
# `.stop_grad()` are called at all in preprocessing, not what they do.
import jittor as jt
from collections.abc import Sequence as _Sequence

__all__ = ["vmap", "BatchedVar"]

_level_counter = [0]

def _new_level():
    _level_counter[0] += 1
    return _level_counter[0]

# Stack of (level, randomness, batch_size) for currently-executing vmap()
# calls, outermost first. jt.random() has no BatchedVar argument to key off
# of (its args are plain shape/dtype), so the only way to give it correct
# "randomness" semantics (see _apply_random) is this ambient context that
# vmap's wrapper pushes/pops around calling `func`.
_active_batch_context = []


class BatchedVar:
    __slots__ = ("value", "level")

    def __init__(self, value, level):
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "level", level)

    def _nesting_depth(self):
        d, v = 0, self
        while isinstance(v, BatchedVar):
            d += 1
            v = v.value
        return d

    def _physical(self):
        v = self.value
        while isinstance(v, BatchedVar):
            v = v.value
        return v

    @property
    def shape(self):
        return self._physical().shape[self._nesting_depth():]

    @property
    def ndim(self):
        return self._physical().ndim - self._nesting_depth()

    @property
    def dtype(self):
        return self._physical().dtype

    def __repr__(self):
        return f"<BatchedVar level={self.level} shape={tuple(self.shape)} dtype={self.dtype}>"

    def __len__(self):
        return int(self.shape[0])

    def __getattr__(self, name):
        if name in _UNARY_NAMES:
            return lambda: _apply_unary(name, self)
        if name in _REDUCE_NAMES:
            return lambda dim=None, keepdims=False: _apply_reduce(name, self, dim, keepdims)
        if name in _MAXMIN_NAMES:
            return lambda dim=None, keepdims=False: _apply_maxmin(name, self, dim, keepdims)
        if name in _BINARY_NAMES:
            return lambda other: _apply_binary(name, self, other)
        if name in ("cast", "astype"):
            return lambda *args, **kwargs: _apply_cast(self, *args, **kwargs)
        if name in ("reshape", "view"):
            return lambda *shape: _apply_reshape(self, _flatten_shape_args(shape))
        if name in ("transpose", "permute"):
            return lambda *dims: _apply_transpose(self, _flatten_shape_args(dims) or None)
        if name == "unsqueeze":
            return lambda dim: _apply_unsqueeze(self, dim)
        if name in ("argmax", "argmin"):
            return lambda dim, keepdims=False: _apply_argreduce(name, self, dim, keepdims)
        # jittor-core-gaps.md §3.1: gradfunctional's jvp/vjp (via
        # _grad_preprocess/_grad_postprocess/_fill_in_zeros) call these on
        # every input/output, so composing vmap(jvp(f))/vmap(vjp(f)) needs
        # them to work on a BatchedVar. All three only touch autograd
        # bookkeeping (a NodeFlags bit, or a real-but-shape-preserving detach
        # op), never axis positions, so -- like the elementwise family above
        # -- they simply peel one level, recurse, and rewrap/return self,
        # correct at any nesting depth. `jt.Var.detach/start_grad/stop_grad`
        # are never monkeypatched (unlike the ops above): a call always
        # lands directly on a single Var with no "other operand" that could
        # itself be a BatchedVar, so there's no mixed-argument case to
        # intercept -- only the BatchedVar side of the dispatch is needed.
        if name == "detach":
            return lambda: _apply_detach(self)
        if name == "start_grad":
            return lambda: _apply_start_grad(self)
        if name == "stop_grad":
            return lambda: _apply_stop_grad(self)
        if name in ("nelement", "numel"):
            return lambda: _logical_numel(self)
        raise NotImplementedError(
            f"vmap: no batching rule for `.{name}` -- supported ops are limited to "
            "elementwise unary/binary, reduce (sum/mean/max/min/prod/argmax/argmin), "
            "cast/astype, reshape/transpose/permute/unsqueeze, matmul/einsum, getitem/setitem, "
            "detach/start_grad/stop_grad, and random. Avoid calling this op inside a "
            "vmapped function, or restructure so it runs outside vmap.")

    def __getitem__(self, index):
        return _apply_getitem(self, index)

    def __setitem__(self, index, value):
        _apply_setitem(self, index, value)

    # jittor-core-gaps.md §3.1: `.real`/`.imag` are properties (not zero-arg
    # methods, unlike `.conj()` which is folded into _UNARY_NAMES below), so
    # they can't go through the generic __getattr__ dispatch above -- Python
    # only calls __getattr__ once ordinary attribute/property lookup fails,
    # and a `property` object here IS that lookup. Elementwise/shape-
    # preserving like the rest of the unary family: peel one level, recurse,
    # rewrap. This (plus `conj`) was a real, previously undocumented gap:
    # any complex-valued (quantum-state-style) expectation value computation
    # inside vmap needs at least one of conj/real/imag and none were
    # batchable (see 2026-08-26-core-gaps-3.1-3.8-verification.md §6.2(b)).
    @property
    def real(self):
        return BatchedVar(_apply_real(self.value), self.level)

    @property
    def imag(self):
        return BatchedVar(_apply_imag(self.value), self.level)

    # jittor-core-gaps.md §3.1: gradfunctional's `_grad_preprocess`/
    # `_autograd_grad`/`_check_requires_grad` all read (and, for
    # `requires_grad_`, write) `.requires_grad`; it is a thin view over the
    # same stop-grad NodeFlags bit `.stop_grad()`/`.start_grad()` above
    # mutate, so it must be defined as a real property here (not routed
    # through __getattr__, for the same "Python checks properties before
    # __getattr__" reason as `.real`/`.imag`) rather than raising
    # NotImplementedError like an unhandled attribute would.
    @property
    def requires_grad(self):
        return _get_requires_grad(self)

    @requires_grad.setter
    def requires_grad(self, flag):
        _set_requires_grad(self, flag)


def _flatten_shape_args(args):
    # Mirrors __init__.py's reshape()/transpose(): a single sequence-like
    # positional (list/tuple/NanoVector/other Sequence) is the shape/dims
    # itself, not a length-1 shape. NanoVector isn't a (list,tuple) so it
    # must be checked for explicitly, same as the functions being wrapped.
    if len(args) == 1 and isinstance(args[0], (list, tuple, jt.NanoVector, _Sequence)):
        return list(args[0])
    return list(args)


def _rewrap_like(a, real_result):
    # Reconstruct the same chain of BatchedVar(level=...) wrappers as `a`,
    # pointing to `real_result` (which must already carry all of a's batch
    # axes, since axis-sensitive ops below operate on the fully unwrapped
    # physical Var directly).
    if not isinstance(a, BatchedVar):
        return real_result
    return BatchedVar(_rewrap_like(a.value, real_result), a.level)


def _moveaxis(v, src, dst):
    # v: plain jt.Var or BatchedVar. src/dst are relative to v's own full
    # ndim (its own `.ndim`, which for a BatchedVar already excludes nothing
    # extra -- it's exactly the rank the caller should reason about).
    nd = v.ndim
    s = src if src >= 0 else src + nd
    d = dst if dst >= 0 else dst + nd
    if s == d:
        return v
    rest = [i for i in range(nd) if i != s]
    perm = rest[:d] + [s] + rest[d:]
    return _apply_transpose(v, perm) if isinstance(v, BatchedVar) else _ORIG_TRANSPOSE(v, perm)


# ---------------------------------------------------------------------------
# Elementwise family (unary/binary/matmul/einsum): shape-position-agnostic,
# so a simple "peel one level, recurse, rewrap" is correct at any nesting depth.
# ---------------------------------------------------------------------------

def _lvl(x):
    return x.level if isinstance(x, BatchedVar) else -1


def _apply_unary(name, a):
    if not isinstance(a, BatchedVar):
        return _ORIG_UNARY[name](a)
    return BatchedVar(_apply_unary(name, a.value), a.level)


def _apply_cast(a, *args, **kwargs):
    if not isinstance(a, BatchedVar):
        return _ORIG_CAST(a, *args, **kwargs)
    # Dtype conversion preserves every batch level and delegates its gradient
    # semantics to the native op, including native unsupported conversions.
    return BatchedVar(_apply_cast(a.value, *args, **kwargs), a.level)


def _apply_real(a):
    if not isinstance(a, BatchedVar):
        return a.real
    return BatchedVar(_apply_real(a.value), a.level)


def _apply_imag(a):
    if not isinstance(a, BatchedVar):
        return a.imag
    return BatchedVar(_apply_imag(a.value), a.level)


# ---------------------------------------------------------------------------
# Autograd bookkeeping (jittor-core-gaps.md §3.1): detach/start_grad/
# stop_grad/requires_grad only flip a NodeFlags bit or make a shape-
# preserving copy -- never touch axis positions -- so, like the elementwise
# family above, they simply peel one level, recurse, and rewrap/return self.
# These are the specific glue gradfunctional's vjp/jvp needs (via
# _grad_preprocess/_grad_postprocess/_autograd_grad/_check_requires_grad) to
# work when its inputs/outputs are BatchedVar, i.e. when vjp/jvp is called
# *inside* a vmapped function (`vmap(lambda x, v: jvp(f, x, v))`).
# ---------------------------------------------------------------------------

def _apply_detach(a):
    if not isinstance(a, BatchedVar):
        return a.detach()
    return BatchedVar(_apply_detach(a.value), a.level)


def _apply_start_grad(a):
    if not isinstance(a, BatchedVar):
        a.start_grad()
        return a
    _apply_start_grad(a.value)
    return a


def _apply_stop_grad(a):
    if not isinstance(a, BatchedVar):
        a.stop_grad()
        return a
    _apply_stop_grad(a.value)
    return a


def _get_requires_grad(a):
    if not isinstance(a, BatchedVar):
        return a.requires_grad
    return _get_requires_grad(a.value)


def _set_requires_grad(a, flag):
    if not isinstance(a, BatchedVar):
        a.requires_grad = flag
        return
    _set_requires_grad(a.value, flag)


def _logical_numel(a):
    n = 1
    for s in a.shape:
        n *= int(s)
    return n


def _apply_like(fn, a, *args, **kwargs):
    # Batching rule for zeros_like/ones_like/full_like. This is NOT just a
    # convenience: gradfunctional's jvp() builds its "double backward trick"
    # seed via jt.zeros_like(out) (see _zeros_seed_like) and that seed MUST
    # itself be a genuinely per-example-independent BatchedVar, not merely a
    # value that *broadcasts* correctly against one. A plain (non-batched)
    # zero broadcasts fine for elementwise use, but differentiating a
    # per-example loss w.r.t. ONE SHARED scalar seed collapses to the SUM
    # over the batch (by the ordinary chain rule for a truly shared
    # variable) instead of yielding each example's own gradient -- jvp
    # silently returned `sum_i jvp_i` repeated across every batch slot
    # instead of each example's own jvp before this fix. Elementwise/shape-
    # preserving like the rest of the unary family: peel one level, recurse,
    # rewrap.
    if not isinstance(a, BatchedVar):
        return fn(a, *args, **kwargs)
    return BatchedVar(_apply_like(fn, a.value, *args, **kwargs), a.level)


def _apply_linalg(fn, a, *args, **kwargs):
    # Batching rule for the single-input factorizations qr/svd/svdvals/eigh/
    # inv (jittor-core-gaps.md §3.2). Unlike matmul/einsum, these need no
    # axis bookkeeping at all: each is already implemented (linalg.py, via
    # jt.numpy_code wrapping np.linalg.*) to treat every leading dim before
    # the trailing matrix dims as an independent batch dim, with the
    # batching done by NumPy/CuPy itself, not by any jittor-side "which axis
    # is dim 0" logic. So vmap's physical batch axis (already at position 0,
    # ahead of any of the function's own logical batch dims) is just one
    # more such leading dim for free -- peel one BatchedVar level, recurse
    # on the physical value, rewrap every output (qr/eigh return a 2-tuple,
    # svd a 2- or 3-tuple depending on `compute_uv`, svdvals/inv a single
    # Var) at the same level. Exactly the "elementwise family" peel-recurse-
    # rewrap argument above, generalized from one output to a tuple of them.
    if not isinstance(a, BatchedVar):
        return fn(a, *args, **kwargs)
    lvl = a.level
    inner = _apply_linalg(fn, a.value, *args, **kwargs)
    # (tuple, list): a jt.Function-based multi-output op (e.g. qr, since
    # jittor-core-gaps.md §3.3) returns a plain `list` from a multi-output
    # `execute()`, not a `tuple` -- checking only `tuple` here silently
    # treated that whole list as ONE output instead of rewrapping each
    # element, a real bug caught by test_vmap_qr_matches_per_sample.
    if isinstance(inner, (tuple, list)):
        return tuple(BatchedVar(r, lvl) for r in inner)
    return BatchedVar(inner, lvl)


def _align_logical_rank(av, bv):
    # Jittor's real binary ops broadcast right-aligned with no notion of "the
    # leading axis is a special batch axis" -- so if av/bv (each already
    # carrying its own batch axis at physical position 0, or none at all) have
    # different LOGICAL ranks, plain delegation would broadcast the batch axis
    # against a logical axis and corrupt/reject the shapes (e.g. (B,3)+(B,)
    # naively fails/misaligns). Insert size-1 axes right after the batch axes
    # of the shorter operand -- mirrors how ordinary (unbatched) broadcasting
    # implicitly left-pads the shorter shape with 1s.
    if not (hasattr(av, "ndim") and hasattr(bv, "ndim")):
        return av, bv  # a bare Python scalar operand broadcasts fine untouched
    nd_a, nd_b = av.ndim, bv.ndim
    if nd_a < nd_b:
        for _ in range(nd_b - nd_a):
            av = _apply_unsqueeze(av, 0)
    elif nd_b < nd_a:
        for _ in range(nd_a - nd_b):
            bv = _apply_unsqueeze(bv, 0)
    return av, bv


def _apply_binary(name, a, b):
    lvl = max(_lvl(a), _lvl(b))
    if lvl == -1:
        return _ORIG_BINARY[name](a, b)
    # Align logical rank BEFORE unwrapping .value: _apply_unsqueeze needs the
    # still-wrapped BatchedVar to know where its batch axis is (a bare
    # physical Var no longer carries that information, so aligning after
    # unwrapping would insert the padding axis in the wrong physical spot).
    a, b = _align_logical_rank(a, b)
    av = a.value if _lvl(a) == lvl else a
    bv = b.value if _lvl(b) == lvl else b
    return BatchedVar(_apply_binary(name, av, bv), lvl)


def _apply_matmul(a, b):
    lvl = max(_lvl(a), _lvl(b))
    if lvl == -1:
        return _ORIG_MATMUL(a, b)
    # jt.matmul (like numpy.matmul) promotes a bare 1-D operand to a row/
    # column vector for the multiply and squeezes the inserted axis back out
    # afterward. That promotion must happen at the LOGICAL (unbatched) rank
    # -- via .ndim, which already excludes the batch axis -- BEFORE peeling
    # off BatchedVar wrappers below: otherwise the real (physical) matmul
    # sees the leading batch axis as an ordinary matrix dimension and
    # misinterprets/rejects the shapes. This was a real, previously
    # undocumented gap: "shared (unbatched) weight matrix @ batched vector"
    # -- the single most common vmap pattern -- used to fail outright (see
    # 2026-08-26-core-gaps-3.1-3.8-verification.md §6.2(a) for the repro:
    # `U(3,3) @ x` where `x` is vmapped raised "dimension not match").
    a_vec = a.ndim == 1
    b_vec = b.ndim == 1
    if a_vec:
        a = _apply_unsqueeze(a, 0)
    if b_vec:
        b = _apply_unsqueeze(b, -1)
    av = a.value if _lvl(a) == lvl else a
    bv = b.value if _lvl(b) == lvl else b
    # matmul already broadcasts arbitrary leading dims (nn.py), so once both
    # operands are resolved to real Vars (or a lower-level BatchedVar,
    # handled by recursion) at this level, no further reshape is needed.
    result = BatchedVar(_apply_matmul(av, bv), lvl)
    if b_vec:
        result = _squeeze_logical(result, -1)
    if a_vec:
        result = _squeeze_logical(result, -1 if b_vec else -2)
    return result


def _squeeze_logical(v, dim):
    # Drop a size-1 LOGICAL axis via reshape (works for both BatchedVar,
    # through its own patched `.reshape`, and a plain Var, through the
    # (possibly-patched-but-passthrough) `jt.Var.reshape`).
    shape = list(v.shape)
    nd = len(shape)
    d = dim if dim >= 0 else dim + nd
    return v.reshape(shape[:d] + shape[d + 1:])


def _apply_random(shape, *args, **kwargs):
    if not _active_batch_context:
        return _ORIG_RANDOM(shape, *args, **kwargs)
    shape = [shape] if isinstance(shape, int) else list(shape)
    # jittor-core-gaps.md §3.1 acceptance #8: "different" (default) means
    # every batch element gets its own independent draw -- which is exactly
    # what jt.random already does per-element, so generating the full
    # [level_batch_sizes..., *shape] tensor directly needs no extra code.
    # "same" means every element along that level's batch axis must see the
    # IDENTICAL sample -- implemented by adding NO physical axis for that
    # level and NOT wrapping the result in a BatchedVar for it: an unwrapped
    # value already broadcasts against that level exactly like any other
    # closure-captured (non-batched) operand does elsewhere in this module
    # (see _align_logical_rank / the "shared weight matrix" case above).
    # Nested vmaps with mixed modes are handled correctly because each
    # active level is decided independently and only "different" levels
    # contribute a physical axis, in original outermost-first order --
    # matching how nested BatchedVar wrapping is defined everywhere else.
    diff_levels = [(lvl, bs) for lvl, mode, bs in _active_batch_context if mode == "different"]
    full_shape = [bs for _, bs in diff_levels] + shape
    real = _ORIG_RANDOM(full_shape, *args, **kwargs)
    result = real
    for lvl, _ in diff_levels:
        result = BatchedVar(result, lvl)
    return result


def _apply_grad(loss, targets, retain_graph=True):
    # vmap(grad(f)) / per-example gradients: since the vmapped forward pass
    # makes example i's loss depend only on example i's inputs (there is no
    # cross-batch mixing anywhere in the batching rules above), summing the
    # (per-example) loss to a true scalar and differentiating ONCE gives
    # exactly the per-example gradient back out -- d(sum_i loss_i)/d(target_j)
    # collapses to d(loss_j)/d(target_j) because all cross terms are zero.
    # This is the standard "vmap-of-grad via sum-then-backward" trick.
    single = not isinstance(targets, (list, tuple))
    tlist = [targets] if single else list(targets)
    if not isinstance(loss, BatchedVar) and not any(isinstance(t, BatchedVar) for t in tlist):
        return _ORIG_GRAD(loss, targets, retain_graph)
    real_loss = loss._physical() if isinstance(loss, BatchedVar) else loss
    while real_loss.ndim > 0:
        real_loss = real_loss.sum()
    real_targets = [t._physical() if isinstance(t, BatchedVar) else t for t in tlist]
    grads = _ORIG_GRAD(real_loss, real_targets, retain_graph)
    wrapped = [_rewrap_like(t, g) if isinstance(t, BatchedVar) else g for t, g in zip(tlist, grads)]
    return wrapped[0] if single else wrapped


def _apply_einsum(spec, *operands):
    lvls = [_lvl(o) for o in operands]
    lvl = max(lvls)
    if lvl == -1:
        return _ORIG_EINSUM(spec, *operands)
    in_spec, out_spec = _parse_einsum_spec(spec, len(operands))
    used = set("".join(in_spec) + out_spec)
    batch_label = next(c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in used)
    new_in, new_operands = [], []
    for o, l, lbl in zip(operands, lvls, in_spec):
        if l == lvl:
            new_in.append(batch_label + lbl)
            new_operands.append(o.value)
        else:
            new_in.append(lbl)
            new_operands.append(o)
    new_spec = ",".join(new_in) + "->" + batch_label + out_spec
    return BatchedVar(_apply_einsum(new_spec, *new_operands), lvl)


def _parse_einsum_spec(spec, n):
    if "->" not in spec:
        raise NotImplementedError("vmap: einsum without an explicit '->' output spec "
            "is not supported; pass an explicit output subscript.")
    lhs, out_spec = spec.split("->")
    in_spec = lhs.split(",")
    if len(in_spec) != n:
        raise ValueError(f"vmap: einsum spec has {len(in_spec)} operand labels but {n} operands were given")
    return in_spec, out_spec


# ---------------------------------------------------------------------------
# Axis-sensitive family (reduce/transpose/reshape/unsqueeze/getitem/setitem):
# operate directly on the fully-unwrapped physical Var, shifting the user's
# axis by the TOTAL nesting depth, then rewrap.
# ---------------------------------------------------------------------------

def _shift_dim_by(dim, nd, depth, required=False):
    if dim is None:
        if required:
            raise ValueError("vmap: this op requires an explicit dim")
        # Full reduction (no dim given) must still spare the `depth` leading
        # batch axes -- reduce over exactly the logical (unbatched) axes.
        return list(range(depth, depth + nd))
    if isinstance(dim, (list, tuple)):
        return [depth + (d if d >= 0 else d + nd) for d in dim]
    return depth + (dim if dim >= 0 else dim + nd)


def _apply_reduce(name, a, dim, keepdims):
    if not isinstance(a, BatchedVar):
        return _ORIG_REDUCE[name](a) if dim is None else _ORIG_REDUCE[name](a, dim, keepdims)
    depth = a._nesting_depth()
    shifted = _shift_dim_by(dim, a.ndim, depth)
    real = a._physical()
    result = _ORIG_REDUCE[name](real, shifted, keepdims)
    return _rewrap_like(a, result)


def _apply_maxmin(name, a, dim=None, keepdims=False):
    # Native max/min return values only, including when dim is explicit.
    # The optional legacy Torch frontend can instead return (values, indices).
    # Preserve that result structure; indexing a native Var as if it were a
    # pair would discard physical batch rows and silently change the shape.
    if dim is None:
        return _apply_reduce(name, a, dim, keepdims)
    if not isinstance(a, BatchedVar):
        return _ORIG_REDUCE[name](a, dim, keepdims)
    depth = a._nesting_depth()
    shifted = _shift_dim_by(dim, a.ndim, depth, required=True)
    real = a._physical()
    result = _ORIG_REDUCE[name](real, shifted, keepdims)
    if isinstance(result, (tuple, list)):
        return tuple(_rewrap_like(a, value) for value in result)
    return _rewrap_like(a, result)


def _apply_argreduce(name, a, dim, keepdims):
    if not isinstance(a, BatchedVar):
        return _ORIG_ARGREDUCE[name](a, dim, keepdims)
    depth = a._nesting_depth()
    shifted = _shift_dim_by(dim, a.ndim, depth, required=True)
    real = a._physical()
    idx, val = _ORIG_ARGREDUCE[name](real, shifted, keepdims)
    return _rewrap_like(a, idx), _rewrap_like(a, val)


def _apply_reshape(a, shape):
    if not isinstance(a, BatchedVar):
        return _ORIG_RESHAPE(a, shape)
    depth = a._nesting_depth()
    real = a._physical()
    batch_dims = list(real.shape[:depth])
    result = _ORIG_RESHAPE(real, batch_dims + list(shape))
    return _rewrap_like(a, result)


def _apply_transpose(a, dims):
    if not isinstance(a, BatchedVar):
        return _ORIG_TRANSPOSE(a) if dims is None else _ORIG_TRANSPOSE(a, dims)
    depth = a._nesting_depth()
    nd = a.ndim
    real = a._physical()
    if dims is None:
        new_dims = list(range(depth)) + [depth + nd - 1 - i for i in range(nd)]
    else:
        new_dims = list(range(depth)) + [depth + (d if d >= 0 else d + nd) for d in dims]
    result = _ORIG_TRANSPOSE(real, new_dims)
    return _rewrap_like(a, result)


def _apply_unsqueeze(a, dim):
    if not isinstance(a, BatchedVar):
        return _ORIG_UNSQUEEZE(a, dim)
    depth = a._nesting_depth()
    nd = a.ndim
    real = a._physical()
    d = dim if dim >= 0 else dim + nd + 1
    result = _ORIG_UNSQUEEZE(real, depth + d)
    return _rewrap_like(a, result)


def _apply_broadcast(a, shape, dims=None):
    if not isinstance(a, BatchedVar):
        return _ORIG_BROADCAST(a, shape, dims) if dims is not None else _ORIG_BROADCAST(a, shape)
    depth = a._nesting_depth()
    real = a._physical()
    batch_dims = list(real.shape[:depth])
    new_shape = batch_dims + list(shape)
    new_dims = [i for i in range(depth)] if dims is None else \
        list(range(depth)) + [depth + (d if d >= 0 else d + len(shape)) for d in dims]
    result = _ORIG_BROADCAST(real, new_shape, new_dims)
    return _rewrap_like(a, result)


def _apply_getitem(a, index):
    if not isinstance(a, BatchedVar):
        return _ORIG_GETITEM(a, index)
    if not isinstance(index, tuple):
        index = (index,)
    if any(isinstance(i, BatchedVar) for i in index):
        raise NotImplementedError(
            "vmap: batched index arrays (data-dependent fancy indexing) are not "
            "supported in the core batching-rule set; restructure to use gather() "
            "explicitly, or avoid vmap for this operation.")
    depth = a._nesting_depth()
    real = a._physical()
    new_index = (slice(None),) * depth + index
    result = real[new_index]
    return _rewrap_like(a, result)


def _apply_setitem(a, index, value):
    if not isinstance(a, BatchedVar):
        _ORIG_SETITEM(a, index, value.value if isinstance(value, BatchedVar) else value)
        return
    if not isinstance(index, tuple):
        index = (index,)
    if any(isinstance(i, BatchedVar) for i in index):
        raise NotImplementedError(
            "vmap: batched index arrays in setitem are not supported in the core "
            "batching-rule set.")
    depth = a._nesting_depth()
    real = a._physical()
    new_index = (slice(None),) * depth + index
    if isinstance(value, BatchedVar):
        if value._nesting_depth() != depth:
            raise NotImplementedError(
                "vmap: setitem with a value batched at a different nesting depth "
                "than the target is not supported in the core batching-rule set.")
        v = value._physical()
    else:
        v = value
    real[new_index] = v


# ---------------------------------------------------------------------------
# pytree: a small, self-contained flatten/unflatten (list/tuple/dict), not
# borrowed from torch_compat.py's shim (core functionality must not depend on
# the optional torch-compat layer). Mirrors its recursive structure.
# ---------------------------------------------------------------------------

class _LeafSpec:
    pass

class _TreeSpec:
    def __init__(self, kind, context, children):
        self.kind = kind
        self.context = context
        self.children = children

def _tree_flatten(x):
    leaves = []
    def rec(o):
        if isinstance(o, dict):
            keys = list(o.keys())
            return _TreeSpec(dict, keys, [rec(o[k]) for k in keys])
        if isinstance(o, (list, tuple)):
            return _TreeSpec(type(o), None, [rec(c) for c in o])
        leaves.append(o)
        return _LeafSpec()
    spec = rec(x)
    return leaves, spec

def _tree_unflatten(leaves, spec):
    it = iter(leaves)
    def rec(s):
        if isinstance(s, _LeafSpec):
            return next(it)
        children = [rec(c) for c in s.children]
        if s.kind is dict:
            return {k: v for k, v in zip(s.context, children)}
        if s.kind is tuple:
            return tuple(children)
        return list(children)
    return rec(spec)


# ---------------------------------------------------------------------------
# vmap() entry point
# ---------------------------------------------------------------------------

def _broadcast_spec(dims, leaves):
    n = len(leaves)
    if isinstance(dims, (list, tuple)):
        flat, _ = _tree_flatten(tuple(dims))
        if len(flat) != n:
            raise ValueError(f"vmap: in_dims/out_dims has {len(flat)} entries but there are {n} leaves")
        return flat
    return [dims] * n


def _infer_batch_size(flat_args, flat_dims):
    size = None
    for a, d in zip(flat_args, flat_dims):
        if d is None:
            continue
        s = int(a.shape[d])
        if size is None:
            size = s
        elif size != s:
            raise ValueError(f"vmap: inconsistent batch size across mapped arguments ({size} vs {s})")
    if size is None:
        raise ValueError("vmap: at least one argument must have a non-None in_dims entry")
    return size


def vmap(func, in_dims=0, out_dims=0, randomness="different"):
    ''' Vectorize `func` over a batch dimension by building one real batched
    op graph, instead of looping over the batch in Python. Differentiable:
    jt.grad/vjp/jvp work through the result with no extra code, since the
    underlying graph is made of ordinary Jittor ops.

    :param func: function to vectorize; may return a Var or a pytree of Vars.
        Prefer `vmap(lambda x: jt.linalg.qr(x))` over the bare `vmap(jt.linalg.qr)`:
        the batching patches below are installed lazily, the first time any
        `vmap()` call runs in the process, and a bare op reference is
        resolved by the caller *before* that installation happens, so on a
        process's very first vmap call it can capture the pre-patch,
        BatchedVar-oblivious op and fail with a confusing low-level error
        instead of dispatching correctly. A lambda re-resolves the attribute
        at call time (after patching), so it is not affected -- as is any
        call after the first `vmap()` use in the process, since installation
        is idempotent and permanent.
    :param in_dims: int, None, or a pytree matching `args` -- which axis of
        each argument to map over (None = broadcast that argument unchanged).
    :param out_dims: int or a pytree matching func's return value -- where to
        place the batch axis in each output.
    :param randomness: "different" (default) -- every random op call inside
        `func` produces an independent sample per batch element, matching
        jt.random's own native per-element-independent semantics one level
        up. "same" -- every batch element sees the IDENTICAL sample for each
        random op call (drawn once at the logical, unbatched shape and
        broadcast against the batch, same as any other non-mapped
        closure-captured value).

    Supported ops inside `func`: elementwise unary/binary (incl. conj),
    real/imag, cast/astype, reduce (sum/mean/max/min/prod/argmax/argmin), reshape/
    transpose/permute/unsqueeze/broadcast, matmul/einsum, getitem/setitem
    (incl. simple fancy indexing), detach/start_grad/stop_grad (so
    gradfunctional's vjp/jvp compose: `vmap(lambda x, v: jvp(f, x, v))`),
    qr/svd/svdvals/eigh/inv, and random. Anything else raises
    NotImplementedError.
    '''
    if randomness not in ("different", "same"):
        raise ValueError(f"vmap: unknown randomness={randomness!r}")
    install_batching_patches()

    def wrapped(*args, **kwargs):
        level = _new_level()
        flat_args, spec = _tree_flatten(args)
        flat_in_dims = _broadcast_spec(in_dims, flat_args)
        batch_size = _infer_batch_size(flat_args, flat_in_dims)
        batched_flat = [
            (BatchedVar(_moveaxis(a, d, 0), level) if d is not None else a)
            for a, d in zip(flat_args, flat_in_dims)
        ]
        batched_args = _tree_unflatten(batched_flat, spec)
        _active_batch_context.append((level, randomness, batch_size))
        try:
            out = func(*batched_args, **kwargs)
        finally:
            _active_batch_context.pop()
        out_is_container = isinstance(out, (list, tuple, dict))
        flat_out, out_spec = _tree_flatten(out if out_is_container else (out,))
        flat_out_dims = _broadcast_spec(out_dims, flat_out)
        result = []
        for o, d in zip(flat_out, flat_out_dims):
            if isinstance(o, BatchedVar) and o.level == level:
                inner = o.value
            else:
                base = o if isinstance(o, jt.Var) else (o.value if isinstance(o, BatchedVar) else jt.array(o))
                inner = _ORIG_BROADCAST(base, [batch_size] + list(base.shape), [0])
            result.append(_moveaxis(inner, 0, d))
        return _tree_unflatten(result, out_spec) if out_is_container else result[0]
    return wrapped


# ---------------------------------------------------------------------------
# Registering original callables + patching entry points
# ---------------------------------------------------------------------------

_BINARY_NAMES = ["add", "subtract", "multiply", "divide", "floor_divide", "mod", "pow",
    "less", "less_equal", "greater", "greater_equal", "equal", "not_equal",
    "left_shift", "right_shift", "bitwise_and", "bitwise_or", "bitwise_xor",
    "minimum", "maximum", "logical_and", "logical_or", "logical_xor"]

_UNARY_NAMES = ["abs", "negative", "logical_not", "bitwise_not", "log", "exp", "sqrt",
    "round", "floor", "ceil", "round_int", "floor_int", "ceil_int",
    "sin", "asin", "sinh", "asinh", "tan", "atan", "tanh", "atanh",
    "cos", "acos", "cosh", "acosh", "sigmoid", "erf", "erfinv", "relu",
    "conj"]

_REDUCE_NAMES = ["sum", "mean", "prod"]
_MAXMIN_NAMES = ["max", "min"]

_BINARY_DUNDERS = {
    "add": ("__add__", "__radd__"), "subtract": ("__sub__", "__rsub__"),
    "multiply": ("__mul__", "__rmul__"), "divide": ("__truediv__", "__rtruediv__"),
    "floor_divide": ("__floordiv__", "__rfloordiv__"), "mod": ("__mod__", "__rmod__"),
    "pow": ("__pow__", "__rpow__"), "less": ("__lt__", None), "less_equal": ("__le__", None),
    "greater": ("__gt__", None), "greater_equal": ("__ge__", None),
    "equal": ("__eq__", None), "not_equal": ("__ne__", None),
    "left_shift": ("__lshift__", "__rlshift__"), "right_shift": ("__rshift__", "__rrshift__"),
    "bitwise_and": ("__and__", "__rand__"), "bitwise_or": ("__or__", "__ror__"),
    "bitwise_xor": ("__xor__", "__rxor__"),
}
_UNARY_DUNDERS = {"abs": "__abs__", "negative": "__neg__"}

_ORIG_BINARY, _ORIG_UNARY, _ORIG_REDUCE = {}, {}, {}
_ORIG_ARGREDUCE = {}
_ORIG_CAST = _ORIG_VAR_CAST = None
_ORIG_RESHAPE = _ORIG_TRANSPOSE = _ORIG_UNSQUEEZE = _ORIG_BROADCAST = None
_ORIG_MATMUL = _ORIG_EINSUM = None
_ORIG_GETITEM = _ORIG_SETITEM = None
_ORIG_GRAD = None
_ORIG_RANDOM = None
_PATCHED = False


def _make_binary_dunder_forward(name, original=None):
    def f(self, other):
        if original is not None and not isinstance(other, BatchedVar):
            return original(self, other)
        return _apply_binary(name, self, other)
    return f

def _make_binary_dunder_reverse(name, original=None):
    def f(self, other):
        if original is not None and not isinstance(other, BatchedVar):
            return original(self, other)
        return _apply_binary(name, other, self)
    return f

def _make_unary_dunder(name):
    def f(self):
        return _apply_unary(name, self)
    return f

def _make_named_binary(name):
    def f(a, b):
        return _apply_binary(name, a, b)
    return f

def _make_named_unary(name):
    def f(a):
        return _apply_unary(name, a)
    return f

def _norm_reduce_args(args, kwargs):
    # torch_compat's reduce wrappers accept dim/dims/axis and keepdim/keepdims
    # in either positional or keyword form (see torch_compat.py's
    # _norm_reduce_kw); a fixed `(a, dim=None, keepdims=False)` signature here
    # would reject any of those other spellings for a batched call. Only
    # matters when `a` IS a BatchedVar -- the non-batched fast path below
    # forwards *args/**kwargs to the original untouched.
    args = list(args)
    dim = None
    if args:
        dim = args.pop(0)
    elif "dim" in kwargs:
        dim = kwargs.pop("dim")
    elif "dims" in kwargs:
        dim = kwargs.pop("dims")
    elif "axis" in kwargs:
        dim = kwargs.pop("axis")
    keepdims = False
    if args:
        keepdims = args.pop(0)
    elif "keepdims" in kwargs:
        keepdims = kwargs.pop("keepdims")
    elif "keepdim" in kwargs:
        keepdims = kwargs.pop("keepdim")
    return dim, keepdims


def _make_named_reduce(name):
    def f(a, *args, **kwargs):
        if not isinstance(a, BatchedVar):
            return _ORIG_REDUCE[name](a, *args, **kwargs)
        dim, keepdims = _norm_reduce_args(args, kwargs)
        return _apply_reduce(name, a, dim, keepdims)
    return f


def _install(namespace, name, fn):
    try:
        setattr(namespace, name, fn)
    except (AttributeError, TypeError):
        pass


def install_batching_patches():
    ''' Idempotent: monkeypatches jt.Var/jt entry points so that any of them
    receiving a BatchedVar argument dispatches to the matching batching rule,
    while a call with no BatchedVar involved is unaffected (falls through to
    the original implementation with zero added overhead beyond one isinstance
    check). Called once, lazily, the first time jt.vmap() is used. '''
    global _PATCHED, _ORIG_RESHAPE, _ORIG_TRANSPOSE, _ORIG_UNSQUEEZE, _ORIG_BROADCAST
    global _ORIG_MATMUL, _ORIG_EINSUM, _ORIG_GETITEM, _ORIG_SETITEM, _ORIG_GRAD, _ORIG_RANDOM
    global _ORIG_CAST, _ORIG_VAR_CAST
    if _PATCHED:
        return
    _PATCHED = True

    for name in _BINARY_NAMES:
        # Prefer the free function (`jt.less`, not `jt.Var.less`): the free
        # function is what public callers actually use for a scalar/scalar
        # or scalar/ndarray call like `jt.less(1, 2)`, and it promotes
        # non-Var arguments before dispatching. `jt.Var.less` is a bound
        # method descriptor -- calling it directly with a non-Var first
        # argument (exactly the non-batched fallback path below does when
        # neither operand is a BatchedVar) fails with "descriptor 'less'
        # for 'jittor_core.Var' objects doesn't apply to a 'int' object"
        # instead of the correct promoted result.
        orig = getattr(jt, name, None) or getattr(jt.Var, name, None)
        if orig is None:
            continue
        _ORIG_BINARY[name] = orig
        patched = _make_named_binary(name)
        _install(jt.Var, name, patched)
        _install(jt, name, patched)
        fwd, rev = _BINARY_DUNDERS.get(name, (None, None))
        if fwd:
            original = getattr(jt.Var, fwd)
            _install(jt.Var, fwd, _make_binary_dunder_forward(name, original))
        if rev:
            original = getattr(jt.Var, rev)
            _install(jt.Var, rev, _make_binary_dunder_reverse(name, original))

    for name in _UNARY_NAMES:
        orig = getattr(jt.Var, name, None) or getattr(jt, name, None)
        if orig is None:
            continue
        _ORIG_UNARY[name] = orig
        patched = _make_named_unary(name)
        _install(jt.Var, name, patched)
        _install(jt, name, patched)
        dunder = _UNARY_DUNDERS.get(name)
        if dunder:
            _install(jt.Var, dunder, _make_unary_dunder(name))

    for name in _REDUCE_NAMES:
        orig = getattr(jt.Var, name, None) or getattr(jt, name, None)
        if orig is None:
            continue
        _ORIG_REDUCE[name] = orig
        patched = _make_named_reduce(name)
        _install(jt.Var, name, patched)
        _install(jt, name, patched)

    for name in _MAXMIN_NAMES:
        orig = getattr(jt.Var, name, None) or getattr(jt, name, None)
        if orig is None:
            continue
        _ORIG_REDUCE[name] = orig
        def make_maxmin(name=name):
            def f(a, *args, **kwargs):
                if not isinstance(a, BatchedVar):
                    return _ORIG_REDUCE[name](a, *args, **kwargs)
                dim, keepdims = _norm_reduce_args(args, kwargs)
                return _apply_maxmin(name, a, dim, keepdims)
            return f
        _install(jt.Var, name, make_maxmin())
        _install(jt, name, make_maxmin())

    for name in ("argmax", "argmin"):
        _ORIG_ARGREDUCE[name] = getattr(jt.Var, name, None) or getattr(jt, name)
        def make(name=name):
            def f(a, *args, **kwargs):
                if not isinstance(a, BatchedVar):
                    return _ORIG_ARGREDUCE[name](a, *args, **kwargs)
                dim, keepdims = _norm_reduce_args(args, kwargs)
                return _apply_argreduce(name, a, dim, keepdims)
            return f
        _install(jt.Var, name, make())
        _install(jt, name, make())

    _ORIG_CAST, _ORIG_VAR_CAST = jt.cast, jt.Var.cast
    def _cast(x, *args, **kwargs):
        # The free function accepts scalar/array inputs and x=/op= keywords;
        # a Var method descriptor cannot stand in for that native entry point.
        return _apply_cast(x, *args, **kwargs)
    def _var_cast(self, *args, **kwargs):
        if not isinstance(self, BatchedVar):
            return _ORIG_VAR_CAST(self, *args, **kwargs)
        return _apply_cast(self, *args, **kwargs)
    _install(jt, "cast", _cast)
    _install(jt.Var, "cast", _var_cast)
    _install(jt.Var, "astype", _var_cast)

    _ORIG_RESHAPE = jt.Var.reshape
    def _reshape(a, *shape):
        return _apply_reshape(a, _flatten_shape_args(shape))
    _install(jt.Var, "reshape", _reshape)
    _install(jt.Var, "view", _reshape)
    _install(jt, "reshape", _reshape)

    _ORIG_TRANSPOSE = jt.Var.transpose
    def _transpose(a, *dims):
        if not isinstance(a, BatchedVar):
            return _ORIG_TRANSPOSE(a, *dims)
        return _apply_transpose(a, _flatten_shape_args(dims) or None)
    _install(jt.Var, "transpose", _transpose)
    _install(jt.Var, "permute", _transpose)
    _install(jt, "transpose", _transpose)
    _install(jt, "permute", _transpose)

    _ORIG_UNSQUEEZE = jt.Var.unsqueeze
    def _unsqueeze(a, dim):
        return _apply_unsqueeze(a, dim)
    _install(jt.Var, "unsqueeze", _unsqueeze)
    _install(jt, "unsqueeze", _unsqueeze)

    _ORIG_BROADCAST = jt.Var.broadcast
    def _broadcast(a, shape, dims=None):
        return _apply_broadcast(a, shape, dims)
    _install(jt.Var, "broadcast", _broadcast)
    _install(jt.Var, "broadcast_var", _broadcast)
    _install(jt, "broadcast", _broadcast)

    _ORIG_MATMUL = jt.matmul
    def _matmul(a, b):
        return _apply_matmul(a, b)
    _install(jt, "matmul", _matmul)
    _install(jt.Var, "matmul", _matmul)
    _install(jt.Var, "__matmul__", _matmul)

    for name in ("qr", "svd", "svdvals", "eigh", "inv"):
        orig = getattr(jt.linalg, name, None)
        if orig is None:
            continue
        def make_linalg(orig=orig):
            def f(a, *args, **kwargs):
                return _apply_linalg(orig, a, *args, **kwargs)
            return f
        _install(jt.linalg, name, make_linalg())

    _ORIG_EINSUM = jt.linalg.einsum
    def _einsum(spec, *operands):
        return _apply_einsum(spec, *operands)
    _install(jt.linalg, "einsum", _einsum)
    _install(jt, "einsum", _einsum)

    _ORIG_GRAD = jt.grad
    def _grad(loss, targets, retain_graph=True):
        return _apply_grad(loss, targets, retain_graph)
    _install(jt, "grad", _grad)

    # Only patch the top-level jt.random -- it internally calls ops.random(),
    # so ALSO patching jt.ops.random would re-intercept every call inside
    # itself and prepend the batch size again on each re-entry (infinite
    # recursion / runaway shape growth).
    _ORIG_RANDOM = jt.random
    def _random(shape, *args, **kwargs):
        return _apply_random(shape, *args, **kwargs)
    _install(jt, "random", _random)

    for name in ("zeros_like", "ones_like", "full_like"):
        orig = getattr(jt, name, None)
        if orig is None:
            continue
        def make_like(orig=orig):
            def f(a, *args, **kwargs):
                return _apply_like(orig, a, *args, **kwargs)
            return f
        _install(jt, name, make_like())

    _ORIG_GETITEM = jt.Var.__getitem__
    _ORIG_SETITEM = jt.Var.__setitem__
    _install(jt.Var, "__getitem__", lambda a, index: _apply_getitem(a, index))
    _install(jt.Var, "__setitem__", lambda a, index, value: _apply_setitem(a, index, value))

    # BatchedVar is not a jt.Var, so it needs its own copies of the operator
    # dunders (patching jt.Var alone doesn't make `batched + 1` work, since
    # Python looks up __add__ on BatchedVar's own type first).
    for name, (fwd, rev) in _BINARY_DUNDERS.items():
        _install(BatchedVar, fwd, _make_binary_dunder_forward(name))
        if rev:
            _install(BatchedVar, rev, _make_binary_dunder_reverse(name))
    for name, dunder in _UNARY_DUNDERS.items():
        _install(BatchedVar, dunder, _make_unary_dunder(name))
    _install(BatchedVar, "__matmul__", lambda a, b: _apply_matmul(a, b))
    _install(BatchedVar, "__rmatmul__", lambda a, b: _apply_matmul(b, a))

"""One kernel launch for a whole parameter list's SGD update.

The portable update is two elementwise ops per parameter plus a holder
rebind, and a transformer has a lot of parameters: an 8-layer d512 model has
96, and the update loop measured 1.11 ms of a 7.41 ms training step -- the
optimizer, not the model. PyTorch does not pay that because its SGD is a
`foreach` kernel: one launch for the whole list.

This is the same idea. Every parameter's pointer, gradient pointer, output
pointer and length travel in one by-value argument struct, and one kernel
walks them all. The struct is what bounds the batch: CUDA gives a kernel 4 KB
of parameter space, so the tensors are processed in chunks that fit.
"""

import jittor as jt
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from jittor._runtime.dispatch import register_kernel


#: Tensors per launch. Each one costs up to five pointers and a length in
#: the argument struct (44 bytes with momentum), and a kernel's parameter
#: space is 4 KB; 80 is 3520 bytes, which leaves room for the scalars and the
#: ABI's own overhead.
_CHUNK = 80


def _supports_fused_sgd(tensors, *args, **kwargs):
    """float32, dense, and allocated: this writes raw pointers.

    `tensors` is every Var the kernel dereferences -- parameters, gradients and
    velocities -- not the parameter list. The kernel's argument struct declares
    all three families as `float*`, so a float16 gradient against a float32
    parameter is a compile error, not a slow path, and that pair is exactly what
    `auto_mixed_precision_level` 4/5/6 produce. See the call site in
    `jittor/optim/algorithms/sgd.py`.
    """
    if not tensors:
        return False
    for p in tensors:
        if not isinstance(p, jt.Var):
            return False
        if _jittor_dtype_name(p.dtype) != "float32":
            return False
        if not p._storage_is_contiguous():
            return False
    return True


def _source(count, momentum, weight_decay, dampening, nesterov, first_step):
    """File-scope CUDA for exactly this configuration.

    The coefficients are baked in as literals rather than passed: they do not
    change between steps, and a branch on `momentum == 0` inside the inner
    loop would be evaluated once per element. `first_step` is baked in the
    same way -- it changes exactly once per parameter group (the step-1 ->
    step-2 transition), so this costs one extra JIT compile there, not a
    recompile every step.
    """
    plain = momentum == 0 and not nesterov
    wd = f"{float(weight_decay):.9e}f"
    mom = f"{float(momentum):.9e}f"
    damp = f"{float(dampening):.9e}f"
    # dp: the gradient, with weight decay folded in when there is any.
    dp = "g" if weight_decay == 0 else f"fmaf(p, {wd}, g)"
    if plain:
        body = f"""
                float p = arg.param[t][i];
                float g = arg.grad[t][i];
                float dp = {dp};
                arg.dst[t][i] = p - dp * lr;"""
    else:
        # First real touch of the velocity buffer: seed it from the raw
        # (undampened) dp, matching PyTorch's `buf = dp.clone()` -- the
        # dampened recurrence below is only correct from the second update
        # onward (jittor-core-gaps.md §3.3).
        # The new velocity goes to its own output, not back into the input
        # buffer: an in-place write is invisible to the graph, so reading the
        # buffer before the parameter was evaluated returned the old value
        # (and a state_dict saved right after step() could store it).
        if first_step:
            step = "arg.vel_dst[t][i] = v = dp;"
        else:
            step = f"arg.vel_dst[t][i] = v = fmaf({mom}, arg.vel[t][i], dp * (1.0f - {damp}));"
        use = f"fmaf({mom}, v, dp)" if nesterov else "v"
        body = f"""
                float p = arg.param[t][i];
                float g = arg.grad[t][i];
                float dp = {dp};
                float v;
                {step}
                arg.dst[t][i] = p - ({use}) * lr;"""
    vel = "" if plain else f"float* vel[{count}];\n        float* vel_dst[{count}];"
    # No member may be called `out`: the code op's JIT template does
    # `#define out out0` around the body, so `args.out[k]` would be rewritten
    # to `args.out0[k]` and nvcc reports a struct with no such member.
    return f"""
    struct FusedSgdArgs {{
        float* param[{count}];
        float* grad[{count}];
        float* dst[{count}];
        {vel}
        int len[{count}];
    }};
    __global__ static void fused_sgd_kernel(FusedSgdArgs arg, float lr) {{
        const int t = blockIdx.y;
        const int n = arg.len[t];
        const int stride = blockDim.x * gridDim.x;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {{
            {body}
        }}
    }}
    """


def _fused_sgd_cuda(entries, lr, momentum, weight_decay, dampening, nesterov, step=1):
    """`entries` is a list of (param, grad, velocity). Returns [(new_p, new_v)]."""
    plain = momentum == 0 and not nesterov
    first_step = step <= 1
    results = []
    for start in range(0, len(entries), _CHUNK):
        chunk = entries[start:start + _CHUNK]
        count = len(chunk)
        params = [e[0] for e in chunk]
        grads = [e[1] for e in chunk]
        vels = [e[2] for e in chunk]
        inputs = params + grads + ([] if plain else vels)
        setup = []
        for k in range(count):
            setup.append(f"args.param[{k}] = in{k}_p;")
            setup.append(f"args.grad[{k}] = in{count + k}_p;")
            setup.append(f"args.dst[{k}] = out{k}_p;")
            if not plain:
                setup.append(f"args.vel[{k}] = in{2 * count + k}_p;")
                setup.append(f"args.vel_dst[{k}] = out{count + k}_p;")
            setup.append(f"args.len[{k}] = {int(params[k].numel())};")
        longest = max(int(p.numel()) for p in params)
        blocks = max(1, min(256, (longest + 255) // 256))
        # The struct and the kernel go in the header: `cuda_src` is spliced
        # into the body of `CodeOp::jit_run`, and a definition there is block
        # scope -- nvcc answers "a block-scope function may only have extern
        # storage class". Only the launch belongs in `cuda_src`.
        header = _source(count, momentum, weight_decay, dampening, nesterov, first_step)
        body = f"""
        FusedSgdArgs args;
        {chr(10).join('        ' + line for line in setup)}
        fused_sgd_kernel<<<dim3({blocks}, {count}), 256>>>(args, {float(lr):.9e}f);
        """
        outputs = params if plain else params + vels
        outs = jt.code(
            [p.shape for p in outputs],
            [p.dtype for p in outputs],
            inputs,
            cuda_header=header,
            cuda_src=body,
        )
        if not isinstance(outs, (list, tuple)):
            outs = [outs]
        # Without momentum the velocity is neither read nor written, so it is
        # handed back unchanged; with momentum it is the kernel's own output.
        results.extend(zip(outs[:count], vels if plain else outs[count:]))
    return results


register_kernel("optim.sgd_fused", "cuda", _fused_sgd_cuda,
                dtypes=("float32",), supports=_supports_fused_sgd)

__all__ = ["_fused_sgd_cuda"]

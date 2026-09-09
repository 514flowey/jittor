// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "mem/allocator.h"
#include "mem/allocator/cuda_dual_allocator.h"
#include "event_queue.h"
#endif
#include <cstring>
#include <cmath>
#include "var.h"
#include "ops/array_op.h"
#include "misc/cuda_flags.h"
#include "mem/allocator.h"
#include "mem/swap.h"

namespace jittor {

#ifdef HAS_CUDA
#pragma GCC visibility push(hidden)
namespace array_local {
cudaStream_t stream;
cudaEvent_t event;

struct Init {
Init() {
    if (!get_device_count()) return;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    checkCudaErrors(cudaEventCreate(&event, cudaEventDisableTiming));
}
~Init() {
    if (!get_device_count()) return;
    peekCudaErrors(cudaDeviceSynchronize());
    peekCudaErrors(cudaStreamDestroy(stream));
    peekCudaErrors(cudaEventDestroy(event));
}
} init;

}
#pragma GCC visibility pop
using namespace array_local;

#endif

ArrayOp::ArrayOp(const void* ptr, NanoVector shape, NanoString dtype)
    : ArrayOp(ArrayArgs{ptr, shape, dtype}) {}

DECLARE_FLAG(int, use_cuda_host_allocator);

ArrayOp::ArrayOp(ArrayArgs&& args) {
    output = create_output(args.shape, args.dtype);
    NanoVector shape = output->shape;
    if (output->num == 1 && shape.size() <= 1) {
        output->flags.set(NodeFlags::_force_fuse);
        output->flags.set(NodeFlags::_is_scalar);
        set_type(OpType::element);
    }
    #ifdef HAS_CUDA
    if (use_cuda && !save_mem && !use_cuda_host_allocator) {
        flags.set(NodeFlags::_cpu, 0);
        flags.set(NodeFlags::_cuda, 1);
        if (!output->flags.get(NodeFlags::_force_fuse)) {
            // free prev allocation first
            event_queue.flush();
            // alloc new allocation
            auto size = output->size;
            new (&allocation) Allocation(&cuda_dual_allocator, size);
            auto host_ptr = cuda_dual_allocator.get_dual_allocation(allocation.allocation).host_ptr;
            std::memcpy(host_ptr, args.ptr, output->size);
            return;
        }
    }
    #endif
    // TODO: args.buffer too many copy
    new (&allocation) Allocation(cpu_allocator, output->size);
    std::memcpy(allocation.ptr, args.ptr, output->size);
}

void ArrayOp::jit_prepare(JK& jk) {
    // Gate on _is_scalar, NOT _force_fuse: _is_scalar is set once at
    // construction (output->num==1 && shape.size()<=1) and never changes,
    // and is exactly what also permanently pins this op's OpType to
    // `element` (see the ctor below) -- which is what makes executor.cc
    // route it through the FusedOp/jit_fused_ops-keyed compile-and-cache
    // path (OpType::other ops instead run via a direct, non-JIT op->run()
    // call and must NOT get a jit_prepare contribution here, since ArrayOp
    // has no jit_run() -- see the confirmed-root-cause note below for why
    // that distinction matters). _force_fuse, in contrast, starts equal to
    // _is_scalar but can be CLEARED LATER by fuser.cc's count_fuse() (when
    // this scalar's consumers don't all agree on a broadcast shape) --
    // gating the dtype (and value) encoding below on _force_fuse meant a
    // scalar whose _force_fuse got cleared silently produced an EMPTY,
    // dtype-blind jit_key contribution, identical for every dtype.
    //
    // Confirmed root cause (jittor-core-gaps.md §3.5 investigation, see
    // agent/workdocs/2026-09-05-core-gaps-fixes.md): two structurally
    // identical size-1 FusedOps wrapping scalar ArrayOps of DIFFERENT
    // dtypes (a float32 `make_number(1.0)` gradient seed from one
    // `jt.grad()` call, and an unrelated float64 `make_number(1.0)` seed
    // from a later, independent call) produced byte-identical jit_keys
    // whenever _force_fuse had been cleared this way, so the second call's
    // FusedOp reused the first's cached compiled entry (jit_fused_ops is a
    // plain string-keyed map) -- silently executing the float64 scalar
    // through a kernel compiled for float32 data and materializing 0.0
    // instead of 1.0. This was the root cause of the qr/rq "silent
    // wrong/zero backward" defect.
    if (output->flags.get(NodeFlags::_is_scalar)) {
        jk << "«T:" << output->dtype();
        // fill or find cbuffer for const var pass
        if (output->dtype().dsize() == 4) {
            auto x = std::abs(ptr<int32>()[0]);
            auto y = std::abs(ptr<float32>()[0]);
            auto z = ptr<uint32>()[0];
            if ((x<=2) || (y==1.0f || y==2.0f))
                jk << "«o:" << z;
        } else if (output->dtype().dsize() == 8) {
            // Same const-buffer fast path as the dsize()==4 branch above,
            // extended to 8-byte scalars (float64/int64) -- these were
            // previously excluded entirely, which is what let two
            // differently-valued 8-byte scalars collide on jit_key (see
            // above). z is the low 32 bits only here; that's fine, it's
            // just a filter for "is this among the handful of very common
            // small constants" -- the "T:" dtype tag above is what actually
            // guarantees dtype-safety, this refinement only decides whether
            // the *value* also gets baked in as a further cache-key
            // differentiator/optimization for common constants.
            auto x = std::abs(ptr<int64>()[0]);
            auto y = std::abs(ptr<float64>()[0]);
            auto z = ptr<uint32>()[0];
            if ((x<=2) || (y==1.0 || y==2.0))
                jk << "«o:" << z << ":" << ptr<uint32>()[1];
        }
        // end of fill cbuffer
    }
}

void ArrayOp::run() {
    #ifdef HAS_CUDA
    if (allocation.allocator == &cuda_dual_allocator) {
        auto host_ptr = cuda_dual_allocator.get_dual_allocation(allocation.allocation).host_ptr;
        checkCudaErrors(cudaMemcpyAsync(
            allocation.ptr, host_ptr, allocation.size, cudaMemcpyHostToDevice, stream));
        checkCudaErrors(cudaEventRecord(event, stream));
        checkCudaErrors(cudaStreamWaitEvent(0, event, 0));
        // delay free this allocation
        allocation.allocator = &delay_free;
    }
    #endif
    // free prev allocation and move into it
    auto o = output;
    if (save_mem)
        free_with_swap(o);
    else
        o->allocator->free(o->mem_ptr, o->size, o->allocation);
    
    o->mem_ptr = allocation.ptr;
    allocation.ptr = nullptr;
    o->allocator = allocation.allocator;
    o->allocation = allocation.allocation;
    if (save_mem) registe_swap(o);
}

} // jittor

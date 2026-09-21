// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <random>

#include "core/var.h"
#include "runtime/init.h"
#include <cuda_runtime.h>
#include <curand.h>
#include "helper_cuda.h"
#include "curand_random_op.h"
#include "curand_wrapper.h"
#include "core/executor.h"
#include "runtime/random_generator.h"

namespace jittor {

#ifndef JIT
CurandRandomOp::CurandRandomOp(NanoVector shape, NanoString dtype, NanoString type, RandomGenerator* generator) {
    set_flag(OpFlags::_cuda, 1);
    // curand generates float and double only. Anything else used to expand to
    // curandGenerate*Double against a pointer of the wrong type and fail deep
    // inside nvcc; say so here instead. jt.random() already lowers float16 and
    // bfloat16 to a float32 draw plus a cast.
    USER_CHECK(dtype == ns_float32 || dtype == ns_float64)
        << "curand_random supports float32 and float64 only, got" << dtype
        << "\n  Draw float32 and cast if another dtype is needed.";
    output = create_output(shape, dtype);
    this->type = type;
    if (generator) this->rand_gen = generator->state;
    USER_CHECK(type == ns_normal || type == ns_uniform);
}

void CurandRandomOp::jit_prepare(JK& jk) {
    jk << "«T:" << output->dtype();
    jk << "«R:" << type;
}

#else // JIT
#ifdef JIT_cpu
void CurandRandomOp::jit_run() {
}
#else // JIT_cuda
void CurandRandomOp::jit_run() {
    @define(TT,@if(@strcmp(@T,float32)==0,,Double))

    auto* __restrict__ x = output->ptr<T>();
    // Draw from this op's own generator (an INDEPENDENT curandGenerator_t,
    // lazily created/positioned via RandomGeneratorState::get_cuda_generator())
    // when one was passed in, instead of the single global stream-bound
    // generator every other draw shares.
    curandGenerator_t local_gen = rand_gen ?
        (curandGenerator_t)rand_gen->get_cuda_generator() : curand_bind_stream();
    index_t num = output->num;
    if (num == 0) return;
    // curandGenerateUniform has no parity requirement; curandGenerateNormal
    // wants an even count for pseudorandom generators. The old code rounded
    // the count up for both and wrote one element past the end of the output
    // -- it only stayed out of trouble because the allocator happened to leave
    // slack -- and consumed one extra value from the generator, so two uniform
    // draws of odd length no longer continued the stream that a single draw of
    // the combined length produces.
    //
    // Uniform now asks for exactly num. Normal fills the even prefix in place
    // and takes the last element from a two-element scratch buffer, so nothing
    // is written outside the output. An odd-length normal draw still consumes
    // num+1 values; that is inherent to the even-count requirement.
    //
    // advance_cuda_offset()'s argument counts in the generator's raw stream
    // units, not output values: curandGenerateUniform draws are 1:1 with
    // output values (no pairing); curandGenerateNormal's Box-Muller step
    // consumes 2 output values per raw unit instead (confirmed empirically
    // against the CUDA sample generator). For the odd-length Normal case the
    // first call draws (num-1) values (an even count since num is odd) --
    // (num-1)/2 raw units -- and the tail call draws 2 more values -- 1 more
    // raw unit, continuing immediately after the first call's pair boundary.
    //
    // NOTE: this whole @if(...)/@for(...) block is jittor's own JIT
    // templating syntax, whose argument splitter does not skip `//`
    // comments -- a comma inside a comment nested in here is parsed as a
    // macro-argument separator ("if wrong arguments" / args.size() checks
    // failing at JIT-compile time, not a normal C++ diagnostic). Comments
    // for this block live above it, comma-free, for exactly that reason.
    @if(@strcmp(@R,uniform)==0,
        checkCudaErrors(curandGenerateUniform@TT (local_gen, x, num));
        if (rand_gen) rand_gen->advance_cuda_offset(num);
    ,
        if (num & 1) {
            if (num > 1)
                checkCudaErrors(curandGenerateNormal@TT (local_gen, x, num-1, 0, 1));
            size_t tail_allocation;
            T* tail = (T*)runtime_executor().temp_allocator->alloc(2*sizeof(T), tail_allocation);
            checkCudaErrors(curandGenerateNormal@TT (local_gen, tail, 2, 0, 1));
            checkCudaErrors(cudaMemcpyAsync(x+num-1, tail, sizeof(T),
                cudaMemcpyDeviceToDevice, cudaStreamPerThread));
            runtime_executor().temp_allocator->free(tail, 2*sizeof(T), tail_allocation);
            if (rand_gen) rand_gen->advance_cuda_offset((num - 1) / 2 + 1);
        } else {
            checkCudaErrors(curandGenerateNormal@TT (local_gen, x, num, 0, 1));
            if (rand_gen) rand_gen->advance_cuda_offset(num / 2);
        }
    )
}
#endif // JIT_cpu
#endif // JIT

} // jittor

// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <random>

#include "var.h"
#include "init.h"
#include <cuda_runtime.h>
#include <curand.h>
#include "helper_cuda.h"
#include "curand_random_op.h"
#include "curand_wrapper.h"

namespace jittor {

#ifndef JIT
CurandRandomOp::CurandRandomOp(NanoVector shape, NanoString dtype, NanoString type, RandomGenerator* generator) {
    flags.set(NodeFlags::_cuda, 1);
    output = create_output(shape, dtype);
    this->type = type;
    if (generator) this->rand_gen = generator->state;
    ASSERT(type == ns_normal || type == ns_uniform);
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
    index_t num = output->num;
    // curand doesn't support even number, we add 1 when it is even
    // because allocator will make odd chunks, so this wouldn't cause
    // segmentation fault
    num += num&1;
    // jittor-core-gaps.md §3.6: draw from this op's own generator (an
    // INDEPENDENT curandGenerator_t, lazily created/positioned via
    // RandomGeneratorState::get_cuda_generator()) when one was passed in,
    // instead of the single global `gen` every other draw shares.
    curandGenerator_t local_gen = rand_gen ?
        (curandGenerator_t)rand_gen->get_cuda_generator() : gen;
    @if(@strcmp(@R,uniform)==0,
        checkCudaErrors(curandGenerateUniform@TT (local_gen, x, num));,
        checkCudaErrors(curandGenerateNormal@TT (local_gen, x, num, 0, 1));
    )
    // curandSetGeneratorOffset() counts in the generator's raw stream units,
    // not output values: curandGenerateUniform consumes 1 raw unit per
    // output, but curandGenerateNormal's Box-Muller step consumes 2 output
    // values per raw unit (confirmed empirically against the CUDA sample
    // generator: continuing a stream naturally after N normal outputs lines
    // up with curandSetGeneratorOffset(gen, N/2), not N). `num` is already
    // padded to even above, so this divides exactly. Track the *raw* offset
    // here so a later set_state()'s curandSetGeneratorOffset() lands on
    // exactly the position the next draw would have reached naturally.
    if (rand_gen) rand_gen->advance_cuda_offset(num / @if(@strcmp(@R,uniform)==0,1,2));
}
#endif // JIT_cpu
#endif // JIT

} // jittor
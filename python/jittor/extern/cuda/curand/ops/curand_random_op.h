// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>. 
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"
#include "misc/random_generator.h"

namespace jittor {

struct CurandRandomOp : Op {
    Var* output;
    NanoString type;
    // jittor-core-gaps.md §3.6: see RandomOp::gen (random_op.h) -- same
    // contract, CUDA side. Named rand_gen (not gen) to avoid colliding with
    // the file-scope global `curandGenerator_t gen` declared in
    // curand_wrapper.h and used for the no-generator-passed default path.
    shared_ptr<RandomGeneratorState> rand_gen;
    CurandRandomOp(NanoVector shape, NanoString dtype=ns_float32, NanoString type=ns_uniform, RandomGenerator* generator=nullptr);

    const char* name() const override { return "curand_random"; }
    DECLARE_jit_run;
};

} // jittor
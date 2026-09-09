// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"
#include "misc/random_generator.h"

namespace jittor {

struct RandomOp : Op {
    Var* output;
    NanoString type;
    // jittor-core-gaps.md §3.6: when non-null, draws consume (and advance)
    // THIS generator's own std::mt19937_64 instead of the global one --
    // captured as a shared_ptr (not the raw RandomGenerator* passed in) so
    // the state stays alive for this op's lazy execution regardless of the
    // Python-side Generator wrapper's own lifetime.
    shared_ptr<RandomGeneratorState> gen;
    RandomOp(NanoVector shape, NanoString dtype=ns_float32, NanoString type=ns_uniform, RandomGenerator* generator=nullptr);

    const char* name() const override { return "random"; }
    DECLARE_jit_run;
};

} // jittor
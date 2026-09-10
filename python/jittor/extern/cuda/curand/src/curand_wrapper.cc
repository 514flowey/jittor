// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "curand_wrapper.h"
#include "init.h"
#include "misc/cuda_flags.h"
#include "misc/random_generator.h"

namespace jittor {

curandGenerator_t gen;

struct curand_initer {

inline curand_initer() {
    if (!get_device_count()) return;
    checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
    add_set_seed_callback([](int seed) {
        checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, seed) );
        // curandSetPseudoRandomGeneratorSeed() alone does not reset this
        // already-created generator's internal stream offset (confirmed
        // empirically: reseeding an existing curandGenerator_t to the same
        // seed twice, without this, yields two DIFFERENT draws) -- only a
        // fresh curandCreateGenerator() or an explicit offset reset does.
        // Without this, jt.set_seed(s)/set_global_seed(s) is not actually
        // reproducible on CUDA: re-seeding to a previously-used seed value
        // silently continues the old stream instead of restarting it.
        checkCudaErrors( curandSetGeneratorOffset(gen, 0) );
    });
    LOGv << "curandCreate finished";
}

inline ~curand_initer() {
    if (!get_device_count()) return;
    checkCudaErrors( curandDestroyGenerator(gen) );
    LOGv << "curandDestroy finished";
}

} init_;

// jittor-core-gaps.md §3.6: hooks that let core code (misc/random_generator.cc,
// which cannot depend on <curand.h>/-lcurand) create/reseed/destroy an
// INDEPENDENT curandGenerator_t per jt.Generator(device="cuda") instance --
// distinct from (and never touching) the single global `gen` above, which
// stays exactly as it was for every draw that doesn't pass an explicit
// generator. Registered unconditionally (device_count==0 just means these
// are never invoked, since RandomGeneratorState only calls them for a
// generator explicitly constructed with device="cuda").
static void* curand_gen_create(int64 seed, int64 offset, int device_id) {
    if (device_id >= 0)
        checkCudaErrors( cudaSetDevice(device_id) );
    curandGenerator_t g;
    checkCudaErrors( curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT) );
    checkCudaErrors( curandSetPseudoRandomGeneratorSeed(g, (unsigned long long)seed) );
    if (offset)
        checkCudaErrors( curandSetGeneratorOffset(g, (unsigned long long)offset) );
    return (void*)g;
}

static void curand_gen_destroy(void* g) {
    checkCudaErrors( curandDestroyGenerator((curandGenerator_t)g) );
}

static void curand_gen_set_seed_offset(void* g, int64 seed, int64 offset) {
    curandGenerator_t cg = (curandGenerator_t)g;
    checkCudaErrors( curandSetPseudoRandomGeneratorSeed(cg, (unsigned long long)seed) );
    checkCudaErrors( curandSetGeneratorOffset(cg, (unsigned long long)offset) );
}

struct curand_generator_hook_initer {
inline curand_generator_hook_initer() {
    cuda_gen_create_hook = curand_gen_create;
    cuda_gen_destroy_hook = curand_gen_destroy;
    cuda_gen_set_seed_offset_hook = curand_gen_set_seed_offset;
}
} generator_hook_init_;

} // jittor

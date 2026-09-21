// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "stream_compat.h"
#include "curand_wrapper.h"
#include "runtime/init.h"
#include "runtime/device.h"
#include "runtime/cuda_streams.h"
#include "runtime/random_generator.h"

namespace jittor {

curandGenerator_t gen;
// One generator per device; the global is the current device's. A generator
// draws from the device it was created on, so a single global one would make
// jt.rand() on device 1 either fail or fill device-0 memory.
static vector<curandGenerator_t> gens;
static vector<uint64> curand_stream_binds;
// The last seed, replayed onto a generator created after set_seed so every
// device answers the same seed the same way.
static int curand_last_seed = -1;

static void curand_seed_generator(curandGenerator_t g, int seed) {
    checkCudaErrors( curandSetPseudoRandomGeneratorSeed(g, seed) );
    // The seed alone does not rewind the generator: it keeps its position
    // in the sequence, so re-seeding with the same value after drawing
    // continues from where it left off and jt.set_seed() does not
    // reproduce. set_seed() resets the CPU side's offset for the same
    // reason; this is the CUDA half of it.
    checkCudaErrors( curandSetGeneratorOffset(g, 0) );
}

curandGenerator_t curand_bind_stream() {
    int device = current_device();
    checkCudaErrors(curandSetStream(gen, cuda_compute_stream(device)));
    if ((int)curand_stream_binds.size() <= device)
        curand_stream_binds.resize(device + 1);
    curand_stream_binds[device]++;
    return gen;
}

uint64 curand_stream_bind_count(int device) {
    return device >= 0 && device < (int)curand_stream_binds.size()
        ? curand_stream_binds[device] : 0;
}

static void curand_switch_device(int device) {
    if ((int)gens.size() <= device) gens.resize(device+1, nullptr);
    if (!gens[device]) {
        checkCudaErrors( curandCreateGenerator(&gens[device], CURAND_RNG_PSEUDO_DEFAULT) );
        // Every library handle must agree with jittor's own launches on the
        // stream; see `compute_stream` in backends/cuda/runtime/driver.cc.
        // `cudaStreamPerThread` does not synchronise with the legacy stream,
        // so a handle left on the default would race with no error.
        checkCudaErrors(curandSetStream(gens[device], cudaStreamPerThread));
        if (curand_last_seed >= 0) curand_seed_generator(gens[device], curand_last_seed);
    }
    gen = gens[device];
}

// See cublas_shutdown: report, never raise, and idempotent.
void curand_shutdown() {
    if (gens.empty()) return;
    for (auto g : gens)
        if (g) peekCudaErrorsAlways( curandDestroyGenerator(g) );
    gens.clear();
    curand_stream_binds.clear();
    gen = nullptr;
    LOGv << "curandDestroy finished";
}

struct curand_initer {

inline curand_initer() {
    if (!get_device_count()) return;
    add_device_switch_hook(curand_switch_device);
    add_set_seed_callback([](int seed) {
        curand_last_seed = seed;
        // The callback list is a separate global: nothing orders it against
        // these generators at exit, so a set_seed after shutdown must not run.
        for (auto g : gens)
            if (g) curand_seed_generator(g, seed);
    });
    LOGv << "curandCreate finished";
}

inline ~curand_initer() {
    curand_shutdown();
}

} init_;

// Hooks that let core code (runtime/random_generator.cc, which cannot depend
// on <curand.h>/-lcurand) create/reseed/destroy an INDEPENDENT
// curandGenerator_t per jt.Generator(device="cuda") instance -- distinct
// from (and never touching) the per-device global `gens`/`gen` above, which
// stay exactly as they were for every draw that doesn't pass an explicit
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

// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
// Independent, device-native stateful RNG generators, aligned with
// torch.Generator's contract --
//   - two generators seeded identically produce identical sequences
//   - interleaved draws from different generators never share state
//   - CPU/CUDA sampling happens natively (no NumPy host round-trip),
//     honoring the requested dtype
//   - state is get/set-able for save/restore across runs
//
// Design: each Generator wraps a `shared_ptr<RandomGeneratorState>`. The
// state -- not the pyjt-visible Generator wrapper -- is what RandomOp/
// CurandRandomOp capture (by copying the shared_ptr) at construction time,
// so a lazily-scheduled op keeps the state alive for as long as IT still
// needs it, independent of whatever happens to the Python-side Generator
// object's own refcount in the meantime.
//
// This header must compile in a plain (non-accelerator-extension)
// translation unit -- it must not include <curand.h> or require -lcurand.
// The CUDA side of a generator's state (a curandGenerator_t) is therefore
// stored as an opaque void*, created/destroyed/reseeded only through
// function pointers that backends/cuda/libraries/curand registers at load
// time (the same "optional extension, looked up by string/registered hook,
// never a hard link-time dependency from core code" pattern
// src/ops/composite/random_op.cc already uses for the plain `random` op's
// accelerated-capability lookup).
#pragma once
#include "core/common.h"
#include "core/types.h"
#include <random>

namespace jittor {

// create: allocate a fresh curandGenerator_t seeded to (seed, offset).
// set_seed_offset: re-seed and reposition an EXISTING curandGenerator_t
// (used both by manual_seed(), with offset 0, and by set_state(), with the
// restored offset) via curandSetPseudoRandomGeneratorSeed +
// curandSetGeneratorOffset -- curand exposes no direct state readback, so
// (seed, offset) is the full serializable state of a CUDA generator here,
// the same contract PyTorch's own counter-based CUDA generators use.
typedef void* (*cuda_gen_create_t)(int64 seed, int64 offset, int device_id);
typedef void (*cuda_gen_destroy_t)(void* gen);
typedef void (*cuda_gen_set_seed_offset_t)(void* gen, int64 seed, int64 offset);
// Registered by backends/cuda/libraries/curand/src/curand_wrapper.cc when
// that extension is loaded; remain null (and therefore unused) on a
// CPU-only build or before an accelerator is initialized --
// RandomGeneratorState only calls these when a CUDA device was actually
// requested.
EXTERN_LIB cuda_gen_create_t cuda_gen_create_hook;
EXTERN_LIB cuda_gen_destroy_t cuda_gen_destroy_hook;
EXTERN_LIB cuda_gen_set_seed_offset_t cuda_gen_set_seed_offset_hook;

struct RandomGeneratorState {
    int64 seed_;
    int device_id; // -1 = cpu, >=0 = cuda device ordinal
    std::default_random_engine cpu_engine;
    // Opaque curandGenerator_t, lazily created on first CUDA draw via
    // cuda_gen_create_hook; owned by this state, destroyed in ~RandomGeneratorState.
    void* cuda_gen = nullptr;
    // Number of curand output values drawn so far through this generator --
    // tracked purely so get_state()/set_state() can round-trip CUDA state
    // via curandSetGeneratorOffset (curand exposes no direct state readback).
    int64 cuda_offset = 0;

    RandomGeneratorState(int64 seed, int device_id);
    ~RandomGeneratorState();
    void set_seed(int64 seed);
    // Lazily creates (if needed) and returns the curandGenerator_t for this
    // state, as an opaque void*; only meaningful when device_id>=0 and CUDA
    // is available. advance_cuda_offset must be called by the caller after
    // each draw so get_state()/set_state() stay accurate.
    void* get_cuda_generator();
    void advance_cuda_offset(int64 n);

    // Serializes cpu_engine's full state (via the standard <random> stream
    // operators) plus seed/device_id/cuda_offset into an opaque byte string.
    string get_state();
    void set_state(const string& s);
};

// @pyjt(Generator)
// @attrs(heaptype)
struct RandomGenerator {
    shared_ptr<RandomGeneratorState> state;
    string device_str;

    // @pyjt(__init__)
    RandomGenerator(const string& device="cpu");

    // @pyjt(manual_seed)
    // @attrs(return_self)
    RandomGenerator* manual_seed(int64 seed);

    // torch.Generator.seed(): reseed from fresh (unpredictable) entropy and
    // return the new seed.
    // @pyjt(seed)
    int64 reseed();

    // @pyjt(initial_seed)
    int64 initial_seed();

    // @pyjt(get_state)
    string get_state();

    // @pyjt(set_state)
    // @attrs(return_self)
    RandomGenerator* set_state(const string& s);

    // @pyjt(__get__device)
    string get_device();
};

} // jittor

// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include <chrono>
#include "runtime/random_generator.h"

namespace jittor {

cuda_gen_create_t cuda_gen_create_hook = nullptr;
cuda_gen_destroy_t cuda_gen_destroy_hook = nullptr;
cuda_gen_set_seed_offset_t cuda_gen_set_seed_offset_hook = nullptr;

RandomGeneratorState::RandomGeneratorState(int64 seed, int device_id)
    : seed_(seed), device_id(device_id), cpu_engine(seed) {}

RandomGeneratorState::~RandomGeneratorState() {
    if (cuda_gen && cuda_gen_destroy_hook)
        cuda_gen_destroy_hook(cuda_gen);
}

void RandomGeneratorState::set_seed(int64 seed) {
    seed_ = seed;
    cpu_engine.seed((uint64)seed);
    cuda_offset = 0;
    if (cuda_gen)
        cuda_gen_set_seed_offset_hook(cuda_gen, seed_, 0);
}

void* RandomGeneratorState::get_cuda_generator() {
    if (!cuda_gen) {
        ASSERT(cuda_gen_create_hook) << "CUDA random generator requested but "
            "the curand extension is not loaded (CUDA not available in this build).";
        // cuda_offset may already be non-zero here if this state was
        // restored via set_state() before its first CUDA use -- create()
        // seeds it directly to (seed_, cuda_offset) so the deferred
        // generator picks up exactly where the saved state left off.
        cuda_gen = cuda_gen_create_hook(seed_, cuda_offset, device_id);
    }
    return cuda_gen;
}

void RandomGeneratorState::advance_cuda_offset(int64 n) {
    cuda_offset += n;
}

string RandomGeneratorState::get_state() {
    std::ostringstream oss;
    oss << seed_ << ' ' << device_id << ' ' << cuda_offset << ' ' << cpu_engine;
    return oss.str();
}

void RandomGeneratorState::set_state(const string& s) {
    std::istringstream iss(s);
    int64 new_seed, new_offset;
    int new_device;
    // std::default_random_engine's own operator>> does not skip leading
    // whitespace (unlike operator>> for the built-in numeric types before
    // it) -- without the explicit std::ws here it tries to parse starting
    // at the separating space itself and silently fails every time.
    iss >> new_seed >> new_device >> new_offset >> std::ws >> cpu_engine;
    ASSERT(!iss.fail()) << "Generator.set_state(): malformed state string "
        "(expected the byte string previously returned by get_state()).";
    seed_ = new_seed;
    device_id = new_device;
    cuda_offset = new_offset;
    if (cuda_gen)
        // Re-seed the already-created generator and jump it directly to the
        // restored offset (curandSetGeneratorOffset), continuing exactly
        // where the saved state left off.
        cuda_gen_set_seed_offset_hook(cuda_gen, seed_, cuda_offset);
}

static int parse_device(const string& device) {
    if (device == "cpu") return -1;
    if (device.substr(0, 4) == "cuda") {
        if (device.size() > 5 && device[4] == ':')
            return std::stoi(device.substr(5));
        return 0;
    }
    LOGf << "Generator: unknown device string" << device << "(expected \"cpu\", \"cuda\", or \"cuda:N\")";
    return -1;
}

static int64 fresh_seed() {
    // matches the intent of torch.Generator().seed(): unpredictable,
    // process/time-derived, not required to be cryptographically secure.
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::random_device rd;
    uint64 a = (uint64)now, b;
    try { b = ((uint64)rd() << 32) ^ (uint64)rd(); }
    catch (...) { b = 0; }
    return (int64)(a ^ b ^ (a << 21) ^ (b >> 13));
}

RandomGenerator::RandomGenerator(const string& device) : device_str(device) {
    int device_id = parse_device(device);
    state = std::make_shared<RandomGeneratorState>(fresh_seed(), device_id);
}

RandomGenerator* RandomGenerator::manual_seed(int64 seed) {
    state->set_seed(seed);
    return this;
}

int64 RandomGenerator::reseed() {
    int64 s = fresh_seed();
    state->set_seed(s);
    return s;
}

int64 RandomGenerator::initial_seed() {
    return state->seed_;
}

string RandomGenerator::get_state() {
    return state->get_state();
}

RandomGenerator* RandomGenerator::set_state(const string& s) {
    state->set_state(s);
    return this;
}

string RandomGenerator::get_device() {
    return device_str;
}

} // jittor

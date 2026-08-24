// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

namespace jittor {

#ifdef HAS_CUDA

// RAII helper: switch the process' current CUDA device for the lifetime of
// this guard, restoring the previous device on destruction.
// device_id < 0 means "don't care / ambient" and is a pure no-op, so existing
// single-GPU call sites that never pass an explicit device id are unaffected.
struct CudaDeviceGuard {
    int prev_device = -1;
    bool active = false;

    inline explicit CudaDeviceGuard(int device_id) {
        if (device_id < 0) return;
        cudaGetDevice(&prev_device);
        if (prev_device != device_id) {
            cudaSetDevice(device_id);
            active = true;
        }
    }
    inline ~CudaDeviceGuard() {
        if (active) cudaSetDevice(prev_device);
    }
    CudaDeviceGuard(const CudaDeviceGuard&) = delete;
    CudaDeviceGuard& operator=(const CudaDeviceGuard&) = delete;
};

#else

struct CudaDeviceGuard {
    inline explicit CudaDeviceGuard(int) {}
};

#endif

} // jittor

// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#ifdef HAS_CUDA
#include "mem/allocator.h"

namespace jittor {

struct CudaDeviceAllocator : Allocator {
    // -1 = ambient/unbound: preserves the legacy behavior of the global
    // singleton below (no explicit cudaSetDevice around alloc/free, relies
    // entirely on the process' current CUDA context). >=0 = this instance
    // is pinned to that physical GPU and guards every alloc/free with it.
    int dev_id = -1;
    inline CudaDeviceAllocator() {}
    inline explicit CudaDeviceAllocator(int dev_id) : dev_id(dev_id) {}
    uint64 flags() const override { return _cuda; }
    // dev_id>=0 (per-device pool instance): return it directly.
    // dev_id==-1 (legacy ambient singleton): fall back to querying the
    // process' actual current CUDA device, so device_id() is still real
    // ground truth for Vars allocated via the unpinned default path.
    int device_id() const override;
    const char* name() const override;
    void* alloc(size_t size, size_t& allocation) override;
    void free(void* mem_ptr, size_t size, const size_t& allocation) override;
};

EXTERN_LIB CudaDeviceAllocator cuda_device_allocator;
// Lazily-constructed, process-lifetime pool of per-device allocators, one
// per physical GPU index actually requested. Backed by a static map inside
// the .cc so callers never need to manage ownership.
CudaDeviceAllocator& get_cuda_device_allocator(int device_id);

}

#endif

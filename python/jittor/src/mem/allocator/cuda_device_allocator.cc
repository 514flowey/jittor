// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#ifdef HAS_CUDA
#include <stdexcept>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <cuda_runtime.h>
#include "mem/mem_info.h"
#include "helper_cuda.h"
#include "misc/cuda_guard.h"
#include "mem/allocator/cuda_device_allocator.h"

namespace jittor {

CudaDeviceAllocator cuda_device_allocator;

CudaDeviceAllocator& get_cuda_device_allocator(int device_id) {
    // Jittor supports multi-threaded op dispatch (executor.cc's use_threading),
    // so concurrent first-touch of a device id must not race on the map's
    // internal structure. Once inserted, the returned reference stays valid
    // even across later rehashes (the pool stores unique_ptr, so the pointee
    // never moves), so the lock only needs to guard find/insert itself.
    static std::mutex pool_mutex;
    static std::unordered_map<int, std::unique_ptr<CudaDeviceAllocator>> pool;
    std::lock_guard<std::mutex> lock(pool_mutex);
    auto iter = pool.find(device_id);
    if (iter != pool.end()) return *iter->second;
    auto* p = new CudaDeviceAllocator(device_id);
    pool[device_id] = std::unique_ptr<CudaDeviceAllocator>(p);
    return *p;
}
EXTERN_LIB bool no_cuda_error_when_free;
DEFINE_FLAG(int, cuda_device_allocator_managed_fallback, 0,
    "Fallback to cudaMallocManaged after cudaMalloc OOM. Disabled by default so "
    "higher-level caching allocators can release cached blocks and retry.");

const char* CudaDeviceAllocator::name() const {return "cuda_device";}

int CudaDeviceAllocator::device_id() const {
    if (dev_id >= 0) return dev_id;
    int cur = 0;
    cudaGetDevice(&cur);
    return cur;
}

void* CudaDeviceAllocator::alloc(size_t size, size_t& allocation) {
    if (size==0) return (void*)0x10;
    CudaDeviceGuard guard(dev_id);
    void* ptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err == cudaSuccess)
        return ptr;
    // Clean the sticky runtime error before a higher-level allocator retries.
    cudaGetLastError();
    if (!cuda_device_allocator_managed_fallback)
        throw std::runtime_error("cudaMalloc failed");
    display_memory_info(__FILELINE__);
    LOGf << "Unable to alloc cuda device memory for size" << size;
    checkCudaErrors(cudaMallocManaged(&ptr, size));
    return ptr;
}

void CudaDeviceAllocator::free(void* mem_ptr, size_t size, const size_t& allocation) {
    if (size==0) return;
    if (no_cuda_error_when_free) return;
    CudaDeviceGuard guard(dev_id);
    checkCudaErrors(cudaFree(mem_ptr));
}

} // jittor

#endif

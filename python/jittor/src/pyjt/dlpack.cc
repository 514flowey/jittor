// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
// DLPack ownership/device/stream semantics (jittor-core-gaps.md section
// 3.7). Design summary (see the design doc for the full rationale):
//
//  - Export (Var -> DLManagedTensor capsule): sync the Var (forces any
//    pending CUDA work to complete, matching device_raw_ptr()'s existing
//    pattern -- Jittor has no non-default-stream story, so a full device
//    sync IS the producer-side "stream" guarantee), read its real
//    device/dtype/shape via the already-completed per-tensor device id
//    infrastructure (VarHolder::location()/device_id()), and stash a VarPtr
//    in manager_ctx so the underlying storage outlives the capsule even if
//    the Python VarHolder is immediately dropped.
//  - Import (capsule -> Var): reuse the exact "hand-built Var over foreign
//    memory" pattern already proven by reuse_np_array() (py_array_op.cc),
//    but route CUDA-sourced tensors through a NEW per-device ForeignAllocator
//    (foreign_allocator.h) instead of the plain CPU-only one, so
//    location()/device_id() report the truth instead of silently lying that
//    GPU memory is CPU memory. The imported Var is explicitly device-pinned
//    (set_device_pin) so later ops route to the correct physical GPU without
//    depending on the ambient global flag.
//  - Ownership: PyCapsule's own single-owner semantics do the "consume
//    exactly once" enforcement (rename "dltensor"->"used_dltensor" on
//    import; a capsule destructor fires the same deleter exactly once if
//    the capsule is garbage-collected without ever being imported).
//  - Contiguity: Jittor's Var/Allocation have no stride concept at all (only
//    ever row-major contiguous) -- importing a DLTensor with non-null,
//    non-contiguous strides is rejected outright rather than silently
//    misread or silently copied.
#include <Python.h>
#include <vector>
#include "pyjt/dlpack.h"
#include "pyjt/dlpack_api.h"
#include "var_holder.h"
#include "var.h"
#include "mem/allocator.h"
#include "mem/allocator/foreign_allocator.h"
#include "misc/cuda_flags.h"
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "misc/cuda_guard.h"
#endif

namespace jittor {

// ---------------------------------------------------------------------
// dtype <-> DLDataType
// ---------------------------------------------------------------------

static DLDataType ns_to_dltype(NanoString ns) {
    DLDataType dt;
    dt.lanes = 1;
    dt.bits = (uint8_t)(ns.dsize() * 8);
    // is_complex()/is_floating_point() both read plain bit flags that don't
    // distinguish bfloat16 from the other floats, nor complex from real --
    // order matters here, and bfloat16 must be checked before the generic
    // is_floating_point() fallback.
    if (ns.is_complex()) dt.code = kDLComplex;
    else if (ns == ns_bfloat16) dt.code = kDLBfloat;
    else if (ns.is_bool()) dt.code = kDLBool;
    else if (ns.is_unsigned()) dt.code = kDLUInt;
    else if (ns.is_int()) dt.code = kDLInt;
    else if (ns.is_floating_point()) dt.code = kDLFloat;
    else {
        LOGf << "dlpack: unsupported dtype for export:" << ns;
        dt.code = 0; dt.bits = 0;
    }
    return dt;
}

static NanoString dltype_to_ns(DLDataType dt) {
    if (dt.lanes != 1)
        LOGf << "dlpack: vector dtypes (lanes=" << dt.lanes << ") are not supported";
    switch (dt.code) {
        case kDLBool:
            if (dt.bits == 8) return ns_bool;
            break;
        case kDLInt:
            if (dt.bits == 8) return ns_int8;
            if (dt.bits == 16) return ns_int16;
            if (dt.bits == 32) return ns_int32;
            if (dt.bits == 64) return ns_int64;
            break;
        case kDLUInt:
            if (dt.bits == 8) return ns_uint8;
            if (dt.bits == 16) return ns_uint16;
            if (dt.bits == 32) return ns_uint32;
            if (dt.bits == 64) return ns_uint64;
            break;
        case kDLFloat:
            if (dt.bits == 16) return ns_float16;
            if (dt.bits == 32) return ns_float32;
            if (dt.bits == 64) return ns_float64;
            break;
        case kDLBfloat:
            if (dt.bits == 16) return ns_bfloat16;
            break;
        case kDLComplex:
            if (dt.bits == 64) return ns_complex64;
            if (dt.bits == 128) return ns_complex128;
            break;
    }
    LOGf << "dlpack: unsupported DLDataType(code=" << (int)dt.code
         << ", bits=" << (int)dt.bits << ", lanes=" << dt.lanes << ") for import";
    return ns_void;
}

// ---------------------------------------------------------------------
// device <-> DLDevice
// ---------------------------------------------------------------------

static DLDevice var_to_dldevice(VarHolder* v) {
    DLDevice dev{kDLCPU, 0};
    string loc = v->location();
    if (loc == "none") {
        // Not yet materialized: force computation (this is also where the
        // "producer has finished writing" sync guarantee for the CUDA case
        // comes from -- see to_dlpack_capsule).
        v->sync(true, false);
        loc = v->location();
    }
    if (loc == "cpu") { dev.device_type = kDLCPU; dev.device_id = 0; return dev; }
    if (loc == "device") { dev.device_type = kDLCUDA; dev.device_id = v->device_id(); return dev; }
    LOGf << "dlpack: cannot export a Var with location" << loc
         << "(swapped-to-disk Vars are not supported)";
    return dev;
}

// ---------------------------------------------------------------------
// export: Var -> DLManagedTensor capsule
// ---------------------------------------------------------------------

struct JittorDLManagerCtx {
    VarPtr vp;                    // keeps the underlying storage alive
    std::vector<int64_t> shape;   // DLTensor.shape points into this
    JittorDLManagerCtx(Var* v) : vp(v) {}
};

static void jittor_dlpack_deleter(DLManagedTensor* self) {
    if (!self) return;
    delete (JittorDLManagerCtx*)self->manager_ctx;
    delete self;
}

// Fires if the capsule is garbage-collected without ever being consumed by
// from_dlpack_capsule (which renames it to "used_dltensor" first) -- the
// standard DLPack idiom (used by PyTorch/NumPy) for "capsule created but
// nobody imported it" cleanup, and the mechanism that makes "deleter called
// exactly once, even if never consumed" true.
static void jittor_dlpack_capsule_destructor(PyObject* capsule) {
    if (!PyCapsule_IsValid(capsule, "dltensor")) return;
    auto* managed = (DLManagedTensor*)PyCapsule_GetPointer(capsule, "dltensor");
    if (managed && managed->deleter) managed->deleter(managed);
}

static PyObject* to_dlpack_capsule(VarHolder* v) {
    // Forces pending computation AND a device sync -- the producer-side half
    // of the "stream" contract: by the time this returns, every earlier op
    // Jittor queued against this Var (on its one and only implicit stream)
    // is guaranteed complete, so the consumer can safely read the pointer
    // with no further synchronization needed.
    v->sync(true, false);
    Var* var = v->var;
    ASSERT(var->mem_ptr || var->num == 0) << "dlpack: exporting an unmaterialized Var";

    // Resolve device/dtype BEFORE any heap allocation below: both can throw
    // (unsupported dtype, swapped-to-disk location), and if that happened
    // after ctx/managed were already allocated, ctx's VarPtr would keep the
    // underlying storage pinned forever with nothing left to free it.
    DLDevice dev = var_to_dldevice(v);
    DLDataType dtype = ns_to_dltype(var->dtype());

    auto* ctx = new JittorDLManagerCtx(var);
    ctx->shape.reserve(var->shape.size());
    for (int i = 0; i < var->shape.size(); i++)
        ctx->shape.push_back(var->shape[i]);

    auto* managed = new DLManagedTensor();
    managed->manager_ctx = ctx;
    managed->deleter = jittor_dlpack_deleter;
    DLTensor& t = managed->dl_tensor;
    t.data = var->mem_ptr;
    t.device = dev;
    t.ndim = (int32_t)ctx->shape.size();
    t.dtype = dtype;
    t.shape = ctx->shape.empty() ? nullptr : ctx->shape.data();
    t.strides = nullptr;   // Jittor Vars are always row-major contiguous
    t.byte_offset = 0;

    PyObject* capsule = PyCapsule_New(managed, "dltensor", jittor_dlpack_capsule_destructor);
    if (!capsule) {
        jittor_dlpack_deleter(managed);
        return nullptr;
    }
    return capsule;
}

PyObject* to_dlpack(VarHolder* v) {
    return to_dlpack_capsule(v);
}

PyObject* VarHolder::dlpack(PyObject* stream, PyObject* max_version, PyObject* dl_device, PyObject* copy) {
    // `stream` (an int stream handle, or None) is accepted for protocol
    // compatibility with numpy/cupy/torch callers but not otherwise used:
    // to_dlpack_capsule()'s sync(true,false) below already conservatively
    // waits for all of Jittor's own (single, default-stream) work to finish
    // regardless of what stream the consumer intends to use next -- see the
    // design doc for why this is a sound simplification given Jittor's
    // execution model, not a shortcut around a real cross-stream hazard.
    // `max_version` is accepted and ignored: this adapter only ever produces
    // the classic unversioned DLManagedTensor, which is DLPack's own
    // mandated fallback for a consumer that didn't get a version match (and
    // is what NumPy/CuPy actually consume by default -- verified against
    // this repo's dev environment).
    (void)stream; (void)max_version;
    if (copy && copy != Py_None && PyObject_IsTrue(copy))
        LOGf << "dlpack: __dlpack__(copy=True) is not supported -- Jittor's "
             << "DLPack export is always zero-copy";
    if (dl_device && dl_device != Py_None) {
        int req_type = 0, req_id = 0;
        if (!PyArg_ParseTuple(dl_device, "ii", &req_type, &req_id))
            LOGf << "dlpack: __dlpack__(dl_device=...) must be a (device_type, device_id) tuple";
        DLDevice actual = var_to_dldevice(this);
        if (req_type != (int)actual.device_type || req_id != actual.device_id)
            LOGf << "dlpack: __dlpack__(dl_device=...) requesting a device other than "
                 << "where this Var actually lives is not supported; "
                 << "call .migrate_to_device() explicitly first";
    }
    return to_dlpack_capsule(this);
}

PyObject* VarHolder::dlpack_device() {
    DLDevice dev = var_to_dldevice(this);
    return Py_BuildValue("(ii)", (int)dev.device_type, (int)dev.device_id);
}

// ---------------------------------------------------------------------
// import: DLManagedTensor capsule -> Var
// ---------------------------------------------------------------------

static bool dl_tensor_is_contiguous(const DLTensor& t) {
    if (t.strides == nullptr) return true;
    int64_t expect = 1;
    for (int32_t i = t.ndim - 1; i >= 0; i--) {
        if (t.shape[i] != 1 && t.strides[i] != expect)
            return false;
        expect *= t.shape[i];
    }
    return true;
}

VarHolder* from_dlpack_capsule(PyObject* capsule) {
    if (!PyCapsule_IsValid(capsule, "dltensor"))
        LOGf << "dlpack: expected a live \"dltensor\" capsule (it may already "
             << "have been consumed by an earlier from_dlpack() call -- a "
             << "DLPack capsule can only be imported once)";
    auto* managed = (DLManagedTensor*)PyCapsule_GetPointer(capsule, "dltensor");

    DLTensor& t = managed->dl_tensor;
    if (!dl_tensor_is_contiguous(t))
        LOGf << "dlpack: importing a non-contiguous (strided) tensor is not "
             << "supported -- Jittor's Var has no stride concept, only "
             << "row-major contiguous storage";
    if (t.device.device_type != kDLCPU && t.device.device_type != kDLCUDA)
        LOGf << "dlpack: unsupported DLDevice.device_type=" << (int)t.device.device_type
             << " (only kDLCPU and kDLCUDA are supported)";
#ifndef HAS_CUDA
    if (t.device.device_type == kDLCUDA)
        LOGf << "dlpack: importing a CUDA tensor requires a CUDA-enabled build";
#endif
    NanoString dtype = dltype_to_ns(t.dtype);
    NanoVector shape;
    if (t.ndim > 0)
        shape = NanoVector::make(t.shape, t.ndim);
    else
        shape.clear();

    // Everything that can fail (contiguity, device type, dtype) has already
    // succeeded -- only now do we commit to consuming this capsule. Renaming
    // any earlier would mean a validation failure above leaves the capsule
    // looking "consumed" (jittor_dlpack_capsule_destructor no-ops on it)
    // while nothing ever took ownership of the producer's buffer, permanently
    // leaking it (real GPU memory, for a CUDA producer).
    PyCapsule_SetName(capsule, "used_dltensor");

    VarPtr vp(shape, dtype);
    vp->finish_pending_liveness();
    vp->mem_ptr = (char*)t.data + t.byte_offset;

    Allocation allocation;
    int target_device_id = -1;
    if (t.device.device_type == kDLCPU) {
        make_foreign_allocation(allocation, vp->mem_ptr, vp->size,
            [managed]() { managed->deleter(managed); });
    }
#ifdef HAS_CUDA
    else {
        target_device_id = t.device.device_id;
        {
            // Consumer-side half of the simplified stream contract: make
            // sure any work the producer queued on ITS device is visible
            // before Jittor (which only ever reads/writes via the default
            // stream) touches the memory.
            CudaDeviceGuard guard(target_device_id);
            checkCudaErrors(cudaDeviceSynchronize());
        }
        make_foreign_allocation(allocation, vp->mem_ptr, vp->size,
            [managed]() { managed->deleter(managed); },
            &get_foreign_allocator(target_device_id));
    }
#endif
    vp->allocator = allocation.allocator;
    vp->allocation = allocation.allocation;
    allocation.ptr = nullptr;
    allocation.allocator = nullptr;
    allocation.allocation = 0;

    if (target_device_id >= 0) {
        vp->set_device_pin(target_device_id);
        // Mirror VarHolder::migrate_to_device_()'s pairing of device_pin with
        // _stop_fuse: without this, the fuser could silently merge this Var's
        // consumer op with a neighbor targeting a different device.
        vp->flags.set(NodeFlags::_stop_fuse);
    }

    return new VarHolder(std::move(vp));
}

} // jittor

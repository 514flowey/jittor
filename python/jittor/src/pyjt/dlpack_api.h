// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
// DLPack (jittor-core-gaps.md section 3.7) module-level free functions,
// mirroring torch.utils.dlpack.to_dlpack(tensor)/from_dlpack(capsule)'s
// calling convention. Kept in a small standalone header (rather than
// declared inline in dlpack.cc) because pyjt_compiler.py's `compile()` only
// scans src/**/*.h for `@pyjt(...)` annotations, never .cc files -- see
// var_holder.h/VarHolder::__dlpack__ for the per-tensor method form.
#pragma once
#include "common.h"
#include "var_holder.h"

namespace jittor {

// @pyjt(to_dlpack)
PyObject* to_dlpack(VarHolder* v);

// @pyjt(from_dlpack_capsule)
VarHolder* from_dlpack_capsule(PyObject* capsule);

} // jittor

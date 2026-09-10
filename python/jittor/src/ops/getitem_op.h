// ***************************************************************
// Copyright (c) 2023 Jittor.  All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"
#include "var_slices.h"
#include "misc/stack_vector.h"

namespace jittor {

struct GetitemOp : Op {
    static constexpr jittor::NanoString::Flags _inplace = (jittor::NanoString::Flags)0;
    // Cached copy of the primary input pointer, populated once at construction.
    // getitem/setitem are unusual among ops in still dereferencing their input
    // at jit_prepare/jit_run time via this pointer (most ops instead read a
    // pre-baked shape/dtype and never touch the Var* again after infer_shape).
    // Do NOT replace this with inputs().front(): the generic executor's
    // per-op input-edge bookkeeping (_inputs) is allowed to be reset once an
    // op is considered "finished", but this op can legitimately be revisited
    // for jit_prepare afterwards (e.g. its output feeds a later, separate
    // sync that shares this op with an earlier one) -- inputs().front() would
    // then read an empty list and crash. See getitem_op.cc jit_prepare.
    //
    // `y` is unused by GetitemOp itself but MUST stay declared here, right
    // after `x`: SetitemOp::infer_shape()/compile_optimize() reinterpret a
    // SetitemOp* as a GetitemOp* to reuse infer_slices()/_compile_optimize()
    // (see setitem_op.cc), which only works if both structs agree on the
    // offset of every field that reinterpreted code touches (vs, i_to_vs,
    // i_to_o, o_shape, first_oid_of_var, var_dim). Dropping this member (or
    // reordering it after vs) desyncs the two layouts and makes that cast
    // read vs/o_shape from the wrong offset -- garbage shapes, not a crash.
    Var *x, *y;
    VarSlices vs;
    // map i to related var slice
    NanoVector i_to_vs;
    // map i to related o
    NanoVector i_to_o;
    NanoVector o_shape;
    int first_oid_of_var, var_dim;

    GetitemOp(Var* x, VarSlices&& slices);
    // @attrs(multiple_outputs)
    GetitemOp(Var* x, VarSlices&& slices, int _);
    
    const char* name() const override { return "getitem"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void grads(Var** dout, VarPtr* dins) override;
    void infer_shape() override;
    void compile_optimize(string& src) override;
    void graph_optimize() override;
    DECLARE_jit_run;

    void infer_slices(
        StackVector<>& __restrict__ i_to_vs, 
        StackVector<>& __restrict__ i_to_o,
        StackVector<>& __restrict__ out_shape
    );
    void _compile_optimize(string& src);
};

void cuda_loop_schedule(NanoVector o_shape, int* masks, int* tdims);

} // jittor

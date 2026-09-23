# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Jittor Group
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# SGD momentum/dampening (jittor-core-gaps.md §3.3): the momentum buffer
# must be seeded from the raw gradient on its first touch, and dampening
# must have zero effect when momentum is off. Cross-checked here against an
# independent, real PyTorch process (not this repo's own torch-compat
# shim, which is jittor's own implementation under another name and so
# cannot serve as an independent reference for a jittor bug).
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import jittor as jt


def _real_torch_python():
    """Locate a Python interpreter with real (non-shim) PyTorch installed,
    isolated from this repo's own jittor/jittor-torch. Mirrors
    tests/linalg/test_complex_spectral_gradients.py's helper of the same
    name/contract."""
    candidates = []
    override = os.environ.get("JITTOR_TEST_REAL_TORCH_PYTHON")
    if override:
        candidates.append(override)
    candidates.append("/opt/real-torch-venv/bin/python3")
    for python in candidates:
        if not Path(python).exists():
            continue
        probe = subprocess.run(
            [python, "-c", "import torch; assert not hasattr(torch, 'compat')"],
            capture_output=True, timeout=60)
        if probe.returncode == 0:
            return python
    raise RuntimeError(
        "no interpreter with real (non-shim) PyTorch found; set "
        "JITTOR_TEST_REAL_TORCH_PYTHON to one, or skip this test")


def _torch_sgd_three_steps(p0, g0, lr, momentum, dampening):
    """Independent oracle: three real torch.optim.SGD steps with a FIXED
    (repeated) gradient every step, matching this repo's own repro
    convention (a stand-in for a real per-step loss/backward)."""
    python = _real_torch_python()
    script = f"""
import torch, json
p = torch.tensor([{p0}], requires_grad=True)
opt = torch.optim.SGD([p], lr={lr}, momentum={momentum}, dampening={dampening})
vals = []
for _ in range(3):
    opt.zero_grad()
    p.grad = torch.tensor([{g0}])
    opt.step()
    vals.append(float(p.detach()[0]))
print(json.dumps(vals))
"""
    # 180s, not 60: this file's own tests run several real-torch child
    # processes back to back while jittor's own CUDA JIT compiles may still
    # be finishing in the same container, and a plain `torch.optim.SGD`
    # subprocess occasionally queues behind that -- seen directly (13s in
    # isolation, timed out at 60s when run after two CUDA-compiling
    # siblings in the same file).
    out = subprocess.run([python, "-c", script], check=True,
                          capture_output=True, timeout=180, text=True)
    return json.loads(out.stdout)


class TestSGDDampeningIndependentOracle(unittest.TestCase):
    def setUp(self):
        try:
            _real_torch_python()
        except RuntimeError as e:
            self.skipTest(str(e))

    def _check(self, momentum, dev):
        p0, g0, lr, damp = 1.0, 2.0, 0.1, 0.2
        expect = _torch_sgd_three_steps(p0, g0, lr, momentum, damp)

        p = jt.array([p0])
        opt = jt.optim.SGD([p], lr=lr, momentum=momentum, dampening=damp)
        got = []
        for _ in range(3):
            g = jt.array([g0])
            loss = (p * g).sum()
            opt.step(loss)
            got.append(float(p.numpy()[0]))
        np.testing.assert_allclose(got, expect, atol=1e-5, rtol=1e-5,
                                    err_msg=f"SGD dampening vs real PyTorch, "
                                            f"momentum={momentum} [{dev}]")

    def test_momentum_zero_matches_real_pytorch(self):
        for dev in ("cpu", "cuda"):
            if dev == "cuda" and not jt.has_cuda:
                continue
            with jt.flag_scope(use_cuda=(dev == "cuda")):
                self._check(0.0, dev)

    def test_momentum_nonzero_matches_real_pytorch(self):
        for dev in ("cpu", "cuda"):
            if dev == "cuda" and not jt.has_cuda:
                continue
            with jt.flag_scope(use_cuda=(dev == "cuda")):
                self._check(0.8, dev)

    def test_cuda_fused_matches_real_pytorch(self):
        if not jt.has_cuda:
            self.skipTest("no CUDA found")
        p0, g0, lr, damp = 1.0, 2.0, 0.1, 0.2
        with jt.flag_scope(use_cuda=1):
            for momentum in (0.0, 0.8):
                expect = _torch_sgd_three_steps(p0, g0, lr, momentum, damp)
                p = jt.array([p0])
                opt = jt.optim.SGD([p], lr=lr, momentum=momentum, dampening=damp,
                                   fused=True)
                got = []
                for _ in range(3):
                    g = jt.array([g0])
                    loss = (p * g).sum()
                    opt.step(loss)
                    got.append(float(p.numpy()[0]))
                np.testing.assert_allclose(
                    got, expect, atol=1e-5, rtol=1e-5,
                    err_msg=f"CUDA fused SGD dampening vs real PyTorch, "
                            f"momentum={momentum}")


if __name__ == "__main__":
    unittest.main()

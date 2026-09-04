import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from unittest import mock


class TestTorchBootstrap(unittest.TestCase):
    def test_gaussian_runtime_caches_lpips_criterion(self):
        from jittor.torch_shim.gaussian_splatting_runtime import _patch_lpips_module

        constructed = []

        class Criterion:
            def __init__(self, net_type, version):
                constructed.append((net_type, version))

            def to(self, device):
                self.device = device
                return self

            def __call__(self, x, y):
                return (self.device, x.value, y.value)

        module = types.SimpleNamespace(lpips=lambda *args, **kwargs: None, LPIPS=Criterion)
        self.assertTrue(_patch_lpips_module(module))
        x = types.SimpleNamespace(device="cuda:0", value=1)
        y = types.SimpleNamespace(device="cuda:0", value=2)
        self.assertEqual(module.lpips(x, y, net_type="vgg"), ("cuda:0", 1, 2))
        self.assertEqual(module.lpips(x, y, net_type="vgg"), ("cuda:0", 1, 2))
        self.assertEqual(constructed, [("vgg", "0.1")])
        self.assertFalse(_patch_lpips_module(module))

    def test_scan_torch_extension_setup(self):
        from jittor.torch_shim import scan_extension_dirs

        with tempfile.TemporaryDirectory(dir=os.getcwd()) as d:
            root = os.path.join(d, "pkg")
            os.makedirs(root)
            with open(os.path.join(root, "setup.py"), "w") as f:
                f.write(textwrap.dedent("""
                    from setuptools import setup
                    from torch.utils.cpp_extension import CUDAExtension, BuildExtension
                    setup(
                        name="pkg",
                        ext_modules=[
                            CUDAExtension(name="pkg._C", sources=["kernel.cu", "ext.cpp"])
                        ],
                        cmdclass={"build_ext": BuildExtension},
                    )
                """))
            for name in ("kernel.cu", "ext.cpp"):
                open(os.path.join(root, name), "w").close()

            exts = scan_extension_dirs(project_root=d)
            self.assertEqual([e.root for e in exts], [root])
            self.assertEqual(len(exts[0].sources), 2)
            self.assertTrue(exts[0].setup_py.endswith("setup.py"))

    def test_entry_script_runtime_defaults_to_project_dir(self):
        import jittor as jt

        with tempfile.TemporaryDirectory(dir=os.getcwd()) as d:
            entry = os.path.join(d, "train.py")
            old_argv0 = jt._sys.argv[0]
            try:
                for source in (
                    "from jittor.torch_shim import enable\n",
                    "import jittor as torch\n",
                    "import jiitor as torch\n",
                ):
                    with self.subTest(source=source.strip()):
                        with open(entry, "w") as f:
                            f.write(source)
                        jt._sys.argv[0] = entry
                        self.assertEqual(
                            jt._jt_torch_entry_runtime_root(),
                            os.path.join(d, ".cache", "jittor_torch"),
                        )
            finally:
                jt._sys.argv[0] = old_argv0

    def test_jiitor_entry_bootstraps_in_subprocess(self):
        with tempfile.TemporaryDirectory() as d:
            entry = os.path.join(d, "probe.py")
            os.makedirs(os.path.join(d, "scene"))
            os.makedirs(os.path.join(d, "gaussian_renderer"))
            for marker in (
                "train.py", "scene/gaussian_model.py", "gaussian_renderer/__init__.py",
            ):
                pathlib.Path(d, marker).touch()
            with open(entry, "w") as f:
                f.write(textwrap.dedent("""
                    import json
                    import os
                    import sys
                    import jiitor as torch
                    import jittor as jt

                    print("RESULT=" + json.dumps({
                        "same_module": torch is jt,
                        "torch_module": sys.modules["torch"] is jt,
                        "runtime_root": os.environ["JITTOR_TORCH_RUNTIME_ROOT"],
                        "jittor_home": os.environ["JITTOR_HOME"],
                        "extensions_dir": os.environ["JITTOR_TORCH_EXTENSIONS_DIR"],
                        "empty_cache": os.environ["JITTOR_TORCH_CUDA_EMPTY_CACHE"],
                    }, sort_keys=True))
                """))
            env = os.environ.copy()
            for name in (
                "JITTOR_TORCH_PROJECT_ROOT",
                "JITTOR_TORCH_RUNTIME_ROOT",
                "JITTOR_TORCH_EXTENSIONS_DIR",
                "JITTOR_TORCH_CUDA_EMPTY_CACHE",
            ):
                env.pop(name, None)
            from jittor_utils import home as jittor_home
            env["JITTOR_HOME"] = jittor_home()
            python_root = os.fspath(pathlib.Path(__file__).resolve().parents[2])
            env["PYTHONPATH"] = os.pathsep.join(filter(None, (
                python_root, env.get("PYTHONPATH", ""),
            )))
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["JITTOR_TORCH_SKIP_EXT_BUILD"] = "1"
            output = subprocess.check_output(
                [sys.executable, entry], cwd=d, env=env, text=True,
            )
            line = next(line for line in output.splitlines() if line.startswith("RESULT="))
            import json
            result = json.loads(line[len("RESULT="):])
            runtime = os.path.join(d, ".cache", "jittor_torch")
            self.assertTrue(result["same_module"])
            self.assertTrue(result["torch_module"])
            self.assertEqual(result["runtime_root"], runtime)
            self.assertEqual(result["jittor_home"], env["JITTOR_HOME"])
            self.assertEqual(
                result["extensions_dir"], os.path.join(runtime, "torch_extensions")
            )
            self.assertEqual(result["empty_cache"], "gc")

    def test_install_does_not_clobber_a_real_torch_namespace(self):
        # Regression test for jittor-core-gaps.md section 3.4.1: importing
        # jittor alongside a REAL PyTorch used to silently replace
        # sys.modules["torch.cuda"] (and dozens of other "torch.*" entries)
        # with torch_compat.py's own fake stand-in stubs -- built for the
        # `import jittor as torch` scenario -- even when jittor was NOT
        # standing in for torch. PyTorch's own lazy CUDA init resolves "the
        # torch.cuda module" via sys.modules to inject native bindings
        # (_get_device_properties, Triton kernel registration through
        # torch.library, ...); those landed in jittor's stub instead of the
        # real module, so torch.cuda._lazy_init() (reading its own module
        # globals) failed with a confusing "module 'torch.cuda' has no
        # attribute '_lazy_init'"/"name '_get_device_properties' is not
        # defined" on the first real CUDA op run afterward -- with no
        # apparent connection to CUDA itself. Needs a subprocess: this
        # process may already have jittor (and its own torch_compat state)
        # loaded from an earlier test in the same run.
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is not installed in this environment")

        script = textwrap.dedent("""
            import json
            import sys
            import torch
            real_cuda = torch.cuda
            real_library = torch.library
            import jittor as jt  # noqa: F401  -- triggers torch_compat.install()

            result = {
                "torch_cuda_sys_modules_preserved":
                    sys.modules.get("torch.cuda") is real_cuda,
                "torch_library_sys_modules_preserved":
                    sys.modules.get("torch.library") is real_library,
                "torch_cuda_attr_preserved": torch.cuda is real_cuda,
            }
            if torch.cuda.is_available():
                try:
                    t = torch.tensor([1.0, 2.0, 3.0], device="cuda:0")
                    result["cuda_lazy_init_ok"] = bool((t.cpu().numpy() == [1, 2, 3]).all())
                except Exception as e:
                    result["cuda_lazy_init_ok"] = False
                    result["cuda_lazy_init_error"] = f"{type(e).__name__}: {e}"
            else:
                result["cuda_lazy_init_ok"] = None
            print("RESULT=" + json.dumps(result))
        """)
        with tempfile.TemporaryDirectory() as d:
            entry = os.path.join(d, "probe.py")
            with open(entry, "w") as f:
                f.write(script)
            env = os.environ.copy()
            python_root = os.fspath(pathlib.Path(__file__).resolve().parents[2])
            env["PYTHONPATH"] = os.pathsep.join(filter(None, (
                python_root, env.get("PYTHONPATH", ""),
            )))
            output = subprocess.check_output(
                [sys.executable, entry], cwd=d, env=env, text=True,
            )
        import json
        line = next(line for line in output.splitlines() if line.startswith("RESULT="))
        result = json.loads(line[len("RESULT="):])
        self.assertTrue(result["torch_cuda_sys_modules_preserved"],
            "sys.modules['torch.cuda'] was replaced by jittor's fake stub")
        self.assertTrue(result["torch_library_sys_modules_preserved"],
            "sys.modules['torch.library'] was replaced by jittor's fake stub")
        self.assertTrue(result["torch_cuda_attr_preserved"])
        if result["cuda_lazy_init_ok"] is not None:
            self.assertTrue(result["cuda_lazy_init_ok"], result.get("cuda_lazy_init_error"))

    def test_strict_bootstrap_propagates_install_failure(self):
        from jittor.torch_shim import enable

        with tempfile.TemporaryDirectory() as d:
            old_sys_path = list(sys.path)
            try:
                with mock.patch.dict(os.environ, {
                    "JITTOR_TORCH_STRICT_BOOTSTRAP": "1",
                    "JITTOR_TORCH_SKIP_EXT_BUILD": "1",
                }), mock.patch(
                    "jittor.torch_compat.install", side_effect=RuntimeError("install failed")
                ):
                    with self.assertRaisesRegex(RuntimeError, "install failed"):
                        enable(
                            project_root=d,
                            runtime_root=os.path.join(d, "runtime"),
                            auto_scan_extensions=False,
                            build_extensions=False,
                            configure_cuda=False,
                            local_home=False,
                            verbose=False,
                        )
            finally:
                sys.path[:] = old_sys_path

    def test_pythonpath_extension_roots_skip_conda_prefix(self):
        from jittor.torch_shim.bootstrap import _pythonpath_extension_roots

        project = pathlib.Path(os.getcwd()).resolve()
        runtime = project / ".jittor_torch_runtime"
        roots = _pythonpath_extension_roots(project, runtime)
        for root in roots:
            self.assertNotIn("site-packages", set(root.parts))
            self.assertFalse(str(root).startswith(os.path.realpath(os.sys.prefix)))

    def test_torch_utils_unknown_submodule_fails_fast(self):
        import importlib
        import time
        import jittor as torch

        start = time.time()
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch.utils.definitely_missing_submodule")
        self.assertLess(time.time() - start, 1.0)
        self.assertTrue(hasattr(torch, "utils"))


if __name__ == "__main__":
    unittest.main()

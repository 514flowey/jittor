"""DLPack ownership with a safe, externally retained CPU producer.

The producer never frees its header or data in a callback: it only counts
releases and optionally changes the version. This catches a post-deleter
header read without deliberately causing a use-after-free. NULL-deleter
imports run in crash-isolated children, including when testing an old core.
CPU buffers exercise the shared lifetime code in CPU and CUDA builds; real
CUDA interoperation is covered separately by the normal DLPack device tests.
"""

import ctypes
import gc
from pathlib import Path

import jittor as jt
import numpy as np
import pytest

from _helpers.child_process import run_child_script


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLPackVersion(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32)]


class _DLManagedTensor(ctypes.Structure):
    pass


class _DLManagedTensorVersioned(ctypes.Structure):
    pass


_ClassicDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_VersionedDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensorVersioned))
_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _ClassicDeleter),
]
_DLManagedTensorVersioned._fields_ = [
    ("version", _DLPackVersion),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _VersionedDeleter),
    ("flags", ctypes.c_uint64),
    ("dl_tensor", _DLTensor),
]

_CapsuleDestructor = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
# Private wrappers avoid changing ctypes.pythonapi's shared argtypes. Capsule
# destructors receive a raw PyObject*, not py_object: the latter would resurrect
# an object whose reference count has already reached zero.
_capsule_new = ctypes.PYFUNCTYPE(
    ctypes.py_object, ctypes.c_void_p, ctypes.c_char_p, _CapsuleDestructor
)(("PyCapsule_New", ctypes.pythonapi))
_capsule_is_valid = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p)(
    ("PyCapsule_IsValid", ctypes.pythonapi)
)
_capsule_get_pointer = ctypes.PYFUNCTYPE(
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p
)(("PyCapsule_GetPointer", ctypes.pythonapi))


class _ManagedProducer:
    """Keep every C pointer/callback alive until the test explicitly cleans up.

    The producer deliberately does not retain capsules. Tests can collect a
    capsule while keeping this owner (and its memory) alive, then inspect the
    callback count. A NULL deleter means this external owner retains the data.
    """

    def __init__(self, versioned, major=1, null_deleter=False, poison_major=False):
        self.versioned = versioned
        self.name = b"dltensor_versioned" if versioned else b"dltensor"
        self.used_name = b"used_dltensor_versioned" if versioned else b"used_dltensor"
        self.buffer = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
        self.shape = (ctypes.c_int64 * 1)(4)
        self.deleter_calls = 0
        self.callback_errors = []
        managed_type = _DLManagedTensorVersioned if versioned else _DLManagedTensor
        deleter_type = _VersionedDeleter if versioned else _ClassicDeleter
        self.managed = managed_type()
        if versioned:
            self.managed.version = _DLPackVersion(major, 0)
        self.managed.dl_tensor = _DLTensor(
            ctypes.addressof(self.buffer),
            _DLDevice(1, 0),
            1,
            _DLDataType(2, 32, 1),
            self.shape,
            None,
            0,
        )

        def release(pointer):
            self.deleter_calls += 1
            if versioned and poison_major:
                # Do not free: the old code safely exposes its post-release
                # read as the wrong version in the resulting error message.
                pointer.contents.version.major = 777

        self.deleter = deleter_type() if null_deleter else deleter_type(release)
        self.managed.deleter = self.deleter
        # Keep the API wrappers and cast function in the closure as well, so
        # delayed capsule cleanup never depends on module teardown ordering.
        valid = _capsule_is_valid
        get_pointer = _capsule_get_pointer
        cast = ctypes.cast
        pointer_type = ctypes.POINTER(managed_type)

        def destroy_capsule(capsule_pointer):
            try:
                if valid(capsule_pointer, self.name):
                    pointer = cast(
                        get_pointer(capsule_pointer, self.name), pointer_type
                    )
                    if pointer.contents.deleter:
                        pointer.contents.deleter(pointer)
            except Exception as error:
                # ctypes would otherwise print and swallow a callback error.
                # Keep it observable by assertions in the surrounding test.
                self.callback_errors.append(repr(error))

        self.capsule_destructor = _CapsuleDestructor(destroy_capsule)

    def capsule(self):
        return _capsule_new(
            ctypes.addressof(self.managed), self.name, self.capsule_destructor
        )

    def is_live(self, capsule):
        return bool(_capsule_is_valid(id(capsule), self.name))

    def is_used(self, capsule):
        return bool(_capsule_is_valid(id(capsule), self.used_name))


def _collect():
    gc.collect()
    jt.gc()


@pytest.fixture(autouse=True)
def _cpu_buffer_scope():
    # Do not migrate the foreign CPU buffer just because a CUDA build is active.
    with jt.flag_scope(use_cuda=0):
        yield
        _collect()


@pytest.mark.parametrize("entrypoint", ["peek", "public"])
def test_version_mismatch_inspection_does_not_consume(entrypoint):
    producer = _ManagedProducer(True, major=2, poison_major=True)
    capsule = producer.capsule()
    inspect = jt.core.dlpack_peek if entrypoint == "peek" else jt.from_dlpack
    with pytest.raises(RuntimeError, match=r"unsupported DLPack major version\s+2\b"):
        inspect(capsule)
    assert producer.is_live(capsule)
    assert producer.deleter_calls == 0
    # Retaining the raw capsule above separates non-consuming inspection from
    # ordinary GC of a temporary protocol capsule after an exception.
    del capsule
    _collect()
    assert producer.deleter_calls == 1
    assert producer.managed.version.major == 777
    assert not producer.callback_errors


def test_version_mismatch_direct_import_caches_major_and_releases_once():
    producer = _ManagedProducer(True, major=2, poison_major=True)
    capsule = producer.capsule()
    with pytest.raises(RuntimeError, match=r"unsupported DLPack major version\s+2\b"):
        jt.core.from_dlpack_capsule(capsule)
    assert producer.managed.version.major == 777
    assert producer.is_used(capsule)
    assert producer.deleter_calls == 1
    with pytest.raises(RuntimeError, match="live|consumed|only.*once"):
        jt.core.from_dlpack_capsule(capsule)
    assert producer.deleter_calls == 1
    del capsule
    _collect()
    assert producer.deleter_calls == 1
    assert not producer.callback_errors


@pytest.mark.parametrize("versioned", [False, True], ids=["classic", "versioned"])
def test_valid_peek_preserves_unconsumed_capsule_cleanup(versioned):
    producer = _ManagedProducer(versioned)
    capsule = producer.capsule()
    shape, strides = jt.core.dlpack_peek(capsule)
    assert tuple(shape) == (4,)
    assert strides is None
    assert producer.is_live(capsule)
    assert producer.deleter_calls == 0
    del capsule
    _collect()
    assert producer.deleter_calls == 1
    assert not producer.callback_errors


@pytest.mark.parametrize("versioned", [False, True], ids=["classic", "versioned"])
def test_imported_storage_releases_only_after_last_alias(versioned):
    producer = _ManagedProducer(versioned)
    capsule = producer.capsule()
    imported = jt.from_dlpack(capsule)
    alias = imported.reshape((2, 2))
    alias.sync()
    assert producer.is_used(capsule)
    assert producer.deleter_calls == 0
    del capsule, imported
    _collect()
    assert producer.deleter_calls == 0
    np.testing.assert_array_equal(alias.numpy(), [[1.0, 2.0], [3.0, 4.0]])
    del alias
    _collect()
    assert producer.deleter_calls == 1
    _collect()
    assert producer.deleter_calls == 1
    assert not producer.callback_errors


@pytest.mark.parametrize("versioned", [False, True], ids=["classic", "versioned"])
def test_validation_failure_preserves_capsule_for_valid_retry(versioned):
    producer = _ManagedProducer(versioned)
    producer.managed.dl_tensor.device.device_type = 99
    capsule = producer.capsule()
    with pytest.raises(RuntimeError, match="unsupported DLDevice.device_type"):
        jt.core.from_dlpack_capsule(capsule)
    assert producer.is_live(capsule)
    assert producer.deleter_calls == 0
    # No buffer was accessed; repair only this owned metadata, then consume it.
    producer.managed.dl_tensor.device.device_type = 1
    imported = jt.core.from_dlpack_capsule(capsule)
    assert producer.is_used(capsule)
    np.testing.assert_array_equal(imported.numpy(), [1.0, 2.0, 3.0, 4.0])
    del imported, capsule
    _collect()
    assert producer.deleter_calls == 1
    assert not producer.callback_errors


def _check_null_deleter(versioned):
    """Executed only by a crash-isolated child, with the owner kept alive."""
    with jt.flag_scope(use_cuda=0):
        producer = _ManagedProducer(versioned, null_deleter=True)
        assert not producer.managed.deleter
        capsule = producer.capsule()
        imported = jt.from_dlpack(capsule)
        assert producer.is_used(capsule)
        np.testing.assert_array_equal(imported.numpy(), [1.0, 2.0, 3.0, 4.0])
        del imported
        _collect()
        del capsule
        _collect()
        assert producer.deleter_calls == 0
        assert not producer.callback_errors
        assert list(producer.buffer) == [1.0, 2.0, 3.0, 4.0]
    print("DLPACK-NULL-DELETER-OK", flush=True)


@pytest.mark.parametrize("versioned", [False, True], ids=["classic", "versioned"])
def test_null_deleter_import_and_release_in_child(versioned):
    child = run_child_script(
        "import runpy\n"
        "case = runpy.run_path(%r)\n"
        "case['_check_null_deleter'](%r)\n"
        % (str(Path(__file__).resolve()), versioned),
        name="dlpack_null_deleter",
        text=True,
        merge_stderr=True,
        crash_isolated=True,
        without_torch_mode=True,
    )
    assert child.returncode == 0, child.stdout
    assert "DLPACK-NULL-DELETER-OK" in child.stdout

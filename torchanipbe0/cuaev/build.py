"""Build helper for the cuaev CUDA extension.

cuaev is a small CUDA/C++ extension (registered under ``torch.ops.cuaev``)
that computes AEVs on-GPU much faster than the pure-PyTorch fallback in
``torchanipbe0/aev.py``. It is not prebuilt/shipped with the repo (there is
no CI/build system here, see CLAUDE.md) - it is compiled once, on demand,
with ``torch.utils.cpp_extension.load`` and cached under ``_build/`` next to
this file.

This machine's CUDA toolchain is split across several package managers
(conda ``cuda-nvcc-tools`` for the compiler, pip ``nvidia-*`` wheels for
runtime libraries, no headers-complete ``cuda-toolkit`` anywhere) and none of
them alone is a complete, discoverable CUDA_HOME. ``_assemble_cuda_home``
stitches a synthetic CUDA_HOME together out of symlinks so that
``torch.utils.cpp_extension`` (which just wants a normal
``$CUDA_HOME/bin/nvcc``) works unmodified. Nothing outside of a cache
directory is written to.

Usage::

    from torchanipbe0.cuaev.build import ensure_cuaev_loaded
    has_cuaev = ensure_cuaev_loaded()  # returns True/False, never raises
"""
import functools
import glob
import os
import site
import sys
import warnings

import torch
import torch.utils.cpp_extension

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_HERE, "_build")
_SANDBOX_DIR = os.path.join(_HERE, "_build", "cuda_home_sandbox")


def _site_dirs():
    dirs = list(site.getsitepackages()) if hasattr(site, "getsitepackages") else []
    if site.ENABLE_USER_SITE:
        dirs.append(site.getusersitepackages())
    dirs += [p for p in sys.path if p and os.path.isdir(p)]
    seen = set()
    for d in dirs:
        d = os.path.realpath(d)
        if d not in seen:
            seen.add(d)
            yield d


def _is_complete_include_dir(include_dir):
    return (os.path.isfile(os.path.join(include_dir, "crt", "host_config.h"))
            and os.path.isfile(os.path.join(include_dir, "nv", "target")))


def _find_complete_cuda_include_dir():
    """A CUDA include tree needs cuda_runtime.h, crt/host_config.h, and
    nv/target (libcu++) to compile anything nontrivial. The pip nvidia-*
    wheels ship a flattened, incomplete subset of these; look for a full
    tree instead: either a system/module-provided toolkit (env-modules
    systems like Compute Canada's EasyBuild stack export CUDA_HOME/
    EBROOTCUDA/CUDA_PATH for a `module load cuda/...`, and this is a
    complete tree there), or a full bundled copy under some package's
    site-packages tree (e.g. tensorflow ships one under
    third_party/gpus/cuda/include)."""
    env = os.environ.get("TORCHANIPBE0_CUDA_INCLUDE")
    if env and _is_complete_include_dir(env):
        return env
    for var in ("CUDA_HOME", "EBROOTCUDA", "CUDA_PATH", "CUDA_ROOT"):
        root = os.environ.get(var)
        if not root:
            continue
        include_dir = os.path.join(root, "include")
        if _is_complete_include_dir(include_dir):
            return include_dir
    cuda_home = torch.utils.cpp_extension.CUDA_HOME
    if cuda_home:
        include_dir = os.path.join(cuda_home, "include")
        if _is_complete_include_dir(include_dir):
            return include_dir
    for base in _site_dirs():
        for hit in glob.glob(os.path.join(base, "**", "cuda", "include", "crt", "host_config.h"), recursive=True):
            include_dir = os.path.dirname(os.path.dirname(hit))
            if _is_complete_include_dir(include_dir):
                return include_dir
    return None


def _find_nvcc():
    env = os.environ.get("TORCHANIPBE0_NVCC")
    if env and os.path.isfile(env):
        return env
    cuda_home = torch.utils.cpp_extension.CUDA_HOME
    if cuda_home:
        candidate = os.path.join(cuda_home, "bin", "nvcc")
        if os.path.isfile(candidate):
            return candidate
    import shutil
    return shutil.which("nvcc")


def _find_nvvm_dir(nvcc_path):
    """nvcc's cicc/libdevice ('nvvm') is looked up relative to nvcc as
    $CUDA_HOME/bin/../targets/<triplet>/nvvm - on this machine (conda
    cuda-nvcc-tools) it instead lives directly at $CUDA_HOME/nvvm."""
    conda_prefix = os.path.dirname(os.path.dirname(nvcc_path))
    direct = os.path.join(conda_prefix, "nvvm")
    if os.path.isfile(os.path.join(direct, "bin", "cicc")):
        return direct
    for hit in glob.glob(os.path.join(conda_prefix, "**", "nvvm", "bin", "cicc"), recursive=True):
        return os.path.dirname(os.path.dirname(hit))
    return None


def _cuda_toolkit_roots():
    """Roots pointing at a CUDA toolkit install (from module-system env vars
    or torch's own CUDA_HOME resolution) - worth a recursive search since
    libcudart could be under lib64/, targets/<triplet>/lib/, etc."""
    roots = []
    for var in ("CUDA_HOME", "EBROOTCUDA", "CUDA_PATH", "CUDA_ROOT"):
        root = os.environ.get(var)
        if root:
            roots.append(root)
    cuda_home = torch.utils.cpp_extension.CUDA_HOME
    if cuda_home:
        roots.append(cuda_home)
    return roots


def _explicit_lib_dirs():
    """Already-flat library directories (LD_LIBRARY_PATH/LIBRARY_PATH
    entries) - checked directly, not recursively, since these can include
    broad system dirs where a recursive glob would be wasteful."""
    dirs = []
    for var in ("LD_LIBRARY_PATH", "LIBRARY_PATH"):
        dirs += [p for p in os.environ.get(var, "").split(os.pathsep) if p]
    return dirs


def _find_libcudart():
    """Match the libcudart major version to the one torch itself links
    against (torch.version.cuda, e.g. '13.0' -> libcudart.so.13). Torch pip
    wheels often bundle their own copy under torch/lib/ with a hash-mangled
    name (e.g. libcudart-9335f6a2.so.13) rather than the plain toolkit name,
    so the glob patterns below wildcard around the version number instead of
    requiring an exact 'libcudart.so.<major>' match."""
    major = (torch.version.cuda or "").split(".")[0]

    def search(pattern):
        for d in _explicit_lib_dirs():
            hits = glob.glob(os.path.join(d, pattern))
            if hits:
                return hits[0]
        for base in _cuda_toolkit_roots() + list(_site_dirs()):
            hits = glob.glob(os.path.join(base, "**", pattern), recursive=True)
            if hits:
                return hits[0]
        return None

    return (search(f"libcudart*.so.{major}*")
            or search(f"libcudart.so.{major}")
            or search("libcudart.so*")
            or search("libcudart*.so*"))


def _assemble_cuda_home():
    """Build (once) a synthetic CUDA_HOME under _build/ made of symlinks
    into whatever CUDA pieces are actually installed on this machine, in
    the layout torch.utils.cpp_extension expects."""
    nvcc = _find_nvcc()
    if not nvcc:
        raise RuntimeError("no nvcc found (set TORCHANIPBE0_NVCC or install a CUDA compiler)")
    include_dir = _find_complete_cuda_include_dir()
    if not include_dir:
        raise RuntimeError(
            "no complete CUDA include tree found (need cuda_runtime.h + crt/host_config.h + nv/target); "
            "set TORCHANIPBE0_CUDA_INCLUDE to one"
        )
    nvvm_dir = _find_nvvm_dir(nvcc)
    if not nvvm_dir:
        raise RuntimeError("no nvvm/bin/cicc found next to nvcc")
    libcudart = _find_libcudart()
    if not libcudart:
        raise RuntimeError("no libcudart.so found")

    os.makedirs(os.path.join(_SANDBOX_DIR, "bin"), exist_ok=True)
    os.makedirs(os.path.join(_SANDBOX_DIR, "targets", "x86_64-linux"), exist_ok=True)
    os.makedirs(os.path.join(_SANDBOX_DIR, "lib64"), exist_ok=True)

    def _link(src, dst):
        if os.path.islink(dst) and os.readlink(dst) == src:
            return
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)

    _link(nvcc, os.path.join(_SANDBOX_DIR, "bin", "nvcc"))
    nvcc_profile = os.path.join(os.path.dirname(nvcc), "nvcc.profile")
    if os.path.isfile(nvcc_profile):
        _link(nvcc_profile, os.path.join(_SANDBOX_DIR, "bin", "nvcc.profile"))
    # nvcc looks for cicc/libdevice ('nvvm') at $CUDA_HOME/nvvm (bin/../nvvm)
    # on some toolkit layouts (e.g. Compute Canada's EasyBuild cudacore/13.2.0)
    # and at $CUDA_HOME/targets/<triplet>/nvvm on others (e.g. conda
    # cuda-nvcc-tools) - link both so either lookup succeeds.
    _link(nvvm_dir, os.path.join(_SANDBOX_DIR, "nvvm"))
    _link(nvvm_dir, os.path.join(_SANDBOX_DIR, "targets", "x86_64-linux", "nvvm"))
    _link(include_dir, os.path.join(_SANDBOX_DIR, "targets", "x86_64-linux", "include"))
    _link(libcudart, os.path.join(_SANDBOX_DIR, "lib64", "libcudart.so"))

    return _SANDBOX_DIR, include_dir


@functools.lru_cache(maxsize=1)
def ensure_cuaev_loaded():
    """Build (if needed) and load the cuaev extension. Returns True if
    torch.ops.cuaev is available afterwards, False otherwise (never raises;
    failures are reported as a warning so CPU-only / no-CUDA use is
    unaffected)."""
    if not torch.cuda.is_available():
        return False
    try:
        sandbox, include_dir = _assemble_cuda_home()
        os.environ["CUDA_HOME"] = sandbox
        torch.utils.cpp_extension.CUDA_HOME = sandbox
        sandbox_bin = os.path.join(sandbox, "bin")
        if sandbox_bin not in os.environ["PATH"].split(os.pathsep):
            os.environ["PATH"] = sandbox_bin + os.pathsep + os.environ["PATH"]

        module = torch.utils.cpp_extension.load(
            name="cuaev",
            sources=[os.path.join(_HERE, "cuaev.cpp"), os.path.join(_HERE, "aev.cu")],
            extra_include_paths=[_HERE, include_dir],
            extra_cflags=["-std=c++17"],
            extra_cuda_cflags=["-std=c++17", "--expt-extended-lambda", "-use_fast_math"],
            build_directory=_BUILD_DIR,
            verbose=False,
        )
        return hasattr(torch.ops, "cuaev") and module is not None
    except Exception as e:
        warnings.warn(f"cuaev extension unavailable, falling back to the pure-PyTorch AEV path: {e}")
        return False

import functools
import os
import sysconfig
import hashlib
import subprocess
import tempfile
from pathlib import Path
from triton.runtime.build import _build
from triton.runtime.cache import get_cache_manager
from triton.runtime import _allocation
from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver
from triton._utils import parse_list_string

current_dir = Path(__file__).resolve().parent
import torch
from torch.utils import cpp_extension
module = cpp_extension.load(name="aipu",sources=[current_dir / "aipu_torch_dev.cpp"], extra_include_paths=[], extra_cflags=["-g"], verbose=True)
torch.utils.rename_privateuse1_backend("aipu")
torch._register_device_module("aipu", module)
torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)



# ------------------------
# Utils
# ------------------------


class AIPUUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(AIPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        #mod = compile_module_from_src(Path(os.path.join(dirname, "aipu_driver.c")).read_text(), "aipu_utils")
        #self.load_binary = mod.load_binary
        #self.get_device_properties = mod.get_device_properties
        pass


# ------------------------
# Launcher
# ------------------------



def make_launcher(constants, signature, ids):
    # create c source code to build as launcher module
    src = ""
    return src


class AIPULauncher(object):

    def __init__(self, src, metadata):
        ids = {"ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()}
        constants = src.constants if hasattr(src, "constants") else dict()
        constants = {idx: value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        src = make_launcher(constants, signature, ids)
        # mod = compile_module_from_src(src, "__triton_launcher")
        # self.launch = mod.launch

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):
        # self.launch(gridX, gridY, gridZ, stream, function, global_scratch, *args)
        # call function here
        pass


class AIPUDriver(GPUDriver):

    def __init__(self):
        self.utils = AIPUUtils()  # TODO: make static
        self.launcher_cls = AIPULauncher
        super().__init__()

    def get_current_target(self):
        device = 0
        capability = self.get_device_capability(device)
        capability = capability[0] * 10 + capability[1]
        warp_size = 32
        return GPUTarget("aipu", capability, warp_size)

    def get_active_torch_device(self):
        import torch
        return torch.device("aipu", 0)

    def get_device_interface(self):
        import torch
        return torch.aipu

    @staticmethod
    def is_active():
        import torch
        return torch.aipu.is_available()

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch

        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2 cache
        # doesn't contain any input data before the run
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='aipu')

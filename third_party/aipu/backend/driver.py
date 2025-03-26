import os
import pickle
import torch
from pathlib import Path
from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


# ------------------------
# Utils
# ------------------------


def load_binary(name, kernel, shared, device):
    return None, kernel, 1, 0


class AIPUUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(AIPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.load_binary = load_binary
        properties_dict = {
            "max_shared_mem": 256 * 1024,
            "multiprocessor_count": 4,
            "max_num_regs": 32,
            "warpSize": 4
        }
        self.get_device_properties = lambda device: properties_dict


# ------------------------
# Launcher
# ------------------------


class AIPULauncher(object):

    def __init__(self, src, metadata):
        ...

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):
        try:
            from flag_gems.utils.tensor_wrapper import StridedBuffer
        except:
            StridedBuffer = torch.Tensor

        ex = pickle.loads(function)
        np_args = []
        args = args[4:]
        for arg in args:
            if isinstance(arg, torch.Tensor):
                np_args.append(arg.cpu().numpy())
            elif isinstance(arg, StridedBuffer):
                np_args.append(arg._base.cpu().numpy())
            else:
                np_args.append(arg)

        tail_args = [gridX, gridY, gridZ, 0, 0, 0]
        tec_num = 4

        for i in range((gridX + tec_num - 1) // tec_num):
            tail_args[3] = i
            ex(*(np_args + tail_args))

        for i, param_info in enumerate(ex._cur_param_infos):
            if param_info.is_output_tensor:
                aipu_tensor = args[i] if isinstance(args[i], torch.Tensor) else args[i]._base
                aipu_tensor.copy_(torch.from_numpy(np_args[i]))


class AIPUDriver(DriverBase):

    def __init__(self):
        self.utils = AIPUUtils()  # TODO: make static
        self.launcher_cls = AIPULauncher

        import torch
        self.get_current_stream = lambda x: x
        self.get_current_device = torch.aipu.current_device

        super().__init__()

    def get_current_target(self):
        warp_size = 4
        return GPUTarget("aipu", "x2", warp_size)

    def get_active_torch_device(self):
        import torch
        return torch.device("aipu", 0)

    def get_device_interface(self):
        import torch
        return torch.aipu

    @staticmethod
    def is_active():
        import torch
        from torch.utils import cpp_extension

        try:
            torch.aipu.is_available()
        except:
            # TODO(aipu-teams): Remove this path later.
            os.environ["CXX"] = "/arm/tools/gnu/gcc/9.3.0/rhe7-x86_64/bin/g++"
            current_dir = Path(__file__).resolve().parent
            extra_ldflags = [f"-L{path}" for path in os.getenv("LD_LIBRARY_PATH").split(":")]
            extra_ldflags.append("-laipudrv")
            module = cpp_extension.load(
                name="aipu",
                sources=[current_dir / "aipu_torch_dev.cpp"],
                extra_include_paths=[os.getenv("ZHOUYI_LINUX_DRIVER_HOME")  + "/driver/umd/include"],
                extra_ldflags=extra_ldflags,
                verbose=True
            )

            torch.utils.rename_privateuse1_backend("aipu")
            torch._register_device_module("aipu", module)
            torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)
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

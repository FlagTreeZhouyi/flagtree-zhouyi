import os
import subprocess
import torch
import numpy as np
from pathlib import Path
from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


# ------------------------
# Utils
# ------------------------


def load_binary(name, kernel, shared, device):
    return None, kernel, 0, 0


class AIPUUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(AIPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.load_binary = load_binary
        self.get_device_properties = lambda device: {"max_shared_mem": 256 * 1024}


# ------------------------
# Launcher
# ------------------------


def launch_aipu(input0, input1):
    os.chdir("./tmp")

    input0.tofile("input0.bin")
    input1.tofile("input1.bin")

    command = ['aipudumper', "aipu.bin", '-i', 'input0.bin,input1.bin']
    subprocess.run(command, capture_output=True, text=True)

    command = ['aipu_simulator_x2', 'temp.cfg']
    subprocess.run(command, capture_output=True, text=True)

    sim_out = np.fromfile("./output.bin", np.float32)
    os.chdir("..")

    return sim_out


class AIPULauncher(object):

    def __init__(self, src, metadata):
        self.launch = launch_aipu

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):
        # Store aipu.bin
        aipu_bin = function
        with open("./tmp/aipu.bin", 'wb') as file:
            file.write(aipu_bin)

        block_size = 1024

        # Store inputs
        print(len(args))
        x, y, output, n = args[-4:]
        x, y = x.aipu(), y.aipu()

        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()

        sim_out = []
        idx = 0
        for i in range(gridX):
            sim_out.append(self.launch(x_np[idx:idx + block_size], y_np[idx:idx + block_size]))
            idx += block_size

        sim_out = np.concatenate(sim_out)
        output.copy_(torch.from_numpy(sim_out[:n]))


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

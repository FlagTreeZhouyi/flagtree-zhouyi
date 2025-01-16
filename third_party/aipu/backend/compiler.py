from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, llvm, aipu

from dataclasses import dataclass
import functools
from typing import Any, Dict, Tuple, Optional
from types import ModuleType
import hashlib
import re
import tempfile
import signal
import os
import subprocess
from pathlib import Path
import sysconfig


@dataclass(frozen=True)
class AIPUOptions:
    num_tecs: int = 4
    num_stages: int = 1
    num_cores: int = 3
    cluster_dims: tuple = (1, 1, 1)
    arch: str = "x2"
    backend_name: str = "aipu"


class AIPUBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'aipu'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.capability = target.arch
        assert isinstance(self.capability, int)
        self.binary_ext = "bin"

    def load_dialects(self, ctx):
        aipu.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        # add pass here
        pm.run(mod)
        return mod

    def get_module_map(self):
        return {}

    def parse_options(self, opts):
        return AIPUOptions(4, 1, 3, (1, 1, 1), "x2", "aipu")

    def add_stages(self, stages, options):
        # add new build stages here
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        return f"aipu_builder"

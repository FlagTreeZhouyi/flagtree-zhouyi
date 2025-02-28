import pickle
from triton.backends.aipu.codegen import codegenAIPU
from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, aipu, passes

from dataclasses import dataclass
import functools
import hashlib
from typing import Any, Dict
from types import ModuleType


@dataclass(frozen=True)
class AIPUOptions:
    num_tecs: int = 4
    num_stages: int = 2
    num_cores: int = 3
    cluster_dims: tuple = (1, 1, 1)
    arch: str = "x2"
    backend_name: str = "aipu"
    debug: bool = False
    sanitize_overflow: bool = True
    num_warps: int = 4

    def hash(self):
        hash_dict = dict(self.__dict__)
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()



class AIPUBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'aipu'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.capability = target.arch
        self.binary_ext = "bin"

    def parse_options(self, opts) -> Any:
        return AIPUOptions()

    def pack_metadata(self, metadata):
        return (
            metadata.num_tecs,
            metadata.num_cores,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )

    def get_codegen_implementation(self):
        return {}

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.cuda import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        aipu.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_linalg(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        # add pass here
        aipu.passes.convert.add_triton_to_linalg_pipeline(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_aipubin(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        # add pass here
        aipu.passes.convert.add_one_shot_bufferize(pm)
        aipu.passes.convert.add_linalg_to_loops(pm)
        pm.run(mod)
        ex = codegenAIPU(mod)

        metadata["name"] = ex._func_name
        metadata["shared"] = 1
        return pickle.dumps(ex)

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["linalg"] = lambda src, metadata: self.make_linalg(src, metadata, options)
        stages["bin"] = lambda src, metadata: self.make_aipubin(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        return f"aipu_builder"

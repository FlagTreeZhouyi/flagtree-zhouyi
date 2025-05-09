import binascii
import hashlib
import importlib.util
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import triton
from triton.compiler.code_generator import kernel_suffix
from triton.backends.xpu.driver import get_xpu_spec

desc = """
Triton ahead-of-time compiler:

This program compiles the kernel with name `kernel-name` in the file at the
provided `path` into self-contained C source-code that embeds the `cubin`
data along with utilities to load, unload and launch the kernel.

signature is provided as a list of (optionally divisibility-hinted) types
or constexpr values, e.g.

`compile.py --kernel-name kernel --signature "*fp32:16, i32:16, 1024, i32" --out-name kernel /path/to/kernel.py`

will compile triton.JITFunction of name `kernel` inside the file `/path/to/kernel.py`.
Said kernel will be specialized such that argument 0, 1 are assumed to be multiple of 16,
and argument 2 is assumed to be a compile-time constant of value 1024, i.e. it won't be part of the generated prototype.

The resulting entry point will have signature

CUresult kernel_{specialization_suffix}(CUstream stream, unsigned gX, unsigned gY, unsigned gZ, float* arg0, int32_t arg1, int32_t arg2)

Different such specialized entry points can be combined using the `linker.py` script.

NOTE: when resolving the scope of /path/to/kernel.py, the file will be executed from within its parent directory with the python interpreter
used to run this `compile.py` script
"""


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "XPUdeviceptr"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


if __name__ == "__main__":

    # command-line arguments
    parser = ArgumentParser(description=desc)
    parser.add_argument("path",
                        help="Path to Python source containing desired kernel in its scope. File will be executed.")
    parser.add_argument("--kernel-name", "-n", type=str, default="", help="Name of the kernel to compile",
                        required=True)
    parser.add_argument("--num-warps", "-w", type=int, default=1, help="Number of warps to launch the kernel")
    parser.add_argument("--num-stages", "-ns", type=int, default=3,
                        help="Number of stages (meta-parameter of the kernel)")
    parser.add_argument("--out-name", "-on", type=str, default=None, help="Out name for the compiled kernel")
    parser.add_argument("--out-path", "-o", type=Path, default=None, help="Out filename")
    parser.add_argument("--signature", "-s", type=str, help="Signature of the kernel", required=True)
    parser.add_argument("--grid", "-g", type=str, help="Launch grid of the kernel", required=True)
    args = parser.parse_args()

    out_name = args.out_name if args.out_name else args.kernel_name
    out_path = args.out_path if args.out_path else Path(out_name)

    # execute python sources and extract functions wrapped in JITFunction
    arg_path = Path(args.path)
    sys.path.insert(0, str(arg_path.parent))
    spec = importlib.util.spec_from_file_location(arg_path.stem, arg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kernel = getattr(mod, args.kernel_name)
    grid = args.grid.split(",")
    assert len(grid) == 3

    # validate and parse signature
    signature = list(map(lambda s: s.strip(" "), args.signature.split(",")))

    def hash_signature(signature: List[str]):
        m = hashlib.sha256()
        m.update(" ".join(signature).encode())
        return m.hexdigest()[:8]

    meta_sig = f"warps{args.num_warps}xstages{args.num_stages}"
    sig_hash = hash_signature(signature + [meta_sig])

    def constexpr(s):
        try:
            ret = int(s)
            return ret
        except ValueError:
            pass
        try:
            ret = float(s)
            return ret
        except ValueError:
            pass
        return None

    hints = {i: constexpr(s.split(":")[1]) for i, s in enumerate(signature) if ":" in s}
    hints = {k: v for k, v in hints.items() if v is not None}
    constants = {i: constexpr(s) for i, s in enumerate(signature)}
    constants = {k: v for k, v in constants.items() if v is not None}
    signature = {i: s.split(":")[0] for i, s in enumerate(signature) if i not in constants}
    const_sig = 'x'.join([str(v) for v in constants.values()])
    doc_string = [f"{kernel.arg_names[i]}={constants[i]}" for i in constants.keys()]
    doc_string += [f"num_warps={args.num_warps}", f"num_stages={args.num_stages}"]

    # compile ast into cubin
    for h in hints.values():
        assert h in [1, 16], f"Only 1 and 16 are valid hints, got {h}"
    divisible_by_16 = [i for i, h in hints.items() if h == 16]
    equal_to_1 = [i for i, h in hints.items() if h == 1]
    attrs = triton.compiler.AttrsDescriptor(divisible_by_16=divisible_by_16, equal_to_1=equal_to_1)
    for i in equal_to_1:
        constants.update({i: 1})
    src = triton.compiler.ASTSource(fn=kernel, constants=constants, signature=signature, attrs=attrs)
    opts = {"num_warps": args.num_warps, "num_stages": args.num_stages}
    ccinfo = triton.compile(src, options=opts)
    arg_names = []
    arg_types = []
    for i in signature.keys():
        if i not in equal_to_1:
            arg_names += [kernel.arg_names[i]]
            arg_types += [signature[i]]

    # dump C stub code
    suffix = kernel_suffix(signature.values(), attrs)
    func_name = '_'.join([out_name, sig_hash, suffix])
    hex_ = str(binascii.hexlify(ccinfo.asm["xpubin"]))[2:-1]
    xpu_arch = ccinfo.metadata.xpu_arch
    is_sdnn = ccinfo.metadata.is_sdnn

    def generate_argument_set_code():
        newline = "\n    "
        eightBytesTypes = ['XPUdeviceptr', 'int64_t', 'uint64_t', 'double']
        lines = []
        for i, arg in enumerate(arg_names):
            is_align_to_8 = (ty_to_cpp(arg_types[i]) in eightBytesTypes) and (xpu_arch == 3)
            if is_align_to_8:
                offset_align_to_8_line = "offset = alignSizeTo8Bytes(offset);"
                lines.append(offset_align_to_8_line)
            align_fn = "alignSizeTo8Bytes" if is_align_to_8 else "alignSizeTo4Bytes"
            xpu_check_line = f"XPU_CHECK(xpu_launch_argument_set(&{arg}, sizeof({arg}), offset));"
            offset_increment_line = f"offset += {align_fn}(sizeof({arg}));"
            lines.append(f"{xpu_check_line}    {offset_increment_line}")
        return newline.join(lines)

    params = {
        "kernel_name": func_name,
        "triton_kernel_name": args.kernel_name,
        "bin_size": len(hex_),
        "bin_data": ", ".join([f"0x{x}{y}" for x, y in zip(hex_[::2], hex_[1::2])]),
        "signature": ", ".join([f"{ty_to_cpp(ty)} {name}" for name, ty in zip(arg_names, arg_types)]),
        "full_signature": ", ".join([f"{ty_to_cpp(signature[i])} {kernel.arg_names[i]}" for i in signature.keys()]),
        "arg_pointers": ", ".join([f"&{arg}" for arg in arg_names]),
        "num_args": len(arg_names),
        "kernel_docstring": doc_string,
        "shared": ccinfo.metadata.shared,
        "num_warps": args.num_warps,
        "algo_info": '_'.join([const_sig, meta_sig]),
        "gridX": grid[0],
        "gridY": grid[1],
        "gridZ": grid[2],
        "_placeholder": "",
        "ewt_data": "",
        "nclusters": get_xpu_spec(xpu_arch, is_sdnn)[0],
        "ncores": get_xpu_spec(xpu_arch, is_sdnn)[1],
        "kernel_type": "KT_SDCDNN" if is_sdnn else "KT_CLUSTER",
        "argument_set_code": generate_argument_set_code(),
        "load_ewtable_code": "",
        "printf_buf_offset": 0,
    }
    for ext in ['h', 'c']:
        template_path = Path(__file__).parent / f"compile_xpu.{ext}"
        with out_path.with_suffix(f".{sig_hash}_{suffix}.{ext}").open("w") as fp:
            fp.write(Path(template_path).read_text().format(**params))

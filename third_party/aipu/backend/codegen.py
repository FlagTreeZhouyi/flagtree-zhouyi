from triton._C.libtriton import ir as triton_ir
from tvm import tir, ir, aipu
from tvm.script.parser import tir as T
from tvm.aipu import script as S


_TYPE_MAPPING = {"f32": "float32", "i32": "int32", "index": "int32"}


def _get_type(ty):
    ty = str(ty)
    if ty.startswith("memref"):
        start = ty.find("x")
        end = ty.find(",", start)
        if end == -1:
            end = ty.find(">", start)
        dtype = _TYPE_MAPPING[ty[start + 1:end]]
        return ir.PointerType(ir.PrimType(dtype))

    if ty in _TYPE_MAPPING:
        return _TYPE_MAPPING[ty]

    raise RuntimeError(f"Cannot parse type {ty}")


def _get_shape(ty):
    ty = str(ty)
    if ty.startswith("memref"):
        start = ty.find("<")
        end = ty.find("x", start)
        return [int(ty[start + 1:end])]

    raise RuntimeError(f"Cannot parse shape {ty}")


class CodeGenerator():
    def __init__(self, mod) -> None:
        self.mod = mod
        self.ib = tir.ir_builder.create()
        self.id_to_var_or_buf = {}
        self.name_idx = 0
        self.prim_func = None
        self.scope_stack = []
        self.gridx_var = None

    def create_var_name(self):
        var_name = "var_" + str(self.name_idx)
        self.name_idx += 1
        return var_name

    def emit_let(self, value, related_value):
        var_name = self.create_var_name()
        let_var = self.ib.let(var_name, value)
        self.id_to_var_or_buf[related_value.id()] = let_var

    def get_operand(self, op, idx):
        return self.get_or_create_var(op.get_operand(idx))

    def get_or_create_var(self, value):
        if value.id() in self.id_to_var_or_buf:
            return self.id_to_var_or_buf[value.id()]

        value_type = _get_type(value.get_type())
        var = T.Var(self.create_var_name(), value_type)
        self.id_to_var_or_buf[value.id()] = var
        return var


    def dispatch(self, op, stage):
        op_name = op.get_name()
        # Memref Dialect
        if op_name == "memref.reinterpret_cast":
            self.gen_memref_reinterpret_cast(op)
        if op_name == "memref.load":
            self.gen_memref_load(op)
        if op_name == "memref.store":
            self.gen_memref_store(op)
        if op_name == "memref.alloc":
            self.gen_memref_alloc(op)
        if op_name == "memref.copy":
            self.gen_memref_copy(op)
        if op_name == "memref.subview":
            self.gen_memref_subview(op)
        # Arith Dialect
        if op_name == "arith.constant":
            self.gen_arith_constant(op)
        if op_name == "arith.index_cast":
            self.gen_arith_index_cast(op)
        if op_name in ("arith.addf", "arith.addi"):
            self.gen_arith_binary(op, T.Add)
        if op_name in ("arith.subf", "arith.subi"):
            self.gen_arith_binary(op, T.Sub)
        if op_name == "arith.muli":
            self.gen_arith_binary(op, T.Mul)
        if op_name == "arith.minsi":
            self.gen_arith_binary(op, T.Min)
        if op_name == "arith.maxsi":
            self.gen_arith_binary(op, T.Max)
        # Func Dialect
        if op_name == "func.return":
            self.gen_func_return(op)
        if op_name == "func.func":
            self.gen_func_func(op, stage)
        # Scf Dialect
        if op_name == "scf.for":
            self.gen_scf_for(op, stage)
        if op_name == "scf.yield":
            pass

    def generate(self):
        self.mod.generic_walk(self.dispatch)
        bm = aipu.tir.BuildManager()
        return bm.build(self.prim_func)

    def gen_memref_reinterpret_cast(self, op):
        result = op.get_result(0)
        arg = self.get_operand(op, 0)
        offset = 0
        if op.get_num_operands() == 2:
            offset = self.get_operand(op, 1)

        buffer = T.Buffer((-1,), elem_offset=offset, data=arg)
        self.id_to_var_or_buf[result.id()] = buffer

    def gen_memref_load(self, op):
        result = op.get_result(0)
        buffer = self.get_operand(op, 0)
        index = self.get_operand(op, 1)

        self.emit_let(T.BufferLoad(buffer, [index]), result)

    def gen_memref_store(self, op):
        value = self.get_operand(op, 0)
        buffer = self.get_operand(op, 1)
        index = self.get_operand(op, 2)

        self.ib.emit(tir.BufferStore(buffer, value, [index]))

    def gen_memref_alloc(self, op):
        result = op.get_result(0)
        result_type = result.get_type()
        dtype = _get_type(result_type).element_type.dtype
        shape = _get_shape(result_type)

        buf = self.ib.allocate(dtype, shape, scope="lsram")
        self.id_to_var_or_buf[result.id()] = buf

    def gen_memref_copy(self, op):
        src = self.get_operand(op, 0)
        dst = self.get_operand(op, 1)
        width = src.shape[0]

        dma_copy = S.dma_copy(dst, src, width)
        self.ib.emit(dma_copy)

    def gen_memref_subview(self, op):
        result = op.get_result(0)
        buffer = self.get_operand(op, 0)
        size = self.get_operand(op, 1)

        buffer = buffer._buffer if isinstance(buffer, tir.ir_builder.BufferVar) else buffer
        subview = T.Buffer(size, elem_offset=buffer.elem_offset, data=buffer.data)
        self.id_to_var_or_buf[result.id()] = subview

    def gen_arith_constant(self, op):
        result = op.get_result(0)
        value = op.get_int_attr("value")

        self.emit_let(T.int32(value), result)

    def gen_arith_index_cast(self, op):
        result = op.get_result(0)
        arg0 = self.get_operand(op, 0)

        self.emit_let(T.Cast("int32", arg0), result)

    def gen_arith_binary(self, op, method):
        result = op.get_result(0)
        arg0 = self.get_operand(op, 0)
        arg1 = self.get_operand(op, 1)

        self.emit_let(method(arg0, arg1), result)

    def gen_func_return(self, op):
        self.ib.emit(T.ret(None))

    def gen_func_func(self, op, stage):
        if stage.is_before_all_regions():
            block = op.get_region(0).get_block(0)
            arg_nums = block.get_num_arguments()
            gridx = block.arg(arg_nums - 3)
            gridx_var = self.get_or_create_var(gridx)
            self.gridx_var = gridx_var

            tid = T.Add(T.Mul(gridx_var, S.get_local_size()), S.get_local_id())
            self.emit_let(tid, gridx)

        if stage.is_after_all_regions():
            func_name = op.get_str_attr("sym_name")
            block = op.get_region(0).get_block(0)
            arg_nums = block.get_num_arguments()

            args = []
            for i in range(arg_nums):
                if i == arg_nums - 3:
                    args.append(self.gridx_var)
                else:
                    arg = block.arg(i)
                    var = self.get_or_create_var(arg)
                    args.append(var)

            self.prim_func = tir.PrimFunc(args, self.ib.get()).with_attr(
                "global_symbol", func_name
            )

    def gen_scf_for(self, op, stage):
        if stage.is_before_all_regions():
            begin = self.get_operand(op, 0)
            end = self.get_operand(op, 1)
            block = op.get_region(0).get_block(0)
            loop_var = block.arg(0)

            for_range = self.ib.for_range(begin, end)
            self.scope_stack.append(for_range)
            self.id_to_var_or_buf[loop_var.id()] = for_range.__enter__()

        if stage.is_after_all_regions():
            self.scope_stack.pop().__exit__(None, None, None)


def codegenAIPU(mod):
    generator = CodeGenerator(mod)
    return generator.generate()

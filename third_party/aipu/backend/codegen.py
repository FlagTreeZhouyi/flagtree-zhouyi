from tvm import tir, ir, aipu
from tvm.script.parser import tir as T
from tvm.aipu import script as S


_TYPE_MAPPING = {"f32": "float32", "i32": "int32", "index": "int32"}
_CMP_MAPPING = {0: T.EQ, 1: T.NE, 2: T.LT, 3: T.LE, 4: T.GT, 5: T.GE, 6: T.LT, 7: T.LE, 8: T.GT, 9: T.GE}


def _get_type(value):
    ty = str(value.get_type())
    if ty.startswith("memref"):
        start = ty.find("x")
        if start == -1:
            start = ty.find("<")
        end = ty.find(",", start)
        if end == -1:
            end = ty.find(">", start)
        dtype = _TYPE_MAPPING[ty[start + 1:end]]
        return ir.PointerType(ir.PrimType(dtype))

    if ty in _TYPE_MAPPING:
        return _TYPE_MAPPING[ty]

    raise RuntimeError(f"Cannot parse type {ty}")


def _get_shape(value):
    ty = str(value.get_type())
    if ty.startswith("memref"):
        start = ty.find("<")
        end = ty.find("x", start)
        if end == -1:
            return [1]
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

        value_type = _get_type(value)
        var = T.Var(self.create_var_name(), value_type)
        self.id_to_var_or_buf[value.id()] = var
        return var

    def for_range(self, begin, end, step, kind="serial"):
        self.ib._seq_stack.append([])

        loop_var = T.Var(self.create_var_name(), "int32")
        extent = end if begin == 0 else (end - begin)
        annotations={"step": step}

        def _exit_cb():
            if kind == "serial":
                kind_id = tir.ForKind.SERIAL
            elif kind == "parallel":
                kind_id = tir.ForKind.PARALLEL
            elif kind == "vectorize":
                kind_id = tir.ForKind.VECTORIZED
            elif kind == "unroll":
                kind_id = tir.ForKind.UNROLLED
            else:
                raise ValueError("Unknown kind")
            self.ib.emit(
                tir.For(
                    loop_var,
                    begin,
                    extent,
                    kind_id,
                    self.ib._pop_seq(),
                    annotations=annotations,
                )
            )

        return tir.ir_builder.WithScope(loop_var, _exit_cb)

    def enter_scope(self, scope):
        assert isinstance(scope, tir.ir_builder.WithScope)
        self.scope_stack.append(scope)
        return scope.__enter__()

    def exit_scope(self):
        self.scope_stack.pop().__exit__(None, None, None)

    def dispatch(self, op, stage):
        op_name = op.get_name()
        match op_name:
            # Memref Dialect
            case "memref.reinterpret_cast":
                self.gen_memref_reinterpret_cast(op)
            case "memref.load":
                self.gen_memref_load(op)
            case "memref.store":
                self.gen_memref_store(op)
            case "memref.alloc":
                self.gen_memref_alloc(op)
            case "memref.copy":
                self.gen_memref_copy(op)
            case "memref.subview":
                self.gen_memref_subview(op)
            # Arith Dialect
            case "arith.constant":
                self.gen_arith_constant(op)
            case "arith.index_cast":
                self.gen_arith_index_cast(op)
            case "arith.addf" | "arith.addi":
                self.gen_arith_binary(op, T.Add)
            case "arith.subf" | "arith.subi":
                self.gen_arith_binary(op, T.Sub)
            case "arith.muli":
                self.gen_arith_binary(op, T.Mul)
            case "arith.minsi":
                self.gen_arith_binary(op, T.Min)
            case "arith.maxsi" | "arith.maxnumf":
                self.gen_arith_binary(op, T.Max)
            case "arith.divf":
                self.gen_arith_binary(op, T.Div)
            case "arith.cmpi":
                self.gen_arith_binary(op, _CMP_MAPPING[op.get_attr("predicate")])
            # Math Dialect
            case "math.exp":
                self.gen_math_exp(op)
            # Func Dialect
            case "func.return":
                self.gen_func_return(op)
            case "func.func":
                self.gen_func_func(op, stage)
            # Scf Dialect
            case "scf.for":
                self.gen_scf_for(op, stage)
            case "scf.if":
                self.gen_scf_if(op, stage)
            case "scf.yield":
                pass
            # Others
            case "builtin.module":
                pass
            case _:
                raise RuntimeError(f"Unsupport op {op_name}.")

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
        index = 0
        if op.get_num_operands() == 2:
            index = self.get_operand(op, 1)

        self.emit_let(T.BufferLoad(buffer, [index]), result)

    def gen_memref_store(self, op):
        value = self.get_operand(op, 0)
        buffer = self.get_operand(op, 1)
        index = 0
        if op.get_num_operands() == 3:
            index = self.get_operand(op, 2)

        self.ib.emit(tir.BufferStore(buffer, value, [index]))

    def gen_memref_alloc(self, op):
        result = op.get_result(0)
        dtype = _get_type(result).element_type.dtype
        shape = _get_shape(result)

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
        dtype = _get_type(result)
        value = op.get_attr("value")

        self.emit_let(tir.const(value, dtype), result)

    def gen_arith_index_cast(self, op):
        result = op.get_result(0)
        arg0 = self.get_operand(op, 0)

        self.emit_let(T.Cast("int32", arg0), result)

    def gen_arith_binary(self, op, method):
        result = op.get_result(0)
        arg0 = self.get_operand(op, 0)
        arg1 = self.get_operand(op, 1)

        self.emit_let(method(arg0, arg1), result)

    def gen_math_exp(self, op):
        result = op.get_result(0)
        arg0 = self.get_operand(op, 0)

        self.emit_let(S.exp(arg0), result)

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
            func_name = op.get_attr("sym_name")
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
            step = self.get_operand(op, 2)

            block = op.get_region(0).get_block(0)
            loop_iter = block.arg(0)

            for_range = self.for_range(begin, end, step)
            loop_var = self.enter_scope(for_range)
            self.id_to_var_or_buf[loop_iter.id()] = loop_var

        if stage.is_after_all_regions():
            self.exit_scope()

    def gen_scf_if(self, op, stage):
        # If branch
        if stage.is_before_all_regions():
            cond = self.get_operand(op, 0)

            if_scope = self.ib.if_scope(cond)
            self.enter_scope(if_scope)
        # Else branch
        if stage.is_after_region(0):
            self.exit_scope()
            else_scope = self.ib.else_scope()
            self.enter_scope(else_scope)
        # Finish
        if stage.is_after_all_regions():
            self.exit_scope()


def codegenAIPU(mod):
    generator = CodeGenerator(mod)
    return generator.generate()

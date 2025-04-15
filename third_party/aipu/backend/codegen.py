from tvm import tir, ir, aipu
from tvm.script.parser import tir as T
from tvm.aipu import script as S
from mlir import ir as mlir_ir
from mlir.dialects import func

_CMP_MAPPING = {0: T.EQ, 1: T.NE, 2: T.LT, 3: T.LE, 4: T.GT, 5: T.GE, 6: T.LT, 7: T.LE, 8: T.GT, 9: T.GE}


class WalkStage:

    def __init__(self, op):
        self.num_regions = len(op.regions)
        self.next_region = 0

    def is_before_all_regions(self):
        return self.next_region == 0

    def is_before_region(self, region):
        return self.next_region == region

    def is_after_region(self, region):
        return self.next_region == region + 1

    def is_after_all_regions(self):
        return self.next_region == self.num_regions

    def advance(self):
        self.next_region += 1

    def get_next_region(self):
        return self.next_region


def _convert_scalar_type(type):
    """convert from mlir_type to tvm_type_str"""
    if isinstance(type, mlir_ir.IndexType):
        return "int32"
    if isinstance(type, mlir_ir.IntegerType):
        sign_str = "u" if type.is_unsigned else ""
        width = min(32, type.width)
        return f"{sign_str}int{width}"
    if isinstance(type, mlir_ir.FloatType):
        return f"float{type.width}"
    raise RuntimeError(f"not scalar type {type}")


def _convert_vector_type(type):
    """convert from mlir_type to tvm_type_str"""
    if isinstance(type, mlir_ir.VectorType):
        assert type.rank == 1
        e_dtype = _convert_scalar_type(type.element_type)
        vtype = f"{e_dtype}x{type.shape[0]}"
        return vtype
    raise RuntimeError(f"not scalar type {type}")


def _get_type(value):
    ty = value.type

    if isinstance(ty, mlir_ir.IndexType | mlir_ir.IntegerType | mlir_ir.FloatType):
        return _convert_scalar_type(ty)
    elif isinstance(ty, mlir_ir.VectorType):
        return _convert_vector_type(ty)
    elif isinstance(ty, mlir_ir.MemRefType | mlir_ir.UnrankedMemRefType):
        e_dtype = _convert_scalar_type(ty.element_type)
        return ir.PointerType(ir.PrimType(e_dtype))

    raise RuntimeError(f"Cannot parse type {ty}")


def _get_shape(value):
    ty = value.type
    if isinstance(ty, mlir_ir.ShapedType):
        if ty.rank > 0:
            return ty.shape
        return [1]

    raise RuntimeError(f"Cannot parse shape {ty}")


class CodeGenerator():

    def __init__(self, mod) -> None:
        self.mod = mod
        self.ib = tir.ir_builder.create()
        # Dictionary to map MLIR values to corresponding TVM TIR variables or buffers.
        # Keys are MLIR values, and values are TVM TIR variables or buffers.
        self.mlir_to_tir_mapping = {}
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
        self.mlir_to_tir_mapping[related_value] = let_var

    def get_operand(self, op, idx):
        return self.get_or_create_var(op.operands[idx])

    def get_or_create_var(self, value):
        if value in self.mlir_to_tir_mapping:
            return self.mlir_to_tir_mapping[value]

        value_type = _get_type(value)
        var = T.Var(self.create_var_name(), value_type)
        self.mlir_to_tir_mapping[value] = var
        return var

    def for_range(self, begin, end, step, kind="serial"):
        self.ib._seq_stack.append([])

        loop_var = T.Var(self.create_var_name(), "int32")
        extent = end if begin == 0 else (end - begin)
        annotations = {"step": step}

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
            self.ib.emit(tir.For(
                loop_var,
                begin,
                extent,
                kind_id,
                self.ib._pop_seq(),
                annotations=annotations,
            ))

        return tir.ir_builder.WithScope(loop_var, _exit_cb)

    def enter_scope(self, scope):
        assert isinstance(scope, tir.ir_builder.WithScope)
        self.scope_stack.append(scope)
        return scope.__enter__()

    def exit_scope(self):
        self.scope_stack.pop().__exit__(None, None, None)

    def dispatch(self, op, stage):
        op_name = "func.func" if isinstance(op, func.FuncOp) else op.name
        # Memref Dialect
        if op_name == "memref.reinterpret_cast":
            self.gen_memref_reinterpret_cast(op)
        elif op_name == "memref.load":
            self.gen_memref_load(op)
        elif op_name == "memref.store":
            self.gen_memref_store(op)
        elif op_name == "memref.alloc":
            self.gen_memref_alloc(op)
        elif op_name == "memref.copy":
            self.gen_memref_copy(op)
        elif op_name == "memref.subview":
            self.gen_memref_subview(op)
        # Arith Dialect
        elif op_name == "arith.constant":
            self.gen_arith_constant(op)
        elif op_name == "arith.index_cast":
            self.gen_arith_index_cast(op)
        elif op_name in ("arith.addf", "arith.addi"):
            self.gen_arith_binary(op, T.Add)
        elif op_name in ("arith.subf", "arith.subi"):
            self.gen_arith_binary(op, T.Sub)
        elif op_name in ("arith.muli", "arith.mulf"):
            self.gen_arith_binary(op, T.Mul)
        elif op_name in ("arith.minsi", "arith.minnumf"):
            self.gen_arith_binary(op, T.Min)
        elif op_name in ("arith.maxsi", "arith.maxnumf"):
            self.gen_arith_binary(op, T.Max)
        elif op_name in ("arith.divf", "arith.divi"):
            self.gen_arith_binary(op, T.Div)
        elif op_name in ("arith.andi", "arith.andf"):
            self.gen_arith_binary(op, T.bitwise_and)
        elif op_name in ("arith.ori", "arith.orf"):
            self.gen_arith_binary(op, T.bitwise_or)
        elif op_name in ("arith.cmpi", "arith.cmpf"):
            self.gen_arith_binary(op, _CMP_MAPPING[op.predicate.value])
        elif op_name in ("arith.sitofp", "arith.extf", "arith.truncf", "arith.extsi", "arith.trunci"):
            self.gen_arith_cast(op)
        # Math Dialect
        elif op_name == "math.exp":
            self.gen_math_exp(op)
        # Func Dialect
        elif op_name == "func.return":
            self.gen_func_return(op)
        elif op_name == "func.func":
            self.gen_func_func(op, stage)
        # Scf Dialect
        elif op_name == "scf.for":
            self.gen_scf_for(op, stage)
        elif op_name == "scf.if":
            self.gen_scf_if(op, stage)
        elif op_name == "scf.yield":
            pass
        # Vector Dialect
        elif op_name == "vector.transfer_read":
            self.gen_vload(op)
        elif op_name == "vector.transfer_write":
            self.gen_vstore(op)
        elif op_name == "vector.broadcast":
            self.gen_vbcast(op)
        # Others
        elif op_name == "builtin.module":
            pass
        else:
            raise RuntimeError(f"Unsupport op {op_name}.")

    def generate(self):
        self.mod.walk(self.dispatch)
        bm = aipu.tir.BuildManager()
        return bm.build(self.prim_func)

    def gen_memref_reinterpret_cast(self, op):
        result = op.result
        arg = self.get_operand(op, 0)
        dtype = _get_type(result).element_type.dtype
        offset = 0
        if len(op.operands) == 2:
            offset = self.get_operand(op, 1)

        buffer = T.Buffer((-1, ), elem_offset=offset, data=arg, dtype=dtype)
        self.mlir_to_tir_mapping[result] = buffer

    def gen_memref_load(self, op):
        result = op.result
        buffer = self.get_operand(op, 0)
        index = 0
        if len(op.operands) == 2:
            index = self.get_operand(op, 1)

        self.emit_let(T.BufferLoad(buffer, [index]), result)

    def gen_memref_store(self, op):
        value = self.get_operand(op, 0)
        buffer = self.get_operand(op, 1)
        index = 0
        if len(op.operands) == 3:
            index = self.get_operand(op, 2)

        self.ib.emit(tir.BufferStore(buffer, value, [index]))

    def gen_memref_alloc(self, op):
        result = op.result
        dtype = _get_type(result).element_type.dtype
        shape = _get_shape(result)

        buf = self.ib.allocate(dtype, shape, scope="lsram")
        self.mlir_to_tir_mapping[result] = buf._buffer

    def gen_memref_copy(self, op):
        src = self.get_operand(op, 0)
        dst = self.get_operand(op, 1)
        width = src.shape[0]

        dma_copy = S.dma_copy(dst, src, width)
        self.ib.emit(dma_copy)

    def gen_memref_subview(self, op):
        result = op.result
        buffer = self.get_operand(op, 0)
        size = self.get_operand(op, 1)

        subview = T.Buffer(size, elem_offset=buffer.elem_offset, data=buffer.data, dtype=buffer.dtype)
        self.mlir_to_tir_mapping[result] = subview

    def gen_arith_constant(self, op):

        def _create_const_expr(op):
            ty = op.result.type
            # scalar
            if isinstance(ty, mlir_ir.IndexType | mlir_ir.IntegerType | mlir_ir.FloatType):
                return tir.const(op.literal_value, _get_type(op.result))
            # vector
            if isinstance(ty, mlir_ir.VectorType):
                if isinstance(ty.element_type, mlir_ir.F32Type):
                    return S.cast(list(mlir_ir.DenseFPElementsAttr(op.value)), _get_type(op.result))
            raise RuntimeError(f"Cannot parse constant {op}")

        expr = _create_const_expr(op)
        self.emit_let(expr, op.result)

    def gen_arith_index_cast(self, op):
        result = op.result
        arg0 = self.get_operand(op, 0)

        self.emit_let(T.Cast("int32", arg0), result)

    def gen_arith_binary(self, op, method):
        result = op.result
        arg0 = self.get_operand(op, 0)
        arg1 = self.get_operand(op, 1)

        self.emit_let(method(arg0, arg1), result)

    def gen_arith_cast(self, op):
        result = op.result
        arg0 = self.get_operand(op, 0)

        self.emit_let(S.cast(arg0, _get_type(result)), result)

    def gen_math_exp(self, op):
        result = op.result
        arg0 = self.get_operand(op, 0)

        self.emit_let(S.exp(arg0), result)

    def gen_func_return(self, op):
        self.ib.emit(T.ret(None))

    def gen_func_func(self, op, stage):
        if stage.is_before_all_regions():
            block = op.regions[0].blocks[0]
            arg_nums = len(block.arguments)
            gridx = block.arguments[arg_nums - 3]
            gridx_var = self.get_or_create_var(gridx)
            self.gridx_var = gridx_var

            tid = T.Add(T.Mul(gridx_var, S.get_local_size()), S.get_local_id())
            self.emit_let(tid, gridx)

        if stage.is_after_all_regions():
            func_name = op.name.value
            block = op.regions[0].blocks[0]
            arg_nums = len(block.arguments)

            args = []
            for i in range(arg_nums):
                if i == arg_nums - 3:
                    args.append(self.gridx_var)
                else:
                    arg = block.arguments[i]
                    var = self.get_or_create_var(arg)
                    args.append(var)

            self.prim_func = tir.PrimFunc(args, self.ib.get()).with_attr("global_symbol", func_name)

    def gen_scf_for(self, op, stage):
        if stage.is_before_all_regions():
            begin = self.get_operand(op, 0)
            end = self.get_operand(op, 1)
            step = self.get_operand(op, 2)

            block = op.regions[0].blocks[0]
            loop_iter = block.arguments[0]

            for_range = self.for_range(begin, end, step)
            loop_var = self.enter_scope(for_range)
            self.mlir_to_tir_mapping[loop_iter] = loop_var

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

    def gen_vload(self, op):
        result = op.result
        buffer = self.get_operand(op, 0)
        index = 0
        if len(op.operands) >= 2:
            index = self.get_operand(op, 1)

        self.emit_let(S.vload(buffer.addr_of(index), lanes=result.type.shape[0]), result)

    def gen_vstore(self, op):
        value = self.get_operand(op, 0)
        buffer = self.get_operand(op, 1)
        index = 0
        if len(op.operands) >= 3:
            index = self.get_operand(op, 2)

        self.ib.emit(S.vstore(value, buffer.addr_of(index)))

    def gen_vbcast(self, op):
        result = op.result
        value = self.get_operand(op, 0)
        dtype = result.type
        self.emit_let(S.vbcast(S.cast(value, value.dtype), lanes=dtype.shape[0]), result)


class AIPUModule:

    def __init__(self, mod):
        # wrap triton module to mlir module
        self.mod = mlir_ir.Module.parse(str(mod), mlir_ir.Context())

    def generic_walk(self, op, callback):
        # operation walk
        stage = WalkStage(op)
        regions = op.regions
        for region in regions:
            callback(op, stage)
            stage.advance()
            for block in region.blocks:
                for nested_op in block.operations:
                    self.generic_walk(nested_op, callback)
        callback(op, stage)

    def walk(self, dispatch):
        # module walk entry
        self.generic_walk(self.mod.operation, dispatch)


def codegenAIPU(mod):
    mod = AIPUModule(mod)
    generator = CodeGenerator(mod)
    return generator.generate()

add_mlir_dialect(LLVMXPUOps llvm_xpu)
# TODO[dyq]: add_mlir_doc permission
# add_mlir_doc(LLVMXPUOps LLVMXPUDialect Dialects/ -gen-dialect-doc -dialect=llvm_xpu)
set(LLVM_TARGET_DEFINITIONS LLVMXPUOps.td)
mlir_tablegen(LLVMXPUConversions.inc -gen-llvmir-conversions)
mlir_tablegen(LLVMXPUFromLLVMIRConversions.inc -gen-intr-from-llvmir-conversions)
mlir_tablegen(LLVMXPUConvertibleLLVMIRIntrinsics.inc -gen-convertible-llvmir-intrinsics)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=llvm_xpu)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=llvm_xpu)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRXPUConversionsIncGen)

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef TRITON_CONVERSION_TRITONXPU_TO_LLVM_TARGETINFOXPU_H
#define TRITON_CONVERSION_TRITONXPU_TO_LLVM_TARGETINFOXPU_H

#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"

namespace mlir {
namespace triton {
namespace xpu {

class TargetInfo : public mlir::triton::TargetInfoBase {
public:
  TargetInfo(uint32_t xpu_arch, uint32_t buffer_size)
      : xpu_arch(xpu_arch), buffer_size(buffer_size) {}

  bool supportMaximumMinimum() const override;

  Value getClusterCTAId(RewriterBase &rewriter, Location loc) const override;

  Value ballot(ConversionPatternRewriter &rewriter, Location loc, Type type,
               Value cmp) const override;

  void storeShared(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
                   Value val, Value pred) const override;

  Value loadShared(ConversionPatternRewriter &rewriter, Location loc,
                   const TypeConverter *converter, Value ptr, Type elemTy,
                   Value pred) const override;

  Value shuffleXor(ConversionPatternRewriter &rewriter, Location loc, Value val,
                   int i) const override;
  Value shuffleUp(ConversionPatternRewriter &rewriter, Location loc, Value val,
                  int i) const override;
  Value shuffleIdx(ConversionPatternRewriter &rewriter, Location loc, Value val,
                   int i) const override;
  Value shuffleIdx(ConversionPatternRewriter &rewriter, Location loc, Value val,
                   Value i) const override;

  Value programId(ConversionPatternRewriter &rewriter, Location loc,
                  ModuleOp moduleOp, int axis) const override;

  bool warpReduce(ConversionPatternRewriter &rewriter, Location loc,
                  SmallVector<Value> &acc, triton::ReduceOp op,
                  unsigned numLaneToReduce) const override;

  bool processReplicaUsingStMatrix(
      ConversionPatternRewriter &rewriter, Location loc, Value smemBase,
      SmallVector<Value> &vals, RankedTensorType srcTy, Type elemTy,
      ArrayRef<unsigned> paddedRepShape, ArrayRef<unsigned> origRepShape,
      ArrayRef<unsigned> outOrd, unsigned accumNumReplicates,
      int swizzleByteWidth) const override;

  std::string getMulhiFuncName(Type resultElementTy) const override;

  void printf(ConversionPatternRewriter &rewriter, Value formatStrStart,
              int formatStrByteCount, ValueRange args) const override;
  void assertFail(ConversionPatternRewriter &rewriter, Location loc,
                  StringRef message, StringRef file, StringRef func,
                  int line) const override;

  uint32_t getXPUArch() const;
  uint32_t getXPUBufferSize() const;

private:
  uint32_t xpu_arch;
  uint32_t buffer_size;
};
} // namespace xpu
} // namespace triton
} // namespace mlir
#endif // TRITON_CONVERSION_TRITONXPU_TO_LLVM_TARGETINFOXPU_H

#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

struct ReturnOpConversion : public ConvertOpToLLVMPattern<triton::ReturnOp> {
  using ConvertOpToLLVMPattern<triton::ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (funcOp->hasAttr("nvvm.kernel") || funcOp->hasAttr("mtgpu.kernel")) {
      // A GPU kernel
      if (op.getNumOperands() > 0) {
        return rewriter.notifyMatchFailure(
            op, "Kernel functions do not support return with operands");
      }
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, TypeRange(), ValueRange(),
                                                  op->getAttrs());
    } else {
      // A device function
      LLVM::ReturnOp newOp;
      if (adaptor.getOperands().size() < 2) {
        // Single or no return value.
        newOp =
            rewriter.create<LLVM::ReturnOp>(op.getLoc(), adaptor.getOperands());
      } else {
        // Pack the results into a struct.
        auto packedResultsTy = this->getTypeConverter()->packFunctionResults(
            funcOp.getResultTypes());
        Value packedResults =
            rewriter.create<LLVM::UndefOp>(op.getLoc(), packedResultsTy);
        auto loc = op.getLoc();
        for (auto it : llvm::enumerate(adaptor.getOperands())) {
          packedResults = insert_val(packedResultsTy, packedResults, it.value(),
                                     it.index());
        }
        newOp = rewriter.create<LLVM::ReturnOp>(op.getLoc(), packedResults);
      }
      newOp->setAttrs(op->getAttrs());
      rewriter.replaceOp(op, newOp->getResults());
    }
    return success();
  }
};

// CallOpInterfaceLowering is adapted from
// https://github.com/llvm/llvm-project/blob/fae656b2dd80246c3c6f01e9c77c49560368752c/mlir/lib/Conversion/FuncToLLVM/FuncToLLVM.cpp#L485
struct CallOpConversion : public ConvertOpToLLVMPattern<triton::CallOp> {
  CallOpConversion(LLVMTypeConverter &converter, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::CallOp>(converter, benefit) {}

  LogicalResult
  matchAndRewrite(triton::CallOp callOp,
                  typename triton::CallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto promotedOperands = promoteOperands(callOp, adaptor, rewriter);
    auto newCallOp =
        convertCallOpToLLVMCallOp(callOp, promotedOperands, rewriter);
    if (!newCallOp)
      return failure();
    auto results = getCallOpResults(callOp, newCallOp, rewriter);
    rewriter.replaceOp(callOp, results);
    return success();
  }

private:
  SmallVector<Value, 4>
  promoteOperands(triton::CallOp callOp,
                  typename triton::CallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const {
    // Get the last argument of the caller, which is the current stack pointer
    // of shared memory and append it to the operands of the callOp.
    auto loc = callOp.getLoc();
    auto caller = callOp->getParentOfType<FunctionOpInterface>();
    auto promotedOperands = this->getTypeConverter()->promoteOperands(
        callOp.getLoc(), /*opOperands=*/callOp->getOperands(),
        adaptor.getOperands(), rewriter);
    if (!caller->hasAttr("allocation.offset")) {
      auto base = LLVM::getStackPointer(rewriter, caller);
      promotedOperands.push_back(base);
      return promotedOperands;
    }
    promotedOperands.push_back(
        LLVM::getSharedMemoryBase(callOp->getLoc(), rewriter, callOp));
    return promotedOperands;
  }

  LLVM::CallOp
  convertCallOpToLLVMCallOp(triton::CallOp callOp,
                            ArrayRef<Value> promotedOperands,
                            ConversionPatternRewriter &rewriter) const {
    // Pack the result types into a struct.
    Type packedResult = nullptr;
    unsigned numResults = callOp.getNumResults();
    auto resultTypes = llvm::to_vector<4>(callOp.getResultTypes());

    if (numResults != 0) {
      if (!(packedResult =
                this->getTypeConverter()->packFunctionResults(resultTypes)))
        return nullptr;
    }
    auto newCallOp = rewriter.create<LLVM::CallOp>(
        callOp.getLoc(), packedResult ? TypeRange(packedResult) : TypeRange(),
        promotedOperands, callOp->getAttrs());
    return newCallOp;
  }

  SmallVector<Value>
  getCallOpResults(triton::CallOp callOp, LLVM::CallOp newCallOp,
                   ConversionPatternRewriter &rewriter) const {
    auto numResults = callOp.getNumResults();
    SmallVector<Value> results;
    if (numResults < 2) {
      // If < 2 results, packing did not do anything and we can just return.
      results.append(newCallOp.result_begin(), newCallOp.result_end());
    } else {
      // Otherwise, it had been converted to an operation producing a structure.
      // Extract individual results from the structure and return them as list.
      results.reserve(numResults);
      for (unsigned i = 0; i < numResults; ++i) {
        results.push_back(rewriter.create<LLVM::ExtractValueOp>(
            callOp.getLoc(), newCallOp->getResult(0), i));
      }
    }
    return results;
  }
};

} // namespace

void mlir::triton::populateControlFlowOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<ReturnOpConversion>(typeConverter, benefit);
  patterns.add<CallOpConversion>(typeConverter, benefit);
}

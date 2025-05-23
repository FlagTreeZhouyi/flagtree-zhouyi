// Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TRITONNVIDIAGPU_DIALECT
#define TRITONNVIDIAGPU_DIALECT

include "mlir/IR/OpBase.td"

def TritonNvidiaGPU_Dialect : Dialect {
  let name = "ttng";

  let cppNamespace = "::mlir::triton::nvidia_gpu";

  let hasOperationAttrVerify = 1;

  let description = [{
    Triton Nvidia GPU Dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
    "triton::gpu::TritonGPUDialect",
    "mlir::gpu::GPUDialect",
  ];

  let extraClassDeclaration = [{
    static std::string getNumWarpsAttrName() { return "ttg.num-warps"; }
    static int getNumWarps(ModuleOp mod) {
      if(!mod->hasAttr("ttg.num-warps"))
        llvm::report_fatal_error(
            "TritonGPU module should contain a ttg.num-warps attribute");
      return cast<IntegerAttr>(mod->getAttr("ttg.num-warps")).getInt();
    }
    static int getNumCTAs(ModuleOp mod) {
      if(!mod->hasAttr("ttg.num-ctas"))
        llvm::report_fatal_error(
            "TritonGPU module should contain a ttg.num-ctas attribute");
      return cast<IntegerAttr>(mod->getAttr("ttg.num-ctas")).getInt();
    }
    void registerTypes();
  }];
  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
  let useDefaultTypePrinterParser = 1;
}

include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUTypes.td"

#endif

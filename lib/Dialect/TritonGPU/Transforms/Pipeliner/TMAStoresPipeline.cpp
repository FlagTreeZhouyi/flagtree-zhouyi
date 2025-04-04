#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

struct TMAStore {
  Operation *op;
  mlir::TypedValue<tt::TensorDescType> desc;
  mlir::TypedValue<RankedTensorType> src;
};

static SmallVector<TMAStore> getTMAStores(scf::ForOp forOp) {
  SmallVector<TMAStore> tmaStores;

  forOp.getBody()->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (auto storeOp = dyn_cast<tt::ExperimentalDescriptorStoreOp>(op)) {
      tmaStores.push_back({storeOp, storeOp.getDesc(), storeOp.getSrc()});
    } else if (auto scatterOp =
                   dyn_cast<tt::ExperimentalDescriptorScatterOp>(op)) {
      tmaStores.push_back({scatterOp, scatterOp.getDesc(), scatterOp.getSrc()});

      // Don't walk into nested loops.
    } else if (isa<scf::ForOp>(op)) {
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });

  return tmaStores;
}

static Value createAlloc(scf::ForOp &forOp, const TMAStore &store) {
  OpBuilderWithAsyncTaskIds builder(forOp);
  RankedTensorType ty = store.src.getType();
  auto order = ttg::getOrder(ty);
  auto ctaLayout = ttg::getCTALayout(ty.getEncoding());
  Attribute encoding = ttg::SwizzledSharedEncodingAttr::get(
      ty.getContext(), 1, 1, 1, order, ctaLayout);
  if (ty.getRank() > 1) {
    encoding = ttg::NVMMASharedEncodingAttr::get(
        ty.getContext(), ty.getShape(), order, ctaLayout, ty.getElementType(),
        /*fp4Padded*/ false);
  }
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(ty.getContext());
  Type memdescType =
      ttg::MemDescType::get(ty.getShape(), ty.getElementType(), encoding,
                            sharedMemorySpace, /*mutableMemory*/ true);
  Value alloc = builder.createWithAsyncTaskIds<ttg::LocalAllocOp>(
      store.op->getLoc(), memdescType);
  return alloc;
}

static void createTMAAsyncCopy(scf::ForOp forOp, const TMAStore &store,
                               Value alloc) {
  OpBuilderWithAsyncTaskIds builder(store.op);
  Location loc = store.op->getLoc();
  RankedTensorType ty = store.src.getType();
  auto order = ttg::getOrder(ty);
  auto ctaLayout = ttg::getCTALayout(ty.getEncoding());

  // Put wait before the local_store make the store truly async. We know
  // that we are the only user of the CopyLocalToGlobal.
  builder.createWithAsyncTaskIds<ttng::TMAStoreWaitOp>(loc, 0);
  builder.createWithAsyncTaskIds<ttg::LocalStoreOp>(loc, store.src, alloc);
  builder.createWithAsyncTaskIds<ttng::FenceAsyncSharedOp>(loc, false);
  Value tmaPtr =
      builder.createWithAsyncTaskIds<triton::nvidia_gpu::TensorDescToTMAPtrOp>(
          loc, store.desc);
  if (auto storeOp = dyn_cast<tt::ExperimentalDescriptorStoreOp>(store.op)) {
    builder.createWithAsyncTaskIds<ttng::AsyncTMACopyLocalToGlobalOp>(
        loc, tmaPtr, storeOp.getIndices(), alloc);
  } else {
    auto scatterOp = cast<tt::ExperimentalDescriptorScatterOp>(store.op);
    builder.createWithAsyncTaskIds<ttng::AsyncTMAScatterOp>(
        loc, tmaPtr, scatterOp.getXOffsets(), scatterOp.getYOffset(), alloc);
  }

  store.op->erase();
}

bool mlir::triton::pipelineTMAStores(scf::ForOp forOp) {
  SmallVector<TMAStore> tmaStores = getTMAStores(forOp);
  if (tmaStores.empty())
    return false;

  DenseMap<Operation *, Value> storeToAlloc;
  DenseMap<std::pair<ArrayRef<int64_t>, Type>, Value> allocs;
  for (const TMAStore &store : tmaStores) {
    // Reuse allocations for stores of the same shape and types. This allows
    // saving shared memory usage. It is valid since we have a wait 0 before
    // every local_store. We could pipeline more aggressively if we didn't
    // reuse but there is a tradeoff with shared memory usage.
    RankedTensorType srcTy = store.src.getType();
    auto key = std::make_pair(srcTy.getShape(), srcTy.getElementType());
    Value &alloc = allocs[key];
    if (!alloc) {
      alloc = createAlloc(forOp, store);
    }
    storeToAlloc[store.op] = alloc;
  }

  for (const TMAStore &store : tmaStores) {
    createTMAAsyncCopy(forOp, store, storeToAlloc[store.op]);
  }

  // Deallocate shared memory buffers.
  OpBuilder builder(forOp);
  builder.setInsertionPointAfter(forOp);
  builder.create<ttng::TMAStoreWaitOp>(forOp->getLoc(), 0);
  for (auto it : storeToAlloc) {
    builder.create<ttg::LocalDeallocOp>(forOp->getLoc(), it.second);
  }
  return true;
}

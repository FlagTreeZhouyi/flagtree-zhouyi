add_triton_library(TritonAMDGPUTransforms
  AccelerateAMDMatmul.cpp
  BlockPingpong.cpp
  CanonicalizePointers.cpp
  ConvertToBufferOps.cpp
  OptimizeEpilogue.cpp
  HoistLayoutConversions.cpp
  ReorderInstructions.cpp
  StreamPipeline.cpp
  MfmaGroup.cpp

  DEPENDS
  TritonAMDGPUIR
  TritonAMDGPUTransformsIncGen
  TritonGPUIR
  TritonAMDUtils
  TritonAMDAnalysis
)

target_include_directories(TritonAMDGPUTransforms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(TritonAMDGPUTransforms PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../../include)

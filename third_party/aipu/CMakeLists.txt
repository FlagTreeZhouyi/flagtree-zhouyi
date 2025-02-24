#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
#include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
#add_subdirectory(include)
#add_subdirectory(lib)
if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonAIPU ${CMAKE_CURRENT_SOURCE_DIR}/triton_aipu.cc)
  target_link_libraries(TritonAIPU PRIVATE
    Python3::Module
    pybind11::headers
    MLIRLinalgUtils
    MLIRLinalgToStandard
    MLIRBufferizationTransforms
    MLIRBufferizationToMemRef
    MLIRArithTransforms
    MLIRFuncAllExtensions
    MLIRAffineToStandard
  )
  target_include_directories(TritonAIPU PRIVATE "${TRITON_SHARED_DIR}/include")
endif()
#if(TRITON_BUILD_UT)
#  add_subdirectory(unittest)
#endif()

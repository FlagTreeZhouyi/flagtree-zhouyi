add_triton_plugin(TritonAIPU ${CMAKE_CURRENT_SOURCE_DIR}/triton_aipu.cc)
target_include_directories(TritonAIPU PRIVATE ${CMAKE_SOURCE_DIR}/third_party/triton_shared/include)
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
    MLIRSCFTransforms
    MLIRAffineTransforms
)

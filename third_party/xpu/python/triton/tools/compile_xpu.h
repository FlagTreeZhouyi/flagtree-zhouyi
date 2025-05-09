#ifndef TT_KERNEL_INCLUDES
#define TT_KERNEL_INCLUDES

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#define CUresult int
#define CUstream XPUStream
#define CUdeviceptr int64_t

#endif

void unload_{kernel_name}(void);
void load_{kernel_name}(void);
// tt-linker: {kernel_name}:{full_signature}:{algo_info}
CUresult{_placeholder} {kernel_name}(CUstream stream, {signature});

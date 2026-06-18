#pragma once
#include "zhilight_cuda_to_hip.h"
// hip_bf16.h is clang-only (clang elementwise builtins); host g++ TUs that
// include <cuda_bf16.h> never use the bf16 type, so skip it there.
#if defined(__clang__)
#include <hip/hip_bf16.h>
#endif

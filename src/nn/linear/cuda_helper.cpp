#include <bmengine/core/exception.h>
#include <cuda.h>

#if defined(USE_HIP)
// gfx90a/CDNA has no programmable L2 access-policy window; this is a no-op on
// AMD (the only caller is commented out at its call sites in linear.cpp).
void setL2AccessPolicyWindow(cudaStream_t stream, void* data, int window_size, float hitRatio) {}
#else
// https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#l2-cache-access-window
void setL2AccessPolicyWindow(cudaStream_t stream, void* data, int window_size, float hitRatio) {
    cudaStreamAttrValue stream_attribute;                                         // Stream level attributes data structure
    stream_attribute.accessPolicyWindow.base_ptr  = data;                         // Global Memory data pointer
    stream_attribute.accessPolicyWindow.num_bytes = window_size;                  // Number of bytes for persisting accesses.
    // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
    stream_attribute.accessPolicyWindow.hitRatio  = 1.0;                          // Hint for L2 cache hit ratio for persisting accesses in the num_bytes region
    stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
    stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

    //Set the attributes to a CUDA stream of type cudaStream_t
    BM_CUDART_ASSERT(cudaStreamSetAttribute(
        stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute));
//    if (window_size == 0)
//        BM_CUDART_ASSERT(cudaCtxResetPersistingL2Cache());
}
#endif

#include "generator/generator.h"
#include "model/allocate_util.hpp"
#include "utils/exception.h"
#include "utils/print.h"
#include <bmengine/functions/typecast.h>
#include <bmengine/functions/softmax.h>
#include <bmengine/functions/reduce.cuh>
#include <stack>
#include <algorithm>
#include <unordered_set>
#include <cub/cub.cuh>

namespace beam_utility {

// rocPRIM (hipCUB) radix sort lacks a built-in codec for __hip_bfloat16 keys
// (half and float are fine). bf16 top-p sampling is not on the validated path,
// so on HIP its cub::DeviceRadixSort instantiation is skipped (would fail to
// compile via rocPRIM's identity_decomposer assertion) and the call throws at
// runtime. CUDA keeps the original cub call for every dtype.
template<typename KeyT>
struct HipBf16Key { static constexpr bool value = false; };
#if defined(USE_HIP)
template<> struct HipBf16Key<nv_bfloat16> { static constexpr bool value = true; };
#endif

template<typename KeyT, typename ValueT>
static inline hipError_t radixSortPairsDescending(
    void* d_temp, size_t& temp_bytes,
    const KeyT* keys_in, KeyT* keys_out,
    const ValueT* vals_in, ValueT* vals_out,
    int num_items, int begin_bit, int end_bit, hipStream_t stream) {
    if constexpr (HipBf16Key<KeyT>::value) {
        BM_EXCEPTION("top-p sampling with bfloat16 probs is not supported on ROCm/HIP "
                     "(rocPRIM radix sort has no bf16 key codec). Use float/half probs.");
        return hipSuccess;
    } else {
        return cub::DeviceRadixSort::SortPairsDescending(
            d_temp, temp_bytes, keys_in, keys_out, vals_in, vals_out,
            num_items, begin_bit, end_bit, stream);
    }
}
// gridDim (n / 1024, 1, 1),    blockDim(1024, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(random_repetition_penalty)(
    int n,
    int vocab_size,
    const float* __restrict__ value,
    const int32_t* __restrict__ tokens,
    const int32_t* __restrict__ batch_id,
    T* __restrict__ logits // (batch_size, vocab_size)
) {
    int pos = threadIdx.x + blockDim.x * blockIdx.x;
    if (pos < n) {
        int32_t token = tokens[pos];
        int32_t batch_idx = batch_id[pos];

        T l = logits[batch_idx * vocab_size + token];
        l = (l < T(0.)) ? (l * T(value[pos])) : (l / T(value[pos]));
        logits[batch_idx * vocab_size + token] = l;
    }
}

void random_repetition_penalty(
    const core::Context& ctx,
    const std::vector<float>& penalty_factor,
    const std::vector<int32_t>& tokens,
    const std::vector<int32_t>& batch_id,
    core::Tensor& logits) {
    BM_ASSERT(tokens.size() == batch_id.size(), "tokens and batch_id must have the same size");
    BM_ASSERT(tokens.size() > 0, "tokens and batch_id must have at least one element");
    auto device = logits.device();
    int n = tokens.size();
    int vocab_size = logits.size(1);

    auto d_factor = ctx.tensor({ penalty_factor.size() }, core::DataType::kFloat);
    auto d_tokens = ctx.tensor({ tokens.size() }, core::DataType::kInt32);
    auto d_batch = ctx.tensor({ batch_id.size() }, core::DataType::kInt32);
    d_factor.from_buffer(penalty_factor.data());
    d_tokens.from_buffer(tokens.data());
    d_batch.from_buffer(batch_id.data());

    int threads = round_up(min(n, 1024), 32);
    dim3 gridDim(round_up(n, threads) / threads, 1, 1);
    dim3 blockDim(threads, 1, 1);
    auto stream = ctx.current_stream()->ptr;

    BM_DTYPE_DISPATCH_FLOAT(logits.dtype(), {
        BM_KERNEL(random_repetition_penalty)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            n,
            vocab_size,
            d_factor.data<float>(),
            d_tokens.data<int32_t>(),
            d_batch.data<int32_t>(),
            logits.mutable_data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

__global__ void BM_KERNEL(arange)(
    int32_t length,
    int32_t* out // (n, len)
) {
    int32_t offset = blockIdx.y * blockDim.x + threadIdx.x;
    if (offset < length) {
        out[blockIdx.x * length + offset] = offset;
    }
}

// gridDim(1, 1, 1),   blockDim(1, 1, 1)
template<typename T>
__global__ void BM_KERNEL(random_sampler_gpu)(
    int num_classes,
    T* probs_cum,      // (num_classes)
    int32_t* indicies, // (num_classes)
    int32_t* select,
    float* ptr_p,
    float top_p,
    int top_k) {
    if (top_k > 0) {
        top_p = min((float) probs_cum[top_k - 1], top_p);
    }
    T v_p = ptr_p[0] * top_p * (float) probs_cum[num_classes - 1];
    int lf = -1;
    int rt = num_classes - 1;
    while (lf + 1 < rt) {
        int mid = (lf + rt) / 2;
        if (probs_cum[mid] < v_p) {
            lf = mid;
        } else {
            rt = mid;
        }
    }
    select[0] = indicies[rt];
}

/*
  samples i from \sum{0}^{i}P(i) <= p, with p ~ U(0, top_p).
  P is descendly sorted and cumulated, then do a binary search.
*/
void random_sampler_gpu(
    const core::Context& ctx,
    curandGenerator_t& gen,
    core::Tensor& probs,  // (..., n_classes)
    core::Tensor& select, // (...)
    float top_p = 1.0f,
    int top_k = 0,
    int num_samples = 1) {
    unsigned int n_classes = probs.size(probs.ndim() - 1);
    BM_ASSERT(top_p <= 1.0f && top_p >= 0.0f, "top_p must be in [0, 1]");
    BM_ASSERT(top_k >= 0 && top_k < probs.size(-1), "invalid top k");
    BM_ASSERT_EQ(select.size(0), probs.numel() / n_classes * num_samples, "invalid select size");
    unsigned int batch = probs.numel() / n_classes;
    unsigned int select_step = select.size(0) / batch;
    auto stream = ctx.current_stream()->ptr;

    core::Tensor indicies_in = ctx.tensor({ n_classes }, core::DataType::kInt32);
    core::Tensor indicies_out = ctx.tensor({ n_classes }, core::DataType::kInt32);
    core::Tensor values_out = ctx.tensor({ n_classes }, probs.dtype());
    {
        int threads = min(round_up(n_classes, 32), 1024);
        dim3 gridDim(1, round_up(n_classes, threads) / threads, 1);
        dim3 blockDim(threads, 1, 1);

        BM_KERNEL(arange)<<<gridDim, blockDim, 0, stream>>>(n_classes, indicies_in.data<int32_t>());
        BM_CUDART_ASSERT(cudaGetLastError());
    }

    BM_DTYPE_DISPATCH_FLOAT(probs.dtype(), {
        size_t temp_buffer_size1, temp_buffer_size2;
        radixSortPairsDescending(
            (void*) nullptr,
            temp_buffer_size1,
            values_out.data<scalar_t>(),
            values_out.data<scalar_t>(),
            indicies_in.data<int32_t>(),
            indicies_out.data<int32_t>(),
            n_classes, 0, sizeof(scalar_t) * 8, (hipStream_t) 0);
        cub::DeviceScan::InclusiveSum(
            nullptr,
            temp_buffer_size2,
            values_out.data<scalar_t>(),
            values_out.data<scalar_t>(),
            n_classes);
        size_t temp_buffer_size = max(temp_buffer_size1, temp_buffer_size2);
        auto temp = ctx.tensor({ temp_buffer_size }, core::DataType::kInt8);

        core::Tensor p_random =
            ctx.tensor({ (size_t) (batch * select_step) }, core::DataType::kFloat);
        CURAND_CHECK(curandGenerateUniform(gen, p_random.data<float>(), p_random.size(0)));
        for (int i = 0; i < batch; i++) {
            scalar_t* offset_prob = (probs.data<scalar_t>() + i * n_classes);
            BM_CUDART_ASSERT(radixSortPairsDescending(
                temp.data(),
                temp_buffer_size,
                offset_prob,
                values_out.data<scalar_t>(),
                indicies_in.data<int32_t>(),
                indicies_out.data<int32_t>(),
                n_classes,
                0,
                sizeof(scalar_t) * 8,
                stream));
            BM_CUDART_ASSERT(cub::DeviceScan::InclusiveSum(
                temp.data(),
                temp_buffer_size,
                values_out.data<scalar_t>(),
                offset_prob,
                n_classes,
                stream));

            dim3 gridDim(1, 1, 1);
            dim3 blockDim(1, 1, 1);
            for (int j = 0; j < select_step; j++) {
                BM_KERNEL(random_sampler_gpu)<<<gridDim, blockDim, 0, stream>>>(
                    n_classes,
                    offset_prob,
                    indicies_out.data<int32_t>(),
                    select.data<int32_t>() + (i * select_step + j),
                    p_random.data<float>() + (i * select_step + j),
                    top_p,
                    top_k);
                BM_CUDART_ASSERT(cudaGetLastError());
            }
        }
    });
}

}

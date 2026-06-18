#pragma once
#include "zhilight_cuda_to_hip.h"
#include <hiprand/hiprand.h>

#define curandGenerator_t        hiprandGenerator_t
#define curandStatus_t           hiprandStatus_t
#define curandRngType_t          hiprandRngType_t
#define curandOrdering_t         hiprandOrdering_t

#define CURAND_STATUS_SUCCESS    HIPRAND_STATUS_SUCCESS
#define CURAND_RNG_PSEUDO_MRG       HIPRAND_RNG_PSEUDO_MRG32K3A
#define CURAND_RNG_PSEUDO_MRG32K3A  HIPRAND_RNG_PSEUDO_MRG32K3A
#define CURAND_ORDERING_PSEUDO_BEST HIPRAND_ORDERING_PSEUDO_BEST

#define curandCreateGenerator    hiprandCreateGenerator
#define curandDestroyGenerator   hiprandDestroyGenerator
#define curandGenerateNormal     hiprandGenerateNormal
#define curandGenerateUniform    hiprandGenerateUniform
#define curandSetGeneratorOffset hiprandSetGeneratorOffset
#define curandSetGeneratorOrdering hiprandSetGeneratorOrdering
#define curandSetPseudoRandomGeneratorSeed hiprandSetPseudoRandomGeneratorSeed
#define curandSetStream          hiprandSetStream

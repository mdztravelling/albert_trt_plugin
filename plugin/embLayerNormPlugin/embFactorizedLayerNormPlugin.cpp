/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda.h>
#if CUDA_VERSION >= 10010

#include <cassert>
#include <cstring>
#include <vector>
#include <stdio.h>

#include "NvInfer.h"
#include "embFactorizedLayerNormPlugin.h"
#include "serialize.hpp"

using namespace nvinfer1;

namespace bert
{
namespace
{
const char* EMB_LAYER_NORM_VERSION{"1"};
const char* EMB_LAYER_NORM_NAME{"CustomEmbFactorizedLayerNormPluginDynamic"};
} // namespace

// Static class fields initialization
PluginFieldCollection EmbFactorizedLayerNormPluginDynamicCreator::mFC{};
std::vector<PluginField> EmbFactorizedLayerNormPluginDynamicCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(EmbFactorizedLayerNormPluginDynamicCreator);

// tune BatchedGemmEx
template<typename T>
int tuneBatchedGemmEx(const int sm_version, const int B, const int S, const int embedding_size, const int headSize) {
    const int nruns = 500;
    cublasHandle_t cublas;
    cublasCreate(&cublas);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cublasSetStream(cublas, stream);
    cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH);

    const int m = S;
    const int n = headSize;
    const int k = embedding_size;

    const size_t dABytes = B * m * k * sizeof(T);
    const size_t dBBytes = k * n * sizeof(T);
    const size_t dCBytes = B * m * n * sizeof(T);

    T* dA = nullptr;
    T* dB = nullptr;
    T* dC = nullptr;
    cudaMalloc(&dA, dABytes);
    cudaMalloc(&dB, dBBytes);
    cudaMalloc(&dC, dCBytes);
    cudaMemset(dA, 1, dABytes);
    cudaMemset(dB, 1, dBBytes);

    const int startAlgo = (int) CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    const int endAlgo = (int) CUBLAS_GEMM_ALGO15_TENSOR_OP;
    int best_algo = startAlgo;
    float min_cost_ms = 1000000.0f;

    ASSERT(sm_version >= kSM_53);
    for (int a = startAlgo; a <= endAlgo; a++)
    {
        cublasGemmAlgo_t algo = static_cast<cublasGemmAlgo_t>(a);
        float cost_ms = 0.0f;
        cudaEventRecord(start, stream);
        for (int r = 0; r < nruns; r++)
        {
            CUBLASASSERT(cublasGemmStridedBatchedEx<T>(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, T(1.f),
                                                       dB, n, 0,
                                                       dA, k, k,
                                                       T(0.0f),
                                                       dC, n, n,
                                                       B, algo));
        }
        cudaEventRecord(stop, stream);
        cudaStreamSynchronize(stream);
        cudaEventElapsedTime(&cost_ms, start, stop);
        if (cost_ms < min_cost_ms)
        {
            best_algo = algo;
            min_cost_ms = cost_ms;
        }
    }
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cublasDestroy(cublas);
    return best_algo;
}

EmbFactorizedLayerNormPluginDynamic::EmbFactorizedLayerNormPluginDynamic(const std::string& name, const DataType type,
    const DataType mhaType, const Weights& beta, const Weights& gamma, const Weights& wordEmb, const Weights& posEmb,
    const Weights& tokEmb, const Weights& wordEmb2, const bool useFullMask)
    : mLayerName(name)
    , mLd(beta.count)
    , mType(type)
    , mUseFullMask(useFullMask)
    , mMhaType(mhaType)
    , mAlgoBatchedEx(CUBLAS_GEMM_DEFAULT_TENSOR_OP)
{

    // Assuming Weights.count is the number of elements and not bytes
    assert(beta.count == gamma.count);
    mSM = getSMVersion();
    // mS is set during configure
    mWordEmbCount = wordEmb.count;
    mWordEmb2Count = wordEmb2.count;
    mTokEmbCount = tokEmb.count;
    mPosEmbCount = posEmb.count;

    gLogVerbose << "EmbFactorizedLayerNormPluginDynamic Construct, [beta.count:" << beta.count << "][wordEmb.count:"
                << wordEmb.count << "][posEmb.count:" << posEmb.count << "][tokEmb.count:" << tokEmb.count
                << "][wordEmb2.count:" << wordEmb2.count << "]" << std::endl;

    mBeta.convertAndCopy(beta, nvinfer1::DataType::kFLOAT);
    mGamma.convertAndCopy(gamma, nvinfer1::DataType::kFLOAT);
    mWordEmb.convertAndCopy(wordEmb, mType);
    mTokEmb.convertAndCopy(tokEmb, mType);
    mPosEmb.convertAndCopy(posEmb, mType);
    mWordEmb2.convertAndCopy(wordEmb2, mType);

    copyToDevice(mGamma, sizeof(float) * mGamma.count, mGammaDev);
    copyToDevice(mBeta, sizeof(float) * mBeta.count, mBetaDev);
    copyToDevice(mWordEmb, getWeightsSize(mWordEmb, mType), mWordEmbDev);
    copyToDevice(mPosEmb, getWeightsSize(mPosEmb, mType), mPosEmbDev);
    copyToDevice(mTokEmb, getWeightsSize(mTokEmb, mType), mTokEmbDev);
    copyToDevice(mWordEmb2, getWeightsSize(mWordEmb2, mType), mWordEmb2Dev);

    CUBLASASSERT(cublasCreate(&mCublas));
}

EmbFactorizedLayerNormPluginDynamic::EmbFactorizedLayerNormPluginDynamic(const std::string& name, const void* data, size_t length)
    : mLayerName(name)
    , mGammaDev(nullptr)
    , mBetaDev(nullptr)
    , mWordEmbDev(nullptr)
    , mTokEmbDev(nullptr)
    , mPosEmbDev(nullptr)
    , mWordEmb2Dev(nullptr)
{
    gLogVerbose << "EmbFactorizedLayerNormPluginDynamic deserialize." << std::endl;

    // Deserialize in the same order as serialization
    deserialize_value(&data, &length, &mType);
    deserialize_value(&data, &length, &mMhaType);
    deserialize_value(&data, &length, &mLd);
    deserialize_value(&data, &length, &mS);

    deserialize_value(&data, &length, &mWordEmbCount);
    deserialize_value(&data, &length, &mWordEmb2Count);
    deserialize_value(&data, &length, &mTokEmbCount);
    deserialize_value(&data, &length, &mPosEmbCount);

    deserialize_value(&data, &length, &mUseFullMask);
    deserialize_value(&data, &length, &mSM);
    deserialize_value(&data, &length, &mAlgoBatchedEx);

    const char* d = static_cast<const char*>(data);
    mBeta.convertAndCopy(d, mLd, nvinfer1::DataType::kFLOAT);
    mGamma.convertAndCopy(d, mLd, nvinfer1::DataType::kFLOAT);

    mWordEmb.convertAndCopy(d, mWordEmbCount, mType);
    mPosEmb.convertAndCopy(d, mPosEmbCount, mType);
    mTokEmb.convertAndCopy(d, mTokEmbCount, mType);
    mWordEmb2.convertAndCopy(d, mWordEmb2Count, mType);

    copyToDevice(mGamma, sizeof(float) * mGamma.count, mGammaDev);
    copyToDevice(mBeta, sizeof(float) * mBeta.count, mBetaDev);

    copyToDevice(mWordEmb, getWeightsSize(mWordEmb, mType), mWordEmbDev);
    copyToDevice(mPosEmb, getWeightsSize(mPosEmb, mType), mPosEmbDev);
    copyToDevice(mTokEmb, getWeightsSize(mTokEmb, mType), mTokEmbDev);
    copyToDevice(mWordEmb2, getWeightsSize(mWordEmb2, mType), mWordEmb2Dev);

    CUBLASASSERT(cublasCreate(&mCublas));
}

// IPluginV2DynamicExt Methods
IPluginV2DynamicExt* EmbFactorizedLayerNormPluginDynamic::clone() const noexcept
{
    try
    {
        gLogVerbose << "EmbFactorizedLayerNormPluginDynamic clone." << std::endl;

        auto p = new EmbFactorizedLayerNormPluginDynamic(
            mLayerName, mType, mMhaType, mBeta, mGamma, mWordEmb, mPosEmb, mTokEmb, mWordEmb2, mUseFullMask);
        p->mS = mS;
        p->setPluginNamespace(mNamespace.c_str());

        return p;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

DimsExprs EmbFactorizedLayerNormPluginDynamic::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        // Input should be input ids and token ids and the input mask
        // Output should be the embeddings tensor and mask indices
        assert(nbInputs == 3);

        assert(inputs[0].nbDims == 2); // BxS
        assert(inputs[0].nbDims == inputs[1].nbDims);
        assert(inputs[0].nbDims == inputs[2].nbDims);

        assert(outputIndex == 0 || outputIndex == 1);

        if (outputIndex == 0)
        {
            DimsExprs ret;
            ret.nbDims = 5;
            ret.d[0] = inputs[0].d[0];
            ret.d[1] = inputs[0].d[1];
            ret.d[2] = exprBuilder.constant(mLd);
            //ret.d[2] = exprBuilder.constant(128);
            ret.d[3] = exprBuilder.constant(1);
            ret.d[4] = exprBuilder.constant(1);
            return ret;
        }

        DimsExprs ret;
        ret.nbDims = 2;
        ret.d[0] = inputs[0].d[BDIM];
        auto cms0 = exprBuilder.constant(unfusedMaskSize);

        // this code must match getMHAMaskPackedSize in bertCommon.h
        bool isSmOK = (mSM == kSM_75 || mSM == kSM_80 || mSM == kSM_86);
        bool isPrecisionOK = (mMhaType == nvinfer1::DataType::kHALF || mMhaType == nvinfer1::DataType::kINT8);
        if (mUseFullMask || (isSmOK && isPrecisionOK))
        {
            // support 128, 384 in both int8 and fp16
            auto cms128 = exprBuilder.constant(packedMaskSize128);
            auto cms384 = exprBuilder.constant(packedMaskSize384);
            auto c128 = exprBuilder.constant(128);
            auto c384 = exprBuilder.constant(384);
            auto is128 = exprBuilder.operation(DimensionOperation::kEQUAL, *inputs[0].d[SDIM], *c128);
            auto is384 = exprBuilder.operation(DimensionOperation::kEQUAL, *inputs[0].d[SDIM], *c384);
            auto sel128 = exprBuilder.operation(DimensionOperation::kPROD, *is128, *cms128);
            auto sel384 = exprBuilder.operation(DimensionOperation::kPROD, *is384, *cms384);
            auto maskSize = exprBuilder.operation(DimensionOperation::kSUM, *sel384, *sel128);

            if (mMhaType == nvinfer1::DataType::kHALF)
            {
                // support 64, 96 only in fp16
                auto cms64 = exprBuilder.constant(packedMaskSize64);
                auto cms96 = exprBuilder.constant(packedMaskSize96);
                auto c64 = exprBuilder.constant(64);
                auto c96 = exprBuilder.constant(96);

                auto is64 = exprBuilder.operation(DimensionOperation::kEQUAL, *inputs[0].d[SDIM], *c64);
                auto is96 = exprBuilder.operation(DimensionOperation::kEQUAL, *inputs[0].d[SDIM], *c96);
                auto sel64 = exprBuilder.operation(DimensionOperation::kPROD, *is64, *cms64);
                auto sel96 = exprBuilder.operation(DimensionOperation::kPROD, *is96, *cms96);
                auto maskSize2 = exprBuilder.operation(DimensionOperation::kSUM, *sel64, *sel96);
                maskSize = exprBuilder.operation(DimensionOperation::kSUM, *maskSize, *maskSize2);
            }

            auto is0 = exprBuilder.operation(DimensionOperation::kEQUAL, *maskSize, *exprBuilder.constant(0));
            auto sel0 = exprBuilder.operation(DimensionOperation::kPROD, *is0, *cms0);
            auto combinedMaskSize = exprBuilder.operation(DimensionOperation::kSUM, *maskSize, *sel0);
            ret.d[1] = combinedMaskSize;
        }
        else
        {
            ret.d[1] = cms0;
        }

        return ret;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool EmbFactorizedLayerNormPluginDynamic::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    // 3 inputs of size BxS
    assert(nbInputs == 3);
    assert(nbOutputs == 2);

    const PluginTensorDesc& desc = inOut[pos];
    if (desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }
    if (pos == 0)
    {
        return desc.type == DataType::kINT32 && desc.dims.nbDims == 2;
    }

    const PluginTensorDesc& prev = inOut[pos - 1];
    if (pos == 1 || pos == 2)
    {
        return desc.type == DataType::kINT32 && desc.dims.nbDims == 2 && desc.dims.d[BDIM] == prev.dims.d[BDIM]
            && desc.dims.d[SDIM] == prev.dims.d[SDIM];
    }

    // embedded sequence
    if (pos == 3)
    {
        return desc.type == mType && desc.dims.nbDims == 5 && desc.dims.d[BDIM] == prev.dims.d[BDIM]
            && desc.dims.d[SDIM] == prev.dims.d[SDIM] && desc.dims.d[3] == 1 && desc.dims.d[4] == 1;
    }
    // mask
    return desc.type == DataType::kFLOAT;
}

void EmbFactorizedLayerNormPluginDynamic::configurePlugin(
    const DynamicPluginTensorDesc* inputs, int nbInputs, const DynamicPluginTensorDesc* outputs, int nbOutputs) noexcept
{
    gLogVerbose << "EmbFactorizedLayerNormPluginDynamic configurePlugin." << std::endl;

    // Validate input arguments
    assert(nbOutputs == 2);
    assert(nbInputs == 3);

    assert(inputs[0].desc.dims.nbDims == 2);
    const int S = inputs[0].desc.dims.d[SDIM];
    mS = S;
    const int B = inputs[0].desc.dims.d[BDIM];
    TRT_UNUSED B;
    assert(mS == static_cast<size_t>(inputs[1].desc.dims.d[SDIM]));
    assert(B == inputs[1].desc.dims.d[BDIM]);
    assert(mS == static_cast<size_t>(inputs[2].desc.dims.d[SDIM]));
    assert(B == inputs[2].desc.dims.d[BDIM]);

    assert(outputs[0].desc.dims.nbDims == 5);
    assert(static_cast<size_t>(outputs[0].desc.dims.d[SDIM]) == mS);
    assert(outputs[0].desc.dims.d[BDIM] == B);
    assert(static_cast<size_t>(outputs[0].desc.dims.d[2]) == mLd);
    assert(outputs[0].desc.dims.d[3] == 1);
    assert(outputs[0].desc.dims.d[4] == 1);

    if (mUseFullMask)
    {
        // user force full_mask
        assert(outputs[1].desc.dims.nbDims == 2);
        assert(outputs[1].desc.dims.d[0] == B);
        assert((outputs[1].desc.dims.d[1] == -1) || (outputs[1].desc.dims.d[1] == packedMaskSize384)
            || (outputs[1].desc.dims.d[1] == packedMaskSize128));
    }
    else
    {
        // auto detect using mhatype
        if (S != -1 && B != -1)
        {
            assert(outputs[1].desc.dims.nbDims == 2);
            assert(outputs[1].desc.dims.d[0] == B);
            int packedSize = getMHAMaskPackedSize(mSM, mMhaType, S);
            TRT_UNUSED packedSize;
            assert(outputs[1].desc.dims.d[1] == -1 || outputs[1].desc.dims.d[1] == packedSize);
        }
    }

    assert(inputs[0].desc.type == DataType::kINT32);
    assert(inputs[1].desc.type == DataType::kINT32);
    assert(inputs[2].desc.type == DataType::kINT32);
    assert(outputs[0].desc.type == mType);
    assert(outputs[1].desc.type == DataType::kFLOAT);

    // tune batched Gemm
    if (mType == DataType::kHALF) {
      mAlgoBatchedEx = tuneBatchedGemmEx<half>(mSM, B, mS, mWordEmb2Count / mLd, mLd);
    } else {
      mAlgoBatchedEx = tuneBatchedGemmEx<float>(mSM, B, mS, mWordEmb2Count / mLd, mLd);
    }
}

size_t EmbFactorizedLayerNormPluginDynamic::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
  const size_t wordSize = getElementSize(mType);
  const int batchSize = inputs->dims.d[BDIM];
  return wordSize * batchSize * mS * (mWordEmb2Count / mLd) +  // bs * seqlen * embsize
         wordSize * batchSize * mS * mLd;   // bs * seqlen * hidden_size
}

int EmbFactorizedLayerNormPluginDynamic::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        const int batchSize = inputDesc->dims.d[BDIM];
        const int S = inputDesc->dims.d[SDIM];
        int status = STATUS_FAILURE;

        // Our plugin outputs only one tensor
        const auto inputIds = static_cast<const int*>(inputs[0]);
        const auto segmentIds = static_cast<const int*>(inputs[1]);
        const auto inputMask = static_cast<const int*>(inputs[2]);

        const float* beta = mBetaDev.get();
        const float* gamma = mGammaDev.get();
        if (mType == DataType::kFLOAT)
        {
            // step 1: word embedding look up
            float* emb_output = static_cast<float*>(workspace);
            const auto wordEmb = static_cast<const float*>(mWordEmbDev.get());
            int embedding_size = mWordEmb2Count / mLd;
            status = wordEmbLayer<float>(stream, embedding_size, batchSize, S, inputIds, wordEmb, emb_output);
            if (status != cudaSuccess)
            {
                return status;
            }

            // step 2: mWordEmb2 matmul
            float kalpha = 1.0f;
            float kbeta = 0.0f;
            float* word_emb_output = emb_output + (batchSize * mS * (mWordEmb2Count / mLd));

           // batch
            CUBLASASSERT(cublasGemmStridedBatchedEx(
                    mCublas, CUBLAS_OP_N, CUBLAS_OP_N, mLd, mS, embedding_size,
                    &kalpha,
                    (const void*)(mWordEmb2Dev.get()), CUDA_R_32F, mLd, 0,
                    (const void*)(emb_output), CUDA_R_32F, embedding_size, embedding_size,
                    &kbeta,
                    (void*)(word_emb_output), CUDA_R_32F, mLd, mLd,
                    batchSize,
                    CUDA_R_32F, static_cast<cublasGemmAlgo_t>(mAlgoBatchedEx)));


            // step 3: embSkipLayerNorm
            auto output = static_cast<float*>(outputs[0]);
            const auto tokEmb = static_cast<const float*>(mTokEmbDev.get());
            const auto posEmb = static_cast<const float*>(mPosEmbDev.get());
            status = embSkipLayerNormV2<float>(stream, static_cast<int>(mLd), batchSize, S, segmentIds, beta, gamma,
                static_cast<const float*>(word_emb_output), posEmb, tokEmb, output);
            if (status != cudaSuccess)
            {
                return status;
            }
        }
        else if (mType == DataType::kHALF)
        {
            // step 1: word embedding look up
            half* emb_output = static_cast<half*>(workspace);
            const auto wordEmb = static_cast<const half*>(mWordEmbDev.get());
            int embedding_size = mWordEmb2Count / mLd;
            status = wordEmbLayer<half>(stream, embedding_size, batchSize, S, inputIds, wordEmb, emb_output);
            if (status != cudaSuccess)
            {
                return status;
            }

            // step 2: mWordEmb2 matmul
            half kalpha = 1.0f;
            half kbeta = 0.0f;
            half* word_emb_output = emb_output + (batchSize * mS * (mWordEmb2Count / mLd));

            // batch
            CUBLASASSERT(cublasGemmStridedBatchedEx(
                    mCublas, CUBLAS_OP_N, CUBLAS_OP_N, mLd, mS, embedding_size,
                    &kalpha,
                    (const void*)(mWordEmb2Dev.get()), CUDA_R_16F, mLd, 0,
                    (const void*)(emb_output), CUDA_R_16F, embedding_size, embedding_size,
                    &kbeta,
                    (void*)(word_emb_output), CUDA_R_16F, mLd, mLd,
                    batchSize,
                    CUDA_R_16F, static_cast<cublasGemmAlgo_t>(mAlgoBatchedEx)));

            // step 3: embSkipLayerNorm
            auto output = static_cast<half*>(outputs[0]);
            const auto tokEmb = static_cast<const half*>(mTokEmbDev.get());
            const auto posEmb = static_cast<const half*>(mPosEmbDev.get());
            status = embSkipLayerNormV2<half>(stream, static_cast<int>(mLd), batchSize, S, segmentIds, beta, gamma,
                static_cast<const half*>(word_emb_output), posEmb, tokEmb, output);
            if (status != cudaSuccess)
            {
                return status;
            }
        }
        else
        {
            gLogError << "Unsupported type error, expected [kHALF,kFLOAT], but received " << static_cast<int>(mType)
                      << std::endl;

            return STATUS_NOT_SUPPORTED;
        }

        // check mha use fused kernel
        if (mUseFullMask || unfusedMaskSize != getMHAMaskPackedSize(mSM, mMhaType, S))
        {
            size_t warps_m = 0, warps_n = 0, warps_k = 1;
            if (S == 64 || S == 96 || S == 128)
            {
                warps_m = 2;
                warps_n = 2;
            }
            else if (S == 384)
            {
                warps_m = 1;
                warps_n = 8;
            }
            uint32_t* inputMaskX = static_cast<uint32_t*>(outputs[1]);

            status = convertMask(S, batchSize, warps_m, warps_n, warps_k, inputMask, inputMaskX, stream);
        }
        else
        {
            int* maskIdx = static_cast<int*>(outputs[1]);
            status = computeMaskIdx(stream, S, batchSize, inputMask, maskIdx);
        }

        return status;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
}

// IPluginV2Ext Methods
DataType EmbFactorizedLayerNormPluginDynamic::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept
{

    assert(index == 0 || index == 1);
    if (index == 0)
    {
        assert(mType == DataType::kHALF || mType == DataType::kFLOAT);
        return mType;
    }
    return DataType::kFLOAT;
}

// IPluginV2 Methods
const char* EmbFactorizedLayerNormPluginDynamic::getPluginType() const noexcept
{
    return EMB_LAYER_NORM_NAME;
}

const char* EmbFactorizedLayerNormPluginDynamic::getPluginVersion() const noexcept
{
    return EMB_LAYER_NORM_VERSION;
}

int EmbFactorizedLayerNormPluginDynamic::getNbOutputs() const noexcept
{
    return 2;
}

int EmbFactorizedLayerNormPluginDynamic::initialize() noexcept
{
    return 0;
}

void EmbFactorizedLayerNormPluginDynamic::terminate() noexcept
{
    gLogVerbose << "EmbFactorizedLayerNormPluginDynamic terminate." << std::endl;
}

size_t EmbFactorizedLayerNormPluginDynamic::getSerializationSize() const noexcept
{
    const size_t wordSize = getElementSize(mType);
    return sizeof(mType)                  // type
        + sizeof(mMhaType)                // mha plugin datatype
        + sizeof(mLd) * 6                 // mLd, mS, mWordEmbCount, mWordEmb2Count, mTokEmbCount, mPosEmbCount
        + sizeof(mUseFullMask)            // mask type
        + sizeof(mSM)                     // smversion
        + 2 * sizeof(float) * mLd         // beta + gamma
        + sizeof(mAlgoBatchedEx)         // mAlgoBatchedEx
        + wordSize * mWordEmbCount // word emb
        + wordSize * mPosEmbCount  // pos emb
        + wordSize * mTokEmbCount  // tok emb
        + wordSize * mWordEmb2Count // word emb2
        ;
}

void EmbFactorizedLayerNormPluginDynamic::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, mType);
    serialize_value(&buffer, mMhaType);
    serialize_value(&buffer, mLd);
    serialize_value(&buffer, mS);

    serialize_value(&buffer, mWordEmbCount);
    serialize_value(&buffer, mWordEmb2Count);
    serialize_value(&buffer, mTokEmbCount);
    serialize_value(&buffer, mPosEmbCount);

    serialize_value(&buffer, mUseFullMask);
    serialize_value(&buffer, mSM);
    serialize_value(&buffer, mAlgoBatchedEx);

    char* d = static_cast<char*>(buffer);
    serFromDev(d, mBetaDev.get(), mLd);
    serFromDev(d, mGammaDev.get(), mLd);
    const size_t wordSize = getElementSize(mType);

    serFromDev(d, static_cast<char*>(mWordEmbDev.get()), mWordEmbCount * wordSize);
    serFromDev(d, static_cast<char*>(mPosEmbDev.get()), mPosEmbCount * wordSize);
    serFromDev(d, static_cast<char*>(mTokEmbDev.get()), mTokEmbCount * wordSize);
    serFromDev(d, static_cast<char*>(mWordEmb2Dev.get()), mWordEmb2Count * wordSize);
}

void EmbFactorizedLayerNormPluginDynamic::destroy() noexcept
{
    gLogVerbose << "EmbFactorizedLayerNormPluginDynamic destroy." << std::endl;
    // This gets called when the network containing plugin is destroyed
    mGammaDev.reset(nullptr);
    mBetaDev.reset(nullptr);
    mWordEmbDev.reset(nullptr);
    mPosEmbDev.reset(nullptr);
    mTokEmbDev.reset(nullptr);
    mWordEmb2Dev.reset(nullptr);
    CUBLASASSERT(cublasDestroy(mCublas));
    delete this;
}

void EmbFactorizedLayerNormPluginDynamic::setPluginNamespace(const char* libNamespace) noexcept
{
    try
    {
        mNamespace = libNamespace;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
}

const char* EmbFactorizedLayerNormPluginDynamic::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

///////////////////////

EmbFactorizedLayerNormPluginDynamicCreator::EmbFactorizedLayerNormPluginDynamicCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* EmbFactorizedLayerNormPluginDynamicCreator::getPluginName() const noexcept
{
    return EMB_LAYER_NORM_NAME;
}

const char* EmbFactorizedLayerNormPluginDynamicCreator::getPluginVersion() const noexcept
{
    return EMB_LAYER_NORM_VERSION;
}

const PluginFieldCollection* EmbFactorizedLayerNormPluginDynamicCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* EmbFactorizedLayerNormPluginDynamicCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        gLogVerbose << "EmbFactorizedLayerNormPluginDynamic createPlugin." << std::endl;

        bool output_fp16 = false;
        bool useFullMask = false;
        Weights beta;
        Weights gamma;
        Weights word_emb;
        Weights pos_emb;
        Weights tok_emb;
        Weights word_emb2;
        int mhaTypeId = 0;
        for (int i = 0; i < fc->nbFields; i++)
        {
            std::string field_name(fc->fields[i].name);
            if (field_name.compare("bert_embeddings_layernorm_beta") == 0)
            {
                gLogVerbose << "Building bert_embeddings_layernorm_beta..." << std::endl;
                beta.values = fc->fields[i].data;
                beta.count = fc->fields[i].length;
                beta.type = fieldTypeToDataType(fc->fields[i].type);
            }

            if (field_name.compare("bert_embeddings_layernorm_gamma") == 0)
            {
                gLogVerbose << "Building bert_embeddings_layernorm_gamma..." << std::endl;
                gamma.values = fc->fields[i].data;
                gamma.count = fc->fields[i].length;
                gamma.type = fieldTypeToDataType(fc->fields[i].type);
            }

            if (field_name.compare("bert_embeddings_word_embeddings") == 0)
            {
                gLogVerbose << "Building bert_embeddings_word_embeddings..." << std::endl;
                word_emb.values = fc->fields[i].data;
                word_emb.count = fc->fields[i].length;
                word_emb.type = fieldTypeToDataType(fc->fields[i].type);
            }

            if (field_name.compare("bert_embeddings_word_embeddings_2") == 0)
            {
                gLogVerbose << "Building bert_embeddings_word_embeddings_2 ..." << std::endl;
                word_emb2.values = fc->fields[i].data;
                word_emb2.count = fc->fields[i].length;
                word_emb2.type = fieldTypeToDataType(fc->fields[i].type);
            }

            if (field_name.compare("bert_embeddings_token_type_embeddings") == 0)
            {
                gLogVerbose << "Building bert_embeddings_token_type_embeddings..." << std::endl;
                tok_emb.values = fc->fields[i].data;
                tok_emb.count = fc->fields[i].length;
                tok_emb.type = fieldTypeToDataType(fc->fields[i].type);
            }

            if (field_name.compare("bert_embeddings_position_embeddings") == 0)
            {
                gLogVerbose << "Building bert_embeddings_position_embeddings..." << std::endl;
                pos_emb.values = fc->fields[i].data;
                pos_emb.count = fc->fields[i].length;
                pos_emb.type = fieldTypeToDataType(fc->fields[i].type);
            }
            if (field_name.compare("output_fp16") == 0)
            {
                gLogVerbose << "Building output_fp16..." << std::endl;
                assert(fc->fields[i].type == PluginFieldType::kINT32);
                output_fp16 = static_cast<const int*>(fc->fields[i].data)[0] != 0;
            }
            if (field_name.compare("full_mask") == 0)
            {
                gLogVerbose << "Building full_mask..." << std::endl;
                assert(fc->fields[i].type == PluginFieldType::kINT32);
                useFullMask = static_cast<const int*>(fc->fields[i].data)[0] != 0;
            }
            if (field_name.compare("mha_type_id") == 0)
            {
                mhaTypeId = *static_cast<const int*>(fc->fields[i].data);
                ASSERT(mhaTypeId >= 0 && mhaTypeId <= 3);
                gLogVerbose << "Building mha typeId: " << mhaTypeId << std::endl;
            }
        }

        gLogVerbose << "Building the Plugin..." << std::endl;
        DataType mhaType = static_cast<DataType>(mhaTypeId);
        EmbFactorizedLayerNormPluginDynamic* p
            = new EmbFactorizedLayerNormPluginDynamic(name, output_fp16 ? DataType::kHALF : DataType::kFLOAT, mhaType, beta,
                gamma, word_emb, pos_emb, tok_emb, word_emb2, useFullMask);
        return p;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* EmbFactorizedLayerNormPluginDynamicCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call EmbFactorizedLayerNormPluginDynamic::destroy()
        return new EmbFactorizedLayerNormPluginDynamic(name, serialData, serialLength);
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void EmbFactorizedLayerNormPluginDynamicCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    try
    {
        mNamespace = libNamespace;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
}

const char* EmbFactorizedLayerNormPluginDynamicCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
} // namespace bert

#endif // CUDA_VERSION >= 10010

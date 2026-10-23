#include "../include/quant.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

namespace LLaMA_Tensor{
namespace{
uint16_t Float2Fp16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(float));

    uint16_t sign = (bits >> 16) & 0x8000u;
    uint32_t exp32 = (bits >> 23) & 0xFFu;
    uint32_t frac32 = bits & 0x7FFFFFu;

    if (exp32 == 0xFFu) {
        if (frac32 == 0) {
            return sign | 0x7C00u; 
        } else {
            return sign | 0x7E00u; 
        }
    }
    // 处理零 / 次正规数
    if (exp32 == 0) {
        if (frac32 == 0) {
            return sign;
        }
        // F32 次正规数，对于 FP16 来说几乎都是 0
        // 更精确的处理需要按真实指数右移，这里简化为 0
        return sign;
    }

    int exponent = (int)exp32 - 112;

    // 溢出检查
    if (exponent >= 31) {
        return sign | 0x7C00u; 
    }

    // 下溢检查（次正规数或 0）
    if (exponent <= 0) {

        int real_exp = (int)exp32 - 127;
        uint32_t significand = frac32 | 0x800000u; 
        int shift = -14 - real_exp;
        if (shift > 24) {
            return sign;
        }
        uint32_t cut = significand & ((1u << shift) - 1);
        uint32_t tail = significand >> shift;
        uint32_t half = 1u << (shift - 1);
        if (cut > half || (cut == half && (tail & 1))) {
            tail++;
        }
        if (tail >= 0x400u) {
            return sign | 0x0400u; 
        }
        return sign | (uint16_t)tail;
    }

    uint32_t tail_cut = frac32 & 0x1FFFu;
    uint32_t tail = (frac32 >> 13) & 0x3FFu;

    if (tail_cut > 0x1000u) {
        tail++;
    } else if (tail_cut == 0x1000u) {
        if (tail & 1u) {
            tail++;
        }
    }

    // 处理尾数溢出
    if (tail >= 0x400u) {
        tail = 0;
        exponent++;
        if (exponent >= 31) {
            return sign | 0x7C00u; 
        }
    }

    return sign | ((uint16_t)exponent << 10) | (uint16_t)tail;
}

float Fp162Float(uint16_t value){
    uint32_t sign = (value & 0x8000u) << 16;    // 移到 bit31
    uint32_t exp16 = (value >> 10) & 0x1Fu;
    uint32_t frac16 = value & 0x3FFu;

    uint32_t bits32;

    if (exp16 == 0) {
        if (frac16 == 0) {
            bits32 = sign;
        } else {
            // 次正规数：规格化为 FP32 正常数
            int p = 9;
            while ((frac16 & (1u << p)) == 0) {
                --p;
            }
            uint32_t E = p + 103; 
            uint32_t fraction = (frac16 & ((1u << p) - 1)) << (23 - p);
            bits32 = sign | (E << 23) | fraction;
        }
    } else if (exp16 == 31) {
        if (frac16 == 0) {
            // Inf
            bits32 = sign | 0x7F800000u;
        } else {
            // NaN
            bits32 = sign | 0x7FC00000u;
        }
    } else {
        // 正常数
        uint32_t E = exp16 + 112;           
        uint32_t fraction = frac16 << 13;   
        bits32 = sign | (E << 23) | fraction;
    }

    float f;
    std::memcpy(&f, &bits32, sizeof(f));
    return f;
}

size_t CheckedNumel(const std::vector<int>& shape, const std::string& caller){
    size_t total = 1;
    for(const int dim : shape){
        if(dim <= 0){
            throw std::runtime_error(caller + ": dim must be positive");
        }
        if(total > std::numeric_limits<size_t>::max() / dim){
            throw std::out_of_range(caller + ": total is beyond the range");
        }
        total *= dim;
    }
    return total;
}


int RowBlockStart(int row, int block, int row_size, int n_blocks_per_row, int K){
    const int padded_row = n_blocks_per_row * K;
    return row * padded_row + block * K - row * (padded_row - row_size);
}

int RowBlockWidth(int block, int row_size, int n_blocks_per_row, int K){
    return block == n_blocks_per_row - 1 ? row_size - (n_blocks_per_row - 1) * K : K;
}
}

std::vector<BlockQ80> QuantizedToQ80(const Tensor& t){
    if(t.TensorNum() == 0){
        return {};
    }

    int row_size = t.shape.size() >= 2 ? t.shape.back() : t.shape[0];
    int n_rows = t.shape.size() >= 2 ? t.TensorNum() / row_size : 1;
    int n_blocks_per_row = (row_size + kQ80BlockSize - 1) / kQ80BlockSize;
    int total_blocks = n_rows * n_blocks_per_row;
    std::vector<BlockQ80> result;
    result.reserve(total_blocks);
    for(int row = 0;row < n_rows;row++){
        for(int block = 0;block < n_blocks_per_row;block++){
            BlockQ80 block_q80;
            std::memset(&block_q80, 0 , sizeof(block_q80)); //全零初始化
            int block_start_base = RowBlockStart(row, block, row_size, n_blocks_per_row, kQ80BlockSize);
            int block_end_base = block_start_base
                + RowBlockWidth(block, row_size, n_blocks_per_row, kQ80BlockSize);


            float max_ele = std::abs(t.data[block_start_base]);
            for(int i = block_start_base;i < block_end_base;i++){
                max_ele = std::max(max_ele, std::abs(t.data[i]));
            }
            if(max_ele == 0.0f){
                block_q80.scale = 0;
                result.push_back(block_q80);
                continue;
            }

            float scale = max_ele / 127.0f;
            float store_scale = Fp162Float(Float2Fp16(scale));
            block_q80.scale = Float2Fp16(store_scale);
            float i_scale = store_scale != 0.0f ? 1.0f / store_scale : 0.0f;
            for(int i = 0;i < block_end_base - block_start_base;i++){
                float qf = t.data[block_start_base + i] * i_scale;
                int qi = static_cast<int>(std::round(qf)); 
                if(qi > 127){
                    qi = 127;
                }else if(qi < -127){
                    qi = -127;
                }
                block_q80.qs[i] = static_cast<int8_t>(qi);
            }
            result.push_back(block_q80);
        }
        
    }
    return result;
}

Tensor DequantizedFromQ80(const std::vector<BlockQ80>& blocks, const std::vector<int>& shape){
    size_t total = CheckedNumel(shape, "DequantizedFromQ80");

    int row_size = shape.size() >= 2 ? shape.back() : static_cast<int>(total);
    int n_rows = shape.size() >= 2 ? static_cast<int>(total) / row_size : 1;
    int n_blocks_per_row = ((row_size + kQ80BlockSize - 1) / kQ80BlockSize);

    if(n_blocks_per_row * n_rows != blocks.size()){
        throw std::runtime_error("DequantizedFromQ80: block size dismatch, expect: "
             + std::to_string(n_blocks_per_row * n_rows));
    }

    Tensor result(shape, 0.0f);
    for(int block = 0;block < static_cast<int>(blocks.size());block++){
        const int row = block / n_blocks_per_row;
        const int block_in_row = block % n_blocks_per_row;
        const int block_start_base = RowBlockStart(row, block_in_row, row_size, n_blocks_per_row, kQ80BlockSize);
        const int block_width = RowBlockWidth(block_in_row, row_size, n_blocks_per_row, kQ80BlockSize);
        const float scale = Fp162Float(blocks[block].scale);
        for(int i = 0;i < block_width;i++){
            result.data[block_start_base + i] =
                scale * static_cast<float>(blocks[block].qs[i]);
        }
    }
    return result;
}

std::vector<BlockQ40> QuantizedToQ40(const Tensor& t){
    if(t.TensorNum() == 0){
        return {};
    }

    int row_size = t.shape.size() >= 2 ? t.shape.back() : t.shape[0];
    int n_rows = t.shape.size() >= 2 ? t.TensorNum() / row_size : 1;
    int n_blocks_per_row = (row_size + kQ40BlockSize - 1) / kQ40BlockSize;
    int total_blocks = n_rows * n_blocks_per_row;
    std::vector<BlockQ40> result;
    result.reserve(total_blocks);
    for(int row = 0;row < n_rows;row++){
        for(int block = 0;block < n_blocks_per_row;block++){
            BlockQ40 block_q40;
            std::memset(&block_q40, 0, sizeof(BlockQ40));
            int block_start_base = RowBlockStart(row, block, row_size, n_blocks_per_row, kQ40BlockSize);
            int block_end_base = block_start_base
                + RowBlockWidth(block, row_size, n_blocks_per_row, kQ40BlockSize);

            float max_ele = std::abs(t.data[block_start_base]);
            for(int i = block_start_base;i < block_end_base;i++){
                max_ele = std::max(max_ele, std::abs(t.data[i]));
            }
            if(max_ele == 0.0f){
                block_q40.scale = 0;
                result.push_back(block_q40);
                continue;
            }
            float scale = max_ele / 7.0f;
            float store_scale = Fp162Float(Float2Fp16(scale));
            block_q40.scale = Float2Fp16(store_scale);
            float i_scale = store_scale != 0.0f ? 1.0f / store_scale : 0.0f;
            const int block_width = RowBlockWidth(block, row_size, n_blocks_per_row, kQ40BlockSize);
            for(int i = 0;i < block_width;i++){
                float qf = t.data[block_start_base + i] * i_scale;
                int qi = static_cast<int>(std::round(qf + 8));
                if(qi > 15){
                    qi = 15;
                }else if(qi < 0){
                    qi = 0;
                }
                if(i < 16){
                    block_q40.qs[i] = static_cast<int8_t>(
                        (static_cast<uint8_t>(block_q40.qs[i]) & 0xF0u) | static_cast<uint8_t>(qi));
                }else{
                    block_q40.qs[i - 16] = static_cast<int8_t>(
                        (static_cast<uint8_t>(block_q40.qs[i - 16]) & 0x0Fu)
                        | (static_cast<uint8_t>(qi) << 4));
                }
            }
            result.push_back(block_q40);
        }
    }
    return result;
}

Tensor DequantizedFromQ40(const std::vector<BlockQ40>& blocks, const std::vector<int>& shape){
    size_t total = CheckedNumel(shape, "DequantizedFromQ40");

    int row_size = shape.size() >= 2 ? shape.back() : static_cast<int>(total);
    int n_rows = shape.size() >= 2 ? static_cast<int>(total) / row_size : 1;
    int n_blocks_per_row = ((row_size + kQ40BlockSize - 1) / kQ40BlockSize);

    if(n_blocks_per_row * n_rows != blocks.size()){
        throw std::runtime_error("DequantizedFromQ40: block size dismatch, expect: "
             + std::to_string(n_blocks_per_row * n_rows));
    }

    Tensor result(shape, 0.0f);
    for(int block = 0;block < static_cast<int>(blocks.size());block++){
        const int row = block / n_blocks_per_row;
        const int block_in_row = block % n_blocks_per_row;
        const int block_start_base = RowBlockStart(row, block_in_row, row_size, n_blocks_per_row, kQ40BlockSize);
        const int block_width = RowBlockWidth(block_in_row, row_size, n_blocks_per_row, kQ40BlockSize);
        const float scale = Fp162Float(blocks[block].scale);
        for(int i = 0;i < block_width;i++){
            const uint8_t byte =
                static_cast<uint8_t>(blocks[block].qs[i < 16 ? i : i - 16]);
            const int q = i < 16 ? static_cast<int>(byte & 0x0fu)
                                 : static_cast<int>(byte >> 4);
            result.data[block_start_base + i] =
                static_cast<float>(q - 8) * scale;
        }
    }
    return result;
}

std::vector<BlockQ41> QuantizedToQ41(const Tensor& t){
    if(t.TensorNum() == 0){
        return {};
    }

    int row_size = t.shape.size() >= 2 ? t.shape.back() : t.shape[0];
    int n_rows = t.shape.size() >= 2 ? t.TensorNum() / row_size : 1;
    int n_blocks_per_row = (row_size + kQ41BlockSize - 1) / kQ41BlockSize;
    int total_blocks = n_rows * n_blocks_per_row;
    std::vector<BlockQ41> result;
    result.reserve(total_blocks);
    for(int row = 0;row < n_rows;row++){
        for(int block = 0;block < n_blocks_per_row;block++){
            BlockQ41 block_q41;
            std::memset(&block_q41, 0, sizeof(BlockQ41));
            int block_start_base = RowBlockStart(row, block, row_size, n_blocks_per_row, kQ41BlockSize);
            int block_end_base = block_start_base
                + RowBlockWidth(block, row_size, n_blocks_per_row, kQ41BlockSize);

            float max_ele = t.data[block_start_base];
            float min_ele = t.data[block_start_base];
            for(int i = block_start_base;i < block_end_base;i++){
                max_ele = std::max(max_ele, t.data[i]);
                min_ele = std::min(min_ele, t.data[i]);
            }
            if(max_ele == 0.0f && min_ele == 0.0f){
                block_q41.scale = 0;
                block_q41.scale_min = 0;
                result.push_back(block_q41);
                continue;
            }
            float scale = (max_ele - min_ele) / 15.0f;
            float store_scale = Fp162Float(Float2Fp16(scale));
            float store_scale_min = Fp162Float(Float2Fp16(min_ele));
            block_q41.scale = Float2Fp16(store_scale);
            block_q41.scale_min = Float2Fp16(store_scale_min);
            float i_scale = store_scale != 0.0f ? 1.0f / store_scale : 0.0f;
            const int block_width = RowBlockWidth(block, row_size, n_blocks_per_row, kQ41BlockSize);
            for(int i = 0;i < block_width;i++){
                float qf = (t.data[block_start_base + i] - store_scale_min) * i_scale;
                int qi = static_cast<int>(std::round(qf));
                if(qi > 15){
                    qi = 15;
                }else if(qi < 0){
                    qi = 0;
                }
                if(i < 16){
                    block_q41.qs[i] = static_cast<uint8_t>(
                        (block_q41.qs[i] & 0xF0u) | static_cast<uint8_t>(qi));
                }else{
                    block_q41.qs[i - 16] = static_cast<uint8_t>(
                        (block_q41.qs[i - 16] & 0x0Fu) | (static_cast<uint8_t>(qi) << 4));
                }
            }
            result.push_back(block_q41);
        }
    }
    return result;
}

Tensor DequantizedFromQ41(const std::vector<BlockQ41>& blocks, const std::vector<int>& shape){
    size_t total = CheckedNumel(shape, "DequantizedFromQ41");

    int row_size = shape.size() >= 2 ? shape.back() : static_cast<int>(total);
    int n_rows = shape.size() >= 2 ? static_cast<int>(total) / row_size : 1;
    int n_blocks_per_row = ((row_size + kQ41BlockSize - 1) / kQ41BlockSize);

    if(n_blocks_per_row * n_rows != blocks.size()){
        throw std::runtime_error("DequantizedFromQ41: block size dismatch, expect: "
             + std::to_string(n_blocks_per_row * n_rows));
    }

    Tensor result(shape, 0.0f);
    for(int block = 0;block < static_cast<int>(blocks.size());block++){
        const int row = block / n_blocks_per_row;
        const int block_in_row = block % n_blocks_per_row;
        const int block_start_base = RowBlockStart(row, block_in_row, row_size, n_blocks_per_row, kQ41BlockSize);
        const int block_width = RowBlockWidth(block_in_row, row_size, n_blocks_per_row, kQ41BlockSize);
        const float scale = Fp162Float(blocks[block].scale);
        const float scale_min = Fp162Float(blocks[block].scale_min);
        for(int i = 0;i < block_width;i++){
            const uint8_t byte = blocks[block].qs[i < 16 ? i : i - 16];
            const int q = i < 16 ? static_cast<int>(byte & 0x0fu)
                                 : static_cast<int>(byte >> 4);
            result.data[block_start_base + i] =
                static_cast<float>(q) * scale + scale_min;
        }
    }
    return result;
}

template <typename BlockType, int KBlockSize>
static Tensor LinearQuantizedImpl(const Tensor& x, const std::vector<BlockType>& blocks
    , const std::vector<int>& weight_shape, float (*dequant_fn)(const BlockType&, int)){
    if(weight_shape.size() != 2){
        throw std::runtime_error("expect 2 dimensions weight");
    }
    int in_features;
    int rows = 1;
    bool is_1d = false;
    if(x.dim() == 1){
        in_features = x.shape[0];
        is_1d = true;
    } else if(x.dim() == 2){
        in_features = x.shape[1];
        rows = x.shape[0];
    } else{
        x.TensorPrint();
        throw std::runtime_error("LinearQuantizedImpl: expect 2 dimensions shape of tensor, now: ");
    }
    int out_features = weight_shape[0];
    if(weight_shape[1] != in_features){
        std::cerr << "x shape:" << std::endl;
        x.TensorPrint();
        std::cerr << "weight shape:" << QuantizedTensor{QuantType::kF32, weight_shape}.ShapeString() << std::endl;
        throw std::runtime_error("LinearQuantizedImpl: dimension is dismatched");
    }
    int n_blocks_per_row = (in_features + KBlockSize - 1) / KBlockSize;
    size_t expected_blocks = static_cast<size_t>(out_features) * n_blocks_per_row;
    if (blocks.size() != expected_blocks) {
        throw std::runtime_error(
            "linear_quantized: block count mismatch: expected " +
            std::to_string(expected_blocks) + ", got " +
            std::to_string(blocks.size()));
    }
    
    Tensor result(is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features}, 0.0f);
    //并行多线程处理线性矩阵乘
    auto f = [&](int begin, int end){
        for (int flat_index = begin; flat_index < end; ++flat_index) {
            int row = flat_index / out_features;
            int j = flat_index % out_features;
            const float* x_row =
                x.data.data() + static_cast<size_t>(row) * in_features;
            float sum = 0.0f;
            int block_base = j * n_blocks_per_row;
            for (int b = 0; b < n_blocks_per_row; ++b) {
                const BlockType& block = blocks[block_base + b];
                int base_k = b * KBlockSize;
                int k_end = std::min(base_k + KBlockSize, in_features);
                for (int k = base_k; k < k_end; ++k) {
                    float w = dequant_fn(block, k - base_k);
                    sum += w * x_row[k];
                }
            }
            result.data[static_cast<size_t>(row) * out_features + j] = sum;
        }
    };
    ParallelFor(rows * out_features, f);
    return result;
}

static float DequantQ80(const BlockQ80& block, int idx) {
    float d = Fp162Float(block.scale);
    return d * static_cast<float>(block.qs[idx]);
}

static float DequantQ40(const BlockQ40& block, int idx) {
    float d = Fp162Float(block.scale);
    int q;
    if (idx < 16) {
        q = static_cast<int>(block.qs[idx] & 0x0F) - 8;
    } else {
        q = static_cast<int>(block.qs[idx - 16] >> 4) - 8;
    }
    return d * static_cast<float>(q);
}

static float DequantQ41(const BlockQ41& block, int idx) {
    float d = Fp162Float(block.scale);
    float m = Fp162Float(block.scale_min);
    int q;
    if (idx < 16) {
        q = static_cast<int>(block.qs[idx] & 0x0F);
    } else {
        q = static_cast<int>(block.qs[idx - 16] >> 4);
    }
    return d * static_cast<float>(q) + m;
}
#ifdef MINI_LLAMA_USE_AVX2
// 8 个 int16（位于 ymm 低 128 位）→ 8 个 int32（占满整个 ymm）
static inline __m256i I16x8ToI32x8(__m256i v16) {
    __m128i lo = _mm256_castsi256_si128(v16);    // 低 4 个 int16
    __m128i hi = _mm_srli_si128(lo, 8);          // 字节右移 8 → 高 4 个 int16 落到低 64 位
    __m256i lo32 = _mm256_cvtepi16_epi32(lo);    // int16 0~3 → int32（低 128 位）
    __m256i hi32 = _mm256_cvtepi16_epi32(hi);    // int16 4~7 → int32（低 128 位）
    return _mm256_setr_m128i(_mm256_castsi256_si128(lo32),
                             _mm256_castsi256_si128(hi32));  // 拼成 8 个 int32
}

// 8 路 float32 水平归约 → 标量
static inline float HsumF32x8(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s = _mm_add_ps(lo, hi);
    __m128 s2 = _mm_add_ps(s, _mm_movehl_ps(s, s));
    __m128 s3 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0x01));
    return _mm_cvtss_f32(s3);
}

// Q8_0 tile 微内核：计算 out[r0, r0+r_nums) x [j0, j0+j_nums) 这一块输出。
// 内层沿用原有的"四路 FMA 一次处理整个 32 元素量化块"方案：每个输出点沿
// 整个 k 维在 4 个 YMM 累加器中累加完毕，才归约并写回一次（无中途写回）。
// 与 F32 MicroKernel8x8Avx2 的差异：Q8_0 权重按 32 元素块量化存储、每个输出
// 列有各自的 fp16 scale，8 列同一 k 位置的反量化权重无法用单条 gather 取齐，
// 因此 SIMD 放在块内 k 维；跨列复用由 tile 内层循环共享同一个 x_row 指针获得。
static void MicroKernelQ80Avx2(const float* x,
                               const BlockQ80* weight,
                               float* out,
                               int r0, int j0, int r_nums, int j_nums,
                               int in_features, int out_features,
                               int n_blocks_per_row) {
    if (r_nums <= 0 || j_nums <= 0) {
        return;
    }
    for (int r = 0; r < r_nums; ++r) {
        // 每行只取一次 x_row，tile 内各输出列复用
        const float* x_row = x + static_cast<size_t>(r0 + r) * in_features;
        for (int j = 0; j < j_nums; ++j) {
            int j_global = j0 + j;
            int block_base = n_blocks_per_row * j_global;
            __m256 sum_vec = _mm256_setzero_ps();  // 8 路累加器
            float scaler = 0.0f;
            for (int b = 0; b < n_blocks_per_row; ++b) {
                const BlockQ80& block = weight[block_base + b];
                float d = Fp162Float(block.scale);
                int base_k = b * kQ80BlockSize;
                int k_end = std::min(base_k + kQ80BlockSize, in_features);
                int k = base_k;

                // 4 路独立累加器
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();

                for (; k + 32 <= k_end; k += 32) {
                    // 一次性加载 32 个 int8
                    __m256i q8 = _mm256_loadu_si256((const __m256i*)(block.qs + (k - base_k)));
                    // 拆成 4 组，每组 8 个 int8 -> int16
                    __m128i q8_lo = _mm256_castsi256_si128(q8);
                    __m128i q8_hi = _mm256_extracti128_si256(q8, 1);
                    __m256i q16_0 = _mm256_cvtepi8_epi16(q8_lo);
                    __m256i q16_1 = _mm256_cvtepi8_epi16(_mm_srli_si128(q8_lo, 8));
                    __m256i q16_2 = _mm256_cvtepi8_epi16(q8_hi);
                    __m256i q16_3 = _mm256_cvtepi8_epi16(_mm_srli_si128(q8_hi, 8));
                    // 每组 int16 -> int32 -> float，再乘以 scale
                    __m256 qf_0 = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(I16x8ToI32x8(q16_0)), _mm256_set1_ps(d));
                    __m256 qf_1 = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(I16x8ToI32x8(q16_1)), _mm256_set1_ps(d));
                    __m256 qf_2 = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(I16x8ToI32x8(q16_2)), _mm256_set1_ps(d));
                    __m256 qf_3 = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(I16x8ToI32x8(q16_3)), _mm256_set1_ps(d));
                    // 加载输入
                    __m256 xf_0 = _mm256_loadu_ps(x_row + k);
                    __m256 xf_1 = _mm256_loadu_ps(x_row + k + 8);
                    __m256 xf_2 = _mm256_loadu_ps(x_row + k + 16);
                    __m256 xf_3 = _mm256_loadu_ps(x_row + k + 24);
                    // 4 条 FMA 指令并行累加到 4 个独立累加器
                    acc0 = _mm256_fmadd_ps(qf_0, xf_0, acc0);
                    acc1 = _mm256_fmadd_ps(qf_1, xf_1, acc1);
                    acc2 = _mm256_fmadd_ps(qf_2, xf_2, acc2);
                    acc3 = _mm256_fmadd_ps(qf_3, xf_3, acc3);
                }
                sum_vec = _mm256_add_ps(sum_vec, acc0);
                sum_vec = _mm256_add_ps(sum_vec, acc1);
                sum_vec = _mm256_add_ps(sum_vec, acc2);
                sum_vec = _mm256_add_ps(sum_vec, acc3);
                // 归约
                scaler += HsumF32x8(sum_vec);
                sum_vec = _mm256_setzero_ps();

                // 回退：不足 32 个元素的尾部（或整块不满）走标量
                for (; k < k_end; ++k) {
                    float w = d * static_cast<float>(block.qs[k - base_k]);
                    scaler += w * x_row[k];
                }
            }
            out[static_cast<size_t>(r0 + r) * out_features + j_global] = scaler;
        }
    }
}

// 逐输出元素并行的原始 AVX2 内核（改造前的实现）。
// 保留它作为独立分发路径，便于在同一负载下对比"改造前整块内核"与
// "改造后 tile 微内核"。每个 (row, j) 独立计算一个输出点：内层仍是
// "四路 FMA 一次处理整个 32 元素量化块 + 尾块标量回退"。
static void LinearQ80PerElementAvx2(const Tensor& x,
                                    const std::vector<BlockQ80>& weight,
                                    int rows, int in_features, int out_features,
                                    int n_blocks_per_row, Tensor& result) {
    auto f = [&](int begin, int end) {
        for (int flat_index = begin; flat_index < end; ++flat_index) {
            int row = flat_index / out_features;
            int j = flat_index % out_features;
            const float* x_row =
                x.data.data() + static_cast<size_t>(row) * in_features;
            int block_base = j * n_blocks_per_row;
            __m256 sum_vec = _mm256_setzero_ps();  // 8 路累加器
            float scaler = 0.0f;
            for (int b = 0; b < n_blocks_per_row; ++b) {
                const BlockQ80& block = weight[block_base + b];
                float d = Fp162Float(block.scale);
                int base_k = b * kQ80BlockSize;
                int k_end = std::min(base_k + kQ80BlockSize, in_features);
                int k = base_k;

                // 4 路独立累加器
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();

                for (; k + 32 <= k_end; k += 32) {
                    // 一次性加载 32 个 int8
                    __m256i q8 = _mm256_loadu_si256(
                        (const __m256i*)(block.qs + (k - base_k)));
                    // 拆成 4 组，每组 8 个 int8 -> int16
                    __m128i q8_lo = _mm256_castsi256_si128(q8);
                    __m128i q8_hi = _mm256_extracti128_si256(q8, 1);
                    __m256i q16_0 = _mm256_cvtepi8_epi16(q8_lo);
                    __m256i q16_1 = _mm256_cvtepi8_epi16(_mm_srli_si128(q8_lo, 8));
                    __m256i q16_2 = _mm256_cvtepi8_epi16(q8_hi);
                    __m256i q16_3 = _mm256_cvtepi8_epi16(_mm_srli_si128(q8_hi, 8));
                    // 每组 int16 -> int32 -> float，再乘以 scale
                    __m256 qf_0 = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(I16x8ToI32x8(q16_0)), _mm256_set1_ps(d));
                    __m256 qf_1 = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(I16x8ToI32x8(q16_1)), _mm256_set1_ps(d));
                    __m256 qf_2 = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(I16x8ToI32x8(q16_2)), _mm256_set1_ps(d));
                    __m256 qf_3 = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(I16x8ToI32x8(q16_3)), _mm256_set1_ps(d));
                    // 加载输入
                    __m256 xf_0 = _mm256_loadu_ps(x_row + k);
                    __m256 xf_1 = _mm256_loadu_ps(x_row + k + 8);
                    __m256 xf_2 = _mm256_loadu_ps(x_row + k + 16);
                    __m256 xf_3 = _mm256_loadu_ps(x_row + k + 24);
                    // 4 条 FMA 指令并行累加到 4 个独立累加器
                    acc0 = _mm256_fmadd_ps(qf_0, xf_0, acc0);
                    acc1 = _mm256_fmadd_ps(qf_1, xf_1, acc1);
                    acc2 = _mm256_fmadd_ps(qf_2, xf_2, acc2);
                    acc3 = _mm256_fmadd_ps(qf_3, xf_3, acc3);
                }
                sum_vec = _mm256_add_ps(sum_vec, acc0);
                sum_vec = _mm256_add_ps(sum_vec, acc1);
                sum_vec = _mm256_add_ps(sum_vec, acc2);
                sum_vec = _mm256_add_ps(sum_vec, acc3);
                // 归约
                scaler += HsumF32x8(sum_vec);
                sum_vec = _mm256_setzero_ps();

                // 回退：不足 32 个元素的尾部（或整块不满）走标量
                for (; k < k_end; ++k) {
                    float w = d * static_cast<float>(block.qs[k - base_k]);
                    scaler += w * x_row[k];
                }
            }
            result.data[static_cast<size_t>(row) * out_features + j] = scaler;
        }
    };
    ParallelFor(rows * out_features, f);
}

#endif

// ---------------------------------------------------------------------------
// LinearQ80 运行时分发状态
// ---------------------------------------------------------------------------
static LinearQ80Mode g_linear_q80_mode = LinearQ80Mode::kTiledSimd;

LinearQ80Mode DefaultLinearQ80Mode() { return LinearQ80Mode::kTiledSimd; }

LinearQ80Mode GetLinearQ80Mode() { return g_linear_q80_mode; }

void SetLinearQ80Mode(LinearQ80Mode mode) { g_linear_q80_mode = mode; }

const char* LinearQ80ModeName(LinearQ80Mode mode) {
    switch (mode) {
        case LinearQ80Mode::kScalar:
            return "scalar";
        case LinearQ80Mode::kPerElementSimd:
            return "per-element";
        case LinearQ80Mode::kTiledSimd:
            return "tiled";
    }
    return "unknown";
}

bool ParseLinearQ80Mode(const std::string& text, LinearQ80Mode& out) {
    if (text == "scalar") {
        out = LinearQ80Mode::kScalar;
        return true;
    }
    if (text == "per-element" || text == "per_element" ||
        text == "perelement") {
        out = LinearQ80Mode::kPerElementSimd;
        return true;
    }
    if (text == "tiled") {
        out = LinearQ80Mode::kTiledSimd;
        return true;
    }
    return false;
}

Tensor LinearQ80(const Tensor& x, const std::vector<BlockQ80>& weight, const std::vector<int>& weight_shape){
    #if !defined(MINI_LLAMA_USE_AVX2) && !defined(MINI_LLAMA_USE_NEON)
    return LinearQuantizedImpl<BlockQ80, kQ80BlockSize>(x, weight, weight_shape, DequantQ80);
    #else
    if (weight_shape.size() != 2) {
        throw std::runtime_error("LinearQ80: expected 2D weight shape");
    }

    int in_features;
    int rows = 1;
    bool is_1d = false;
    if (x.dim() == 1) {
        in_features = x.shape[0];
        is_1d = true;
    } else if (x.dim() == 2) {
        rows = x.shape[0];
        in_features = x.shape[1];
    } else {
        std::cerr << "x shape: " << std::endl;
        x.TensorPrint();
        throw std::runtime_error(
            "LinearQ80: expected x shape [in_features] or [batch, in_features]");
    }

    int out_features = weight_shape[0];
    if (weight_shape[1] != in_features) {
        std::cerr << "x shape: " << std::endl;
        x.TensorPrint();
        std::cerr << "weight shape: " << QuantizedTensor{QuantType::kF32, weight_shape}.ShapeString() << std::endl;
        throw std::runtime_error("LinearQ80: dimension mismatch");
    }

    int n_blocks_per_row = (in_features + kQ80BlockSize - 1) / kQ80BlockSize;
    size_t expected_blocks = static_cast<size_t>(out_features) * n_blocks_per_row;
    if (weight.size() != expected_blocks) {
        throw std::runtime_error("LinearQ80: block count mismatch: expected " +
                                std::to_string(expected_blocks) + ", got " +
                                std::to_string(weight.size()));
    }

    // 标量路径：与无 SIMD 构建完全相同的实现，作为参考/回退基线。
    if (GetLinearQ80Mode() == LinearQ80Mode::kScalar) {
        return LinearQuantizedImpl<BlockQ80, kQ80BlockSize>(
            x, weight, weight_shape, DequantQ80);
    }

    Tensor result(is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features}, 0.0f);
    #endif
    #ifdef MINI_LLAMA_USE_NEON
    auto f = [&](int begin, int end){
        for(int flat_index = begin;flat_index < end;flat_index++){
            int row = flat_index / out_features;
            int j = flat_index % out_features;
            const float* x_row = x.data.data() + row * in_features;
            int block_base = n_blocks_per_row * j;
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            float sum_scaler = 0.0f;
            for (int b = 0; b < n_blocks_per_row; ++b) {
                const BlockQ80& block = weight[block_base + b];
                float d = Fp162Float(block.scale);
                int base_k = b * kQ80BlockSize;
                int k_end = std::min(base_k + kQ80BlockSize, in_features);
                // Process 8 elements at a time (2x float32x4_t)
                int k = base_k;
                for(;k + 8 <= k_end;k += 8){
                    int8x8_t q8 = vld1_s8(block.qs + (k - base_k));
                    int16x8_t q16 = vmovl_s8(q8);
                    int32x4_t q32_lo = vmovl_s16(vget_low_s16(q16));
                    int32x4_t q32_hi = vmovl_s16(vget_high_s16(q16));
                    float32x4_t qf_lo = vcvtq_f32_s32(q32_lo);
                    float32x4_t qf_hi = vcvtq_f32_s32(q32_hi);
                    qf_lo = vmulq_n_f32(qf_lo, d);
                    qf_hi = vmulq_n_f32(qf_hi, d);
                    float32x4_t xf_lo = vld1q_s32(x_row + k);
                    float32x4_t xf_hi = v1d1q_s32(x_row + k + 4);
                    sum_vec = vfmaq_f32(sum_vec, qf_lo, xf_lo);
                    sum_vec = vfmaq_f32(sum_vec, qf_hi, xf_hi);
                }
                float32x2_t sum_lo = vget_low_f32(sum_vec);
                float32x2_t sum_hi = vget_high_f32(sum_vec);
                sum_lo = vadd_f32(sum_lo, sum_hi);
                float32x2_t sum_fp = vpadd_f32(sum_lo, sum_lo);
                sum_scalar += vget_lane_f32(sum_fp, 0);
                sum_vec = vdupq_n_f32(0.0f);
                for (; k < k_end; ++k) {
                    float w = d * static_cast<float>(block.qs[k - base_k]);
                    sum_scalar += w * x_row[k];
                }
            }
            result.data[static_cast<size_t>(row) * out_features + j] = sum_scalar;
        }
    };
    PrallelFor(rows * out_features, f);
    return result;
    #elif defined(MINI_LLAMA_USE_AVX2)
    // 逐输出元素并行的原始内核（改造前实现），用于分发路径对比。
    if (GetLinearQ80Mode() == LinearQ80Mode::kPerElementSimd) {
        LinearQ80PerElementAvx2(x, weight, rows, in_features, out_features,
                                n_blocks_per_row, result);
        return result;
    }
    // 分块多线程：与 F32 LinearThreadedSimd 的方案一致，把输出按 row_blocks x
    // col_blocks 的 tile 划分，对 tile 编号做 ParallelFor（而不是逐输出元素
    // 切分），tile 内复用同一个 x_row 指针。内层仍由 MicroKernelQ80Avx2 沿用
    // 四路 FMA 整块内核 + 尾块标量回退。
    const int row_blocks = 8;
    const int n_row_blocks = (rows + row_blocks - 1) / row_blocks;
    // ParallelFor 在任务数 < 线程数 * kParallelForMinChunk 时会退化为单线程
    // 内联执行。默认 8 列 tile 在 out_features 较窄时（如 896 -> 112 个 tile）
    // 达不到阈值，会丢掉并行度；这里自适应缩小 tile 宽度，保持与改造前相同的
    // 多线程粒度。若缩到 1 列仍达不到阈值，说明本来就会内联，则保留 8 列以
    // 最大化 x_row 的跨列复用。
    const int min_tiles = GetThreadCount() * kParallelForMinChunk;
    int col_blocks = 8;
    for (int cand = 8; cand >= 1; cand /= 2) {
        const int n_tiles = n_row_blocks * ((out_features + cand - 1) / cand);
        if (n_tiles >= min_tiles) {
            col_blocks = cand;
            break;
        }
    }
    const int n_col_blocks = (out_features + col_blocks - 1) / col_blocks;

    auto f = [&](int begin, int end) {
        for (int b = begin; b < end; ++b) {
            int r0 = (b / n_col_blocks) * row_blocks;
            int j0 = (b % n_col_blocks) * col_blocks;
            int r_nums = std::min(rows - r0, row_blocks);
            int j_nums = std::min(out_features - j0, col_blocks);
            MicroKernelQ80Avx2(x.data.data(), weight.data(), result.data.data(),
                               r0, j0, r_nums, j_nums,
                               in_features, out_features, n_blocks_per_row);
        }
    };
    ParallelFor(n_col_blocks * n_row_blocks, f);
    return result;
    #endif
}

Tensor LinearQ40(const Tensor& x, const std::vector<BlockQ40>& weight, const std::vector<int>& weight_shape){
    #if !defined(MINI_LLAMA_USE_AVX2) && !defined(MINI_LLAMA_USE_NEON)
    return LinearQuantizedImpl<BlockQ40, kQ40BlockSize>(x, weight, weight_shape, DequantQ40);
    #else
    if (weight_shape.size() != 2) {
        throw std::runtime_error("LinearQ40: expected 2D weight shape");
    }

    int in_features;
    int rows = 1;
    bool is_1d = false;
    if (x.dim() == 1) {
        in_features = x.shape[0];
        is_1d = true;
    } else if (x.dim() == 2) {
        rows = x.shape[0];
        in_features = x.shape[1];
    } else {
        std::cerr << "x shape: " << std::endl;
        x.TensorPrint();
        throw std::runtime_error(
            "LinearQ40: expected x shape [in_features] or [batch, in_features]");
    }

    int out_features = weight_shape[0];
    if (weight_shape[1] != in_features) {
        std::cerr << "x shape: " << std::endl;
        x.TensorPrint();
        std::cerr << "weight shape: " << QuantizedTensor{QuantType::kF32, weight_shape}.ShapeString() << std::endl;
        throw std::runtime_error("LinearQ40: dimension mismatch");
    }

    int n_blocks_per_row = (in_features + kQ40BlockSize - 1) / kQ40BlockSize;
    size_t expected_blocks = static_cast<size_t>(out_features) * n_blocks_per_row;
    if (weight.size() != expected_blocks) {
        throw std::runtime_error("LinearQ40: block count mismatch: expected " +
                                std::to_string(expected_blocks) + ", got " +
                                std::to_string(weight.size()));
    }

    Tensor result(is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features}, 0.0f);
    #endif
    #ifdef MINI_LLAMA_USE_NEON
    auto f = [&](int begin, int end){
        for (int j = begin; j < end; ++j) {
            int block_base = j * n_blocks_per_row;
            for(int r = 0;r < rows;r++){
                const float* x_row = x.data.data() + static_cast<size_t>(r) * in_features;
                float sum_scalar = 0.0f;

                for (int b = 0; b < n_blocks_per_row; ++b) {
                    const BlockQ40& block = weight[block_base + b];
                    float d = Fp162Float(block.scale);
                    int base_k = b * kQ40BlockSize;
                    int k_end = std::min(base_k + kQ40BlockSize, in_features);
                    int len = k_end - base_k;

                    // Process full 16-value half-blocks with NEON. Partial tails
                    // stay scalar so loads never cross the valid input range.
                    int half = 0;
                    for (; half < 2 && base_k + (half + 1) * 16 <= k_end; ++half) {
                        uint8x16_t packed = vld1q_u8(block.qs);
                        uint8x16_t nibble8;
                        uint8x16_t nibble8 = (half == 0) ? vandq_u8(packed, vdupq_n_u8(0x0F))
                                                         : vshrq_n_u8(packed, 4);
                        int8x16_t q8 = vsubq_s8(vreinterpretq_s8_u8(nibble8), vdupq_n_s8(8));

                        int16x8_t q16_lo = vmovl_s8(vget_low_s8(q8));
                        int16x8_t q16_hi = vmovl_s8(vget_high_s8(q8));

                        int32x4_t q32_0 = vmovl_s16(vget_low_s16(q16_lo));
                        int32x4_t q32_1 = vmovl_s16(vget_high_s16(q16_lo));
                        int32x4_t q32_2 = vmovl_s16(vget_low_s16(q16_hi));
                        int32x4_t q32_3 = vmovl_s16(vget_high_s16(q16_hi));

                        float32x4_t qf_0 = vmulq_n_f32(vcvtq_f32_s32(q32_0), d);
                        float32x4_t qf_1 = vmulq_n_f32(vcvtq_f32_s32(q32_1), d);
                        float32x4_t qf_2 = vmulq_n_f32(vcvtq_f32_s32(q32_2), d);
                        float32x4_t qf_3 = vmulq_n_f32(vcvtq_f32_s32(q32_3), d);

                        int k_off = base_k + half * 16;
                        float32x4_t xf_0 = vld1q_f32(x_row + k_off);
                        float32x4_t xf_1 = vld1q_f32(x_row + k_off + 4);
                        float32x4_t xf_2 = vld1q_f32(x_row + k_off + 8);
                        float32x4_t xf_3 = vld1q_f32(x_row + k_off + 12);

                        float32x4_t sum0 = vmulq_f32(qf_0, xf_0);
                        float32x4_t sum1 = vmlaq_f32(sum0, qf_1, xf_1);
                        float32x4_t sum2 = vmlaq_f32(sum1, qf_2, xf_2);
                        float32x4_t sum3 = vmlaq_f32(sum2, qf_3, xf_3);

                        // Horizontal reduce
                        float32x2_t r_lo = vget_low_f32(sum3);
                        float32x2_t r_hi = vget_high_f32(sum3);
                        r_lo = vadd_f32(r_lo, r_hi);
                        float32x2_t r = vpadd_f32(r_lo, r_lo);
                        sum_scalar += vget_lane_f32(r, 0);
                    }

                    // Tail: should not happen for kQ40BlockSize=32 unless in_features is
                    // odd
                    int k = base_k + half * 16;
                    for (int k = base_k + (half * 16); k < k_end; ++k) {
                        int idx = k - base_k;
                        int q = (idx < 16) ? (block.qs[idx] & 0x0F) - 8
                                           : (block.qs[idx - 16] >> 4) - 8;
                        sum_scalar += (d * static_cast<float>(q)) * x_row[k];
                    }
                }
            result.data[static_cast<size_t>(r) * out_features + j] = sum_scalar;
            }
        }
    };
    ParallelFor(out_features, f);
    return result;
    #elif MINI_LLAMA_USE_AVX2
    auto f = [&](int begin, int end){
        for(int j = begin;j < end;j++){
            int block_base = j * n_blocks_per_row;
            for (int r = 0; r < rows; ++r) {
                const float* x_row = x.data.data() + static_cast<size_t>(r) * in_features;
                float sum_scaler = 0.0f;
                for(int b = 0;b < n_blocks_per_row;b++){
                    const BlockQ40& block = weight[block_base + b];
                    float d = Fp162Float(block.scale);
                    int base_k = b * kQ40BlockSize;
                    int k_end = std::min(base_k + kQ40BlockSize, in_features);
                    int len = k_end - base_k;
                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    __m256 acc3 = _mm256_setzero_ps();
                    if(len == kQ40BlockSize){
                        const __m128i packed = _mm_loadu_si128((const __m128i*)(block.qs));
                        const __m128i mask_low = _mm_set1_epi8(0x0F);
                        const __m128i zp = _mm_set1_epi8(8);
                        const __m256 d_vec = _mm256_set1_ps(d);
                        for(int half = 0;half < 2;half++){
                            __m128i nibble;
                            if(half == 0){
                                //每字节取低4位
                                nibble = _mm_and_si128(packed, mask_low);
                            } else {
                                //每字节取高4位
                                nibble = _mm_and_si128(_mm_srli_epi16(packed, 4), mask_low);
                            }
                            __m128i q8 = _mm_sub_epi8(nibble, zp);
                            __m256i q16_lo = _mm256_cvtepi8_epi16(q8);
                            __m256i q16_hi = _mm256_cvtepi8_epi16(_mm_srli_si128(q8, 8));
                            __m256 q32_0 = _mm256_mul_ps(_mm256_cvtepi32_ps(I16x8ToI32x8(q16_lo)), d_vec);
                            __m256 q32_1 = _mm256_mul_ps(_mm256_cvtepi32_ps(I16x8ToI32x8(q16_hi)), d_vec);

                            int k_off = base_k + half * 16;
                            __m256 xf_0 = _mm256_loadu_ps(x_row + k_off);
                            __m256 xf_1 = _mm256_loadu_ps(x_row + k_off + 8);

                            if (half == 0) {
                                acc0 = _mm256_fmadd_ps(q32_0, xf_0, acc0);
                                acc1 = _mm256_fmadd_ps(q32_1, xf_1, acc1);
                            } else {
                                acc2 = _mm256_fmadd_ps(q32_0, xf_0, acc2);
                                acc3 = _mm256_fmadd_ps(q32_1, xf_1, acc3);
                            }
                        }
                        __m256 sum = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
                        sum_scaler += HsumF32x8(sum);
                    } else {
                        for(int k = base_k;k < k_end;k++){
                            int idx = k - base_k;
                            int q;
                            if(idx < 16){
                                q = static_cast<int>(block.qs[idx] & 0x0F) - 8;
                            } else {
                                q = static_cast<int>(block.qs[idx - 16] >> 4) - 8;
                            }
                            float w = d * static_cast<float>(q);
                            sum_scaler += w * x_row[k];
                        }
                    }
                }
                result.data[static_cast<size_t>(r) * out_features + j] = sum_scaler;
            }
        }
    };
    ParallelFor(out_features, f);
    return result;
    #endif
}

Tensor LinearQ41(const Tensor& x, const std::vector<BlockQ41>& weight,
                 const std::vector<int>& weight_shape) {
  return LinearQuantizedImpl<BlockQ41, kQ41BlockSize>(x, weight, weight_shape,
                                                      DequantQ41);
}

Tensor MatmulQ80(const std::vector<BlockQ80>& weight, const Tensor& input,
                 const std::vector<int>& weight_shape) {
  Tensor weight_f32 = DequantizedFromQ80(weight, weight_shape);
  return MatrixMulti(weight_f32, input);
}

Tensor MatmulQ40(const std::vector<BlockQ40>& weight, const Tensor& input,
                 const std::vector<int>& weight_shape) {
  Tensor weight_f32 = DequantizedFromQ40(weight, weight_shape);
  return MatrixMulti(weight_f32, input);
}

// ---------------------------------------------------------------------------
// Benchmark helpers
// ---------------------------------------------------------------------------
float CompareMatmulError(const Tensor& weight, const Tensor& input) {
    Tensor f32_result = MatrixMulti(weight, input);
    std::vector<BlockQ80> qweight = QuantizedToQ80(weight);
    Tensor q8_result = MatmulQ80(qweight, input, weight.shape);

    if (f32_result.shape != q8_result.shape) {
        throw std::runtime_error("CompareMatmulError: shape mismatch");
    }

    float max_err = 0.0f;
    for (size_t i = 0; i < f32_result.TensorNum(); ++i) {
        float err = std::abs(f32_result.data[i] - q8_result.data[i]);
        if (err > max_err) {
        max_err = err;
        }
    }
    return max_err;
}

float CompareQ40Error(const Tensor& weight, const Tensor& input) {
    Tensor f32_result = Linear(input, weight);
    std::vector<BlockQ40> qweight = QuantizedToQ40(weight);
    Tensor q4_result = LinearQ40(input, qweight, weight.shape);

    if (f32_result.shape != q4_result.shape) {
        throw std::runtime_error("CompareQ40Error: shape mismatch");
    }

    float max_err = 0.0f;
    for (size_t i = 0; i < f32_result.TensorNum(); ++i) {
        float err = std::abs(f32_result.data[i] - q4_result.data[i]);
        if (err > max_err) {
        max_err = err;
        }
    }
    return max_err;
    }

}


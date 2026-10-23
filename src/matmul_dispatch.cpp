#include "../include/matmul_dispatch.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "../include/thread_pool.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define MINI_LLAMA_USE_NEON 1
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define MINI_LLAMA_USE_AVX2 1
#endif
namespace LLaMA_Tensor{
static void CheckTensorZero(const Tensor& x){
    if(!x.TensorNum()){
        throw std::out_of_range("zero dimension tensor is not permitted");
    }
}

static void CheckTensorMulti(const Tensor& t1, const Tensor& t2){
    CheckTensorZero(t1);
    CheckTensorZero(t2);
    if(t1.dim() > 2 || t2.dim() > 2){
        throw std::runtime_error("MatrixMulti only supports 1D vectors and 2D matrices");
    }
    //一维向量方向约定(PyTorch matmul 风格):
    //  1D 左操作数视为行向量 {1, n}, 其"列数"为 shape[0];
    //  1D 右操作数视为列向量 {n, 1}, 其"行数"恒为 shape[0].
    const int t1_cols = (t1.dim() == 1) ? t1.shape[0] : t1.shape[1];
    const int t2_rows = t2.shape[0];
    if(t1_cols != t2_rows){
        t1.TensorPrint();
        t2.TensorPrint();
        throw std::runtime_error("tensors' dimensions are not matched");
    }
}

//标量微内核(4x4 = 16个XMM寄存器)
static void MicroKernel4x4Scaler(const float* x, const float* weight, float* out, int r0
    , int j0, int r_nums, int j_nums, int in_features, int out_features){

    float acc[8][8] = {};
    for(int k = 0;k < in_features;k++){
        float xv[8], wv[8];
        for(int r = 0;r < r_nums;r++){
            xv[r] = x[(r0 + r) * in_features + k];
        }
        for(int j = 0;j < j_nums;j++){
            wv[j] = weight[(j0 + j) * in_features + k];
        }
        for(int r = 0;r < r_nums;r++){
            for(int j = 0;j < j_nums;j++){
                acc[r][j] += xv[r] * wv[j];
            }
        }
    }
    for(int r = 0;r < r_nums;r++){
        for(int j = 0;j < j_nums;j++){
            out[(r0 + r) * out_features + (j0 + j)] = acc[r][j];
        }
    }
}

static Tensor MatmulNaive(const Tensor& t1, const Tensor& t2) {
    CheckTensorMulti(t1, t2);
    //一维操作数按行/列向量提升: 1D 左->行 {1,n}, 1D 右->列 {n,1}
    const int M = (t1.dim() == 1) ? 1 : t1.shape[0];
    const int K = (t1.dim() == 1) ? t1.shape[0] : t1.shape[1];
    const int N = (t2.dim() == 1) ? 1 : t2.shape[1];
    const bool out2d = (t1.dim() == 2 && t2.dim() == 2);
    Tensor multi_tensor;
    multi_tensor.data = std::vector<float>(M * N, 0.0f);
    //输出形状按 PyTorch matmul 语义去掉被提升的向量维度:
    //  1D×1D -> {1}(点积标量), 1D×2D -> {N}, 2D×1D -> {M}, 2D×2D -> {M,N}
    if(out2d){
        multi_tensor.shape = {M, N};
    }else if(t1.dim() == 1 && t2.dim() == 1){
        multi_tensor.shape = {1};
    }else{
        multi_tensor.shape = {(t1.dim() == 1) ? N : M};
    }
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            const float a = (t1.dim() == 1) ? t1.At1(k) : t1.At2(i, k);
            if(out2d){
                for (int j = 0; j < N; ++j) {
                    multi_tensor.At2(i, j) += a * t2.At2(k, j);
                }
            }else if(t2.dim() == 1){
                multi_tensor.At1(i) += a * t2.At1(k);
            }else{ // t1 1D, t2 2D (M == 1), 输出 1D {N}
                for (int j = 0; j < N; ++j) {
                    multi_tensor.At1(j) += a * t2.At2(k, j);
                }
            }
        }
    }
    return multi_tensor;
}

static Tensor LinearNaive(const Tensor& x, const Tensor& weight) {
    int in_features = weight.shape[1];
    int out_features = weight.shape[0];
    bool is_1d = (x.dim() == 1);
    int rows = (x.dim() == 2) ? x.shape[0] : 1;

    Tensor result(is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features}, 0.0f);
    const int row_blocks = 4;
    const int col_blocks = 4;
    for(int r0 = 0;r0 < rows;r0 += row_blocks){
        int r_nums = std::min(row_blocks, rows - r0);
        for(int j0 = 0;j0 < out_features;j0 += col_blocks){
            int j_nums = std::min(col_blocks, out_features - j0);
            MicroKernel4x4Scaler(x.data.data(), weight.data.data(), result.data.data()
            , r0, j0, r_nums, j_nums, in_features, out_features);
        }
    }
    return result;
}

static Tensor MatmulThreaded(const Tensor& a, const Tensor& b) {
    int m = a.shape[0];
    int inner_dim = a.shape[1];
    int n = b.shape[1];
    Tensor c({m, n}, 0.0f);
    ParallelFor(m, [&](int begin, int end) {
        for (int i = begin; i < end; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < inner_dim; ++k) {
                    sum += a.data[i * inner_dim + k] * b.data[k * n + j];
                }
                c.data[i * n + j] = sum;
            }
        }
    });
    return c;
}

static Tensor LinearThreaded(const Tensor& x, const Tensor& weight) {
    int in_features  = weight.shape[1];
    int out_features = weight.shape[0];

    bool is_1d = (x.dim() == 1);
    int rows = 1;
    if (x.dim() == 1) {
        rows = 1;
        if (x.shape[0] != in_features) throw std::runtime_error("Linear: dimension mismatch");
    } else if (x.dim() == 2) {
        rows = x.shape[0];
        if (x.shape[1] != in_features) throw std::runtime_error("Linear: dimension mismatch");
    } else {
        throw std::runtime_error("Linear: expected x shape [in_features] or [rows, in_features]");
    }

    Tensor result(is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features}, 0.0f);
    
    const int row_blocks = 4;
    const int col_blocks = 4;
    const int n_row_blocks = (rows + row_blocks - 1) / row_blocks;
    const int n_col_blocks = (out_features + col_blocks - 1) / col_blocks;

    auto f = [&](int begin, int end){
        for(int b = begin;b < end;b++){
            // 块索引按行列展开: 行块 = b / n_col_blocks, 列块 = b % n_col_blocks
            int r0 = (b / n_col_blocks) * row_blocks;
            int j0 = (b % n_col_blocks) * col_blocks;
            int r_nums = std::min(rows - r0, row_blocks);
            int j_nums = std::min(out_features - j0, col_blocks);
            MicroKernel4x4Scaler(x.data.data(), weight.data.data(), result.data.data()
            , r0, j0, r_nums, j_nums, in_features, out_features);
        }
    };
    ParallelFor(n_col_blocks * n_row_blocks, f);
    return result;
}

#ifdef MINI_LLAMA_USE_NEON
static float DotSimdNeon(const float* a, const float* b, int n) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum = vfmaq_f32(sum, va, vb);
    }
    float32x2_t r = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    r = vpadd_f32(r, r);
    float result = vget_lane_f32(r, 0);
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}
#endif

#ifdef MINI_LLAMA_USE_AVX2
//AVX2向量微内核
static void MicroKernel8x8Avx2(const float* x, const float* w, float* out,
                               int r0, int j0, int rb, int jb,
                               int in_features, int out_features) {
    // 边界保护: rb/jb 为实际参与计算的行/列数, 越界写会破坏相邻行甚至堆内存
    if (rb <= 0 || jb <= 0) {
        return;
    }
    // 尾部列块不足 8 列时, AVX2 的 gather/store 都是 8 宽, 会越界读写,
    // 因此回退到逐元素标量微内核
    if (jb < 8) {
        MicroKernel4x4Scaler(x, w, out, r0, j0, rb, jb, in_features, out_features);
        return;
    }
    __m256 acc[8];
    for (int r = 0; r < rb; ++r){ 
        acc[r] = _mm256_setzero_ps();
    }
    __m256i j_idx = _mm256_setr_epi32(0, in_features, 2*in_features, 3*in_features,
                                      4*in_features, 5*in_features, 6*in_features, 7*in_features);

    for (int k = 0; k < in_features; ++k) {
        __m256 wcol = _mm256_i32gather_ps(w + j0 * in_features + k, j_idx, 4);
        for (int r = 0; r < rb; ++r) {
            __m256 xb = _mm256_set1_ps(x[(r0 + r) * in_features + k]);
            acc[r] = _mm256_fmadd_ps(xb, wcol, acc[r]);
        }
    }
    for (int r = 0; r < rb; ++r){
        _mm256_storeu_ps(out + (r0 + r) * out_features + j0, acc[r]);
    }
}

static Tensor LinearSimd(const Tensor& x, const Tensor& weight){
    int in_features = weight.shape[1];
    int out_features = weight.shape[0];
    bool is_1d = (x.dim() == 1);
    int rows = (x.dim() == 2) ? x.shape[0] : 1;

    Tensor result(is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features}, 0.0f);
    const int row_blocks = 8;
    const int col_blocks = 8;
    for(int r0 = 0;r0 < rows;r0 += row_blocks){
        int r_nums = std::min(row_blocks, rows - r0);
        for(int j0 = 0;j0 < out_features;j0 += col_blocks){
            int j_nums = std::min(col_blocks, out_features - j0);
            MicroKernel8x8Avx2(x.data.data(), weight.data.data(), result.data.data()
            , r0, j0, r_nums, j_nums, in_features, out_features);
        }
    }
    return result;
}

static Tensor LinearThreadedSimd(const Tensor& x, const Tensor& weight){
    int in_features  = weight.shape[1];
    int out_features = weight.shape[0];

    bool is_1d = (x.dim() == 1);
    int rows = 1;
    if (x.dim() == 1) {
        rows = 1;
        if (x.shape[0] != in_features) throw std::runtime_error("Linear: dimension mismatch");
    } else if (x.dim() == 2) {
        rows = x.shape[0];
        if (x.shape[1] != in_features) throw std::runtime_error("Linear: dimension mismatch");
    } else {
        throw std::runtime_error("Linear: expected x shape [in_features] or [rows, in_features]");
    }

    Tensor result(is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features}, 0.0f);
    
    const int row_blocks = 8;
    const int col_blocks = 8;
    const int n_row_blocks = (rows + row_blocks - 1) / row_blocks;
    const int n_col_blocks = (out_features + col_blocks - 1) / col_blocks;

    auto f = [&](int begin, int end){
        for(int b = begin;b < end;b++){
            // 块索引按行列展开: 行块 = b / n_col_blocks, 列块 = b % n_col_blocks
            int r0 = (b / n_col_blocks) * row_blocks;
            int j0 = (b % n_col_blocks) * col_blocks;
            int r_nums = std::min(rows - r0, row_blocks);
            int j_nums = std::min(out_features - j0, col_blocks);
            MicroKernel8x8Avx2(x.data.data(), weight.data.data(), result.data.data()
            , r0, j0, r_nums, j_nums, in_features, out_features);
        }
    };  
    ParallelFor(n_col_blocks * n_row_blocks, f);
    return result;
}
#endif

MatmulMode DefaultMatmulMode() { return MatmulMode::kThreadedSimd; }

// 仅改变形状(元素总数不变), 用于一维提升 / 结果降维
static Tensor Reshaped(const Tensor& t, const std::vector<int>& shape) {
  Tensor out;
  out.data = t.data;
  out.shape = shape;
  return out;
}

Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode) {
  if (a.dim() < 1 || b.dim() < 1 || a.dim() > 2 || b.dim() > 2) {
    throw std::runtime_error("MatmulDispatch: expected 1D or 2D tensors");
  }
  if (mode == MatmulMode::kNaive) {
    // MatmulNaive 内部已实现 PyTorch 的一维提升语义(1Dx1D/1Dx2D/2Dx1D)
    return MatmulNaive(a, b);
  }
  // 其余模式基于二维实现: 先把一维操作数提升为行/列向量, 算完再按
  // PyTorch 语义去掉被提升的维度(1Dx1D -> {1}, 1Dx2D -> {N}, 2Dx1D -> {M})
  const Tensor a2 = (a.dim() == 1) ? Reshaped(a, {1, a.shape[0]}) : a;
  const Tensor b2 = (b.dim() == 1) ? Reshaped(b, {b.shape[0], 1}) : b;
  if (a2.shape[1] != b2.shape[0]) {
    throw std::runtime_error("MatmulDispatch: dimension mismatch");
  }
  const Tensor c = MatmulThreaded(a2, b2);
  if (a.dim() == 1 && b.dim() == 1) {
    return Reshaped(c, {1});
  }
  if (a.dim() == 1) {
    return Reshaped(c, {c.shape[1]});
  }
  if (b.dim() == 1) {
    return Reshaped(c, {c.shape[0]});
  }
  return c;
}

Tensor LinearDispatch(const Tensor& x, const Tensor& weight, MatmulMode mode) {
  if (weight.dim() != 2) {
    throw std::runtime_error(
        "LinearDispatch: expected weight shape [out_features, in_features]");
  }
  int in_features;
  if (x.dim() == 1) {
    in_features = x.shape[0];
  } else if (x.dim() == 2) {  // 批量输入 [rows, in_features]; rows == 1 亦走此分支
    in_features = x.shape[1];
  } else {
    throw std::runtime_error(
        "LinearDispatch: expected x shape [in_features] or [rows, in_features]");
  }
  if (weight.shape[1] != in_features) {
    throw std::runtime_error("LinearDispatch: dimension mismatch");
  }
  switch (mode) {
    case MatmulMode::kNaive:
      return LinearNaive(x, weight);
    case MatmulMode::kThreaded:
      return LinearThreaded(x, weight);
    case MatmulMode::kSimd:
#ifdef MINI_LLAMA_USE_AVX2
      return LinearSimd(x, weight);
#else
      return LinearThreaded(x, weight);
#endif
    case MatmulMode::kThreadedSimd:
#ifdef MINI_LLAMA_USE_AVX2
      return LinearThreadedSimd(x, weight);
#else
      return LinearThreaded(x, weight);
#endif
  }
#ifdef MINI_LLAMA_USE_AVX2
  return LinearThreadedSimd(x, weight);
#else
  return LinearThreaded(x, weight);
#endif
}
}

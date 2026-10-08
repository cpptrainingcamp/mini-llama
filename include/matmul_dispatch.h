#pragma once

#include "tensor.h"

namespace LLaMA_Tensor{
enum class MatmulMode{
    kNaive,
    kThreaded,
    kSimd,
    kThreadedSimd,
};

MatmulMode DefaultMatmulMode();

Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode);

Tensor LinearDispatch(const Tensor& x, const Tensor& weight, MatmulMode mode);

}
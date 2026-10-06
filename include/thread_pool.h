#pragma once

#include <functional>
#include <thread>
#include <vector>

namespace LLaMA_Tensor{
//获取线程数
int GetThreadCount();
//设置并行计算线程数
void SetThreadCount(int n);
//切分任务[0,n)为尽可能相等区间，每个区间并行进行fn处理
void ParallelFor(int n, const std::function<void(int begin, int end)>& fn);
}
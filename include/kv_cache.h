#pragma once

#include "tensor.h"

namespace LLaMA_Tensor{
    struct KVCache{
        //KV缓存分别张量存储
        Tensor KCache;
        Tensor VCache;
        //layers:注意力层；max_seq_len:最大token序列长度；n_kv_heads:KV头数；head_dim:头维度
        KVCache() = default;
        KVCache(int layers, int max_seq_len, int n_kv_heads, int head_dim);
        //将一个token的k,v写入缓存
        void Write(int layer, int pos, const Tensor& k, const Tensor& v);
        //将一个token的k,v行数据写入缓存 (批量路径用)
        //  k/v 均指向 n_kv_heads * head_dim 个连续 float
        void WriteRow(int layer, int pos, const float* k, const float* v);
        //获取KCache行向量地址
        const float* KeyPtr(int layer, int pos, int kv_head) const;
        //获取VCache行向量地址
        const float* ValuePtr(int layer, int pos, int kv_head) const;
    };
}

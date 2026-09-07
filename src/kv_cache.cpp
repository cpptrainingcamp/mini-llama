#include "kv_cache.h"
#include <cstring>

namespace LLaMA_Tensor{
    KVCache::KVCache(int layers, int max_seq_len, int n_kv_heads, int head_dim){
        KCache = MakeTensor4D(layers, max_seq_len, n_kv_heads, head_dim, 0.0f);
        VCache = MakeTensor4D(layers, max_seq_len, n_kv_heads, head_dim, 0.0f);
    }

    void KVCache::Write(int layer, int pos, const Tensor& k, const Tensor& v){
        if(layer < 0 || layer >= KCache.shape[0] ||
            pos < 0 || pos   >= KCache.shape[1]){
            throw std::out_of_range("kv cache index out of range");
        }
        size_t base = KCache.FlatIndex({layer, pos, 0, 0});
        if(k.shape != std::vector{KCache.shape[2], KCache.shape[3]} 
            || v.shape != std::vector{VCache.shape[2], VCache.shape[3]}){
            throw std::runtime_error("Tensor to write must be n_kv_heads * head_dim");
        }
        memcpy(KCache.data.data() + base, k.data.data(), k.TensorNum() * sizeof(float));
        memcpy(VCache.data.data() + base, v.data.data(), v.TensorNum() * sizeof(float));
    }

    const float* KVCache::KeyPtr(int layer, int pos, int kv_head) const{
        if(layer < 0 || layer >= KCache.shape[0] ||
            pos < 0 || pos   >= KCache.shape[1] ||
            kv_head < 0 || kv_head >= KCache.shape[2]){
            throw std::out_of_range("kv cache index out of range");
        }
        size_t base = KCache.FlatIndex({layer, pos, kv_head, 0});
        return KCache.data.data() + base;
    }

    const float* KVCache::ValuePtr(int layer, int pos, int kv_head) const{
        if(layer < 0 || layer >= VCache.shape[0] ||
            pos   < 0 || pos   >= VCache.shape[1] ||
            kv_head < 0 || kv_head >= VCache.shape[2]){
            throw std::out_of_range("kv cache index out of range");
        }
        size_t base = VCache.FlatIndex({layer, pos, kv_head, 0});
        return VCache.data.data() + base;
    }
}

#include "../include/forward.h"
#include "../include/cuda_attention.h"
#include "../include/cuda_matmul.h"
#include "../include/cuda_ops.h"
#include "../include/cuda_quant.h"
#include "../include/ops.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace LLaMA_Tensor{
namespace{
//模型结构检查（单 token 与批量路径共用）
void ValidateModelInputs(const MiniLlamaModel& model){
    if(!model.loaded){
        throw std::runtime_error("model is not loaded");
    }
    if(model.layers.size() != model.config.n_layers){
        throw std::runtime_error("model's layers are not mathed with config's, need: "
             + std::to_string(model.config.n_layers));
    }
    if(model.token_embedding.TensorNum() < model.config.vocab_size * model.config.dim){
        throw std::runtime_error("token embedding's size is not enough, it should be >= "
             + std::to_string(model.config.vocab_size));
    }
    model.token_embedding.CheckDim(2);
    if(model.token_embedding.shape[1] != model.config.dim){
        throw std::runtime_error("the token embedding vector dimension is not matched with the config's, it should be: "
             + std::to_string(model.config.dim));
    }
}

//输入检查
void ValidateForwardInputs(const Context& ctx, const MiniLlamaModel& model, int token){
    ValidateModelInputs(model);
    if(token < 0 || token >= model.config.vocab_size){
        throw std::out_of_range("can not find this token: " + std::to_string(token));
    }
    if(ctx.pos < 0 || ctx.pos >= model.config.max_seq_len){
        throw std::out_of_range("tokens are out of the context size");
    }
}
//张量形状判断
void RequireShape(const Tensor& t, const std::vector<int>& expected, const std::string& caller){
    if(t.shape != expected){
        throw std::runtime_error(caller + "'s tensor dimension is not expected");
    }
}

Tensor AddOptionalBias(const Tensor& x, const Tensor& bias, const std::string& caller){
    if(bias.data.empty()){
        return x;
    }
    if(x.dim() != 1 || bias.dim() != 1 || x.shape[0] != bias.shape[0]){
        throw std::runtime_error(caller + "'s tensor is not matched with bias");
    }
    Tensor result = x;
    for(size_t i = 0;i<result.data.size();i++){
        result.data[i] += bias.data[i];
    }
    return result;
}

//批量行偏置: x [B, out], bias [out]（bias 为空则原样返回）
Tensor AddOptionalBiasRows(const Tensor& x, const Tensor& bias, const std::string& caller){
    if(bias.data.empty()){
        return x;
    }
    if(x.dim() != 2 || bias.dim() != 1 || x.shape[1] != bias.shape[0]){
        throw std::runtime_error(caller + "'s tensor is not matched with bias");
    }
    Tensor result = x;
    for(int r = 0;r < x.shape[0];r++){
        float* row = result.data.data() + static_cast<size_t>(r) * x.shape[1];
        for(int j = 0;j < x.shape[1];j++){
            row[j] += bias.data[j];
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// CUDA 设备路径辅助函数
// ---------------------------------------------------------------------------
const CudaUploadedWeight* FindCudaWeight(const MiniLlamaModel& model,
                                         const std::string& name,
                                         QuantType type){
    if(!model.cuda_weights){
        return nullptr;
    }
    for(const CudaUploadedWeight& weight : model.cuda_weights->weights){
        if(weight.name == name && weight.type == type){
            return &weight;
        }
    }
    return nullptr;
}

void RecordCudaLinearCall(const MiniLlamaModel& model){
    if(model.cuda_weights){
        model.cuda_weights->linear_calls += 1;
    }
}

void RecordCudaActivationCall(const MiniLlamaModel& model, size_t calls = 1){
    if(model.cuda_weights){
        model.cuda_weights->activation_calls += calls;
    }
}

void RecordCudaAttentionCall(const MiniLlamaModel& model){
    if(model.cuda_weights){
        model.cuda_weights->attention_calls += 1;
    }
}

void RecordCudaKvCacheWrite(const MiniLlamaModel& model, size_t bytes){
    if(model.cuda_weights){
        model.cuda_weights->kv_cache_write_bytes += bytes;
    }
}

void RecordCudaKvCacheRead(const MiniLlamaModel& model, size_t bytes){
    if(model.cuda_weights){
        model.cuda_weights->kv_cache_read_bytes += bytes;
    }
}

void RecordCudaHostToDeviceCopy(const MiniLlamaModel& model, size_t bytes){
    if(model.cuda_weights){
        model.cuda_weights->host_to_device_copies += 1;
        model.cuda_weights->host_to_device_bytes += bytes;
    }
}

void RecordCudaDeviceToHostCopy(const MiniLlamaModel& model, size_t bytes){
    if(model.cuda_weights){
        model.cuda_weights->device_to_host_copies += 1;
        model.cuda_weights->device_to_host_bytes += bytes;
    }
}

int CudaDeviceId(const MiniLlamaModel& model){
    return model.cuda_weights ? model.cuda_weights->device_id : 0;
}

bool CudaLinearSupported(const QuantizedTensor& weight){
    return weight.type == QuantType::kF32 || weight.type == QuantType::kQ80 ||
           weight.type == QuantType::kQ40 || weight.type == QuantType::kQ41;
}

bool CudaLinearReady(const MiniLlamaModel& model, const std::string& name,
                     const QuantizedTensor& weight){
    return model.cuda_weights && CudaLinearSupported(weight) &&
           FindCudaWeight(model, name, weight.type) != nullptr;
}

bool HasUploadedWeight(const MiniLlamaModel& model, const std::string& name,
                       QuantType type, bool linear){
    const CudaUploadedWeight* weight = FindCudaWeight(model, name, type);
    return weight != nullptr && weight->linear == linear;
}

CudaTensor UploadCudaTensor(const MiniLlamaModel& model, const Tensor& x){
    CudaTensor x_dev = CudaTensorFromHost(x, CudaDeviceId(model));
    RecordCudaHostToDeviceCopy(model, x.size() * sizeof(float));
    return x_dev;
}

Tensor DownloadCudaTensor(const MiniLlamaModel& model, const CudaTensor& x){
    Tensor y = x.Download();
    RecordCudaDeviceToHostCopy(model, y.size() * sizeof(float));
    return y;
}

CudaTensor ForwardLinearDevice(const MiniLlamaModel& model,
                               const std::string& name, const CudaTensor& x,
                               const QuantizedTensor& weight){
    if(!CudaLinearSupported(weight)){
        throw std::runtime_error("CUDA forward Linear unsupported weight type: " +
                                 name);
    }
    const CudaUploadedWeight* cuda_weight =
        FindCudaWeight(model, name, weight.type);
    if(cuda_weight == nullptr){
        throw std::runtime_error("CUDA forward Linear missing uploaded weight: " +
                                 name);
    }

    CudaTensor y;
    switch(weight.type){
        case QuantType::kF32:
            y = CudaLinearDeviceInput(x, cuda_weight->buffer.data(),
                                      cuda_weight->shape, CudaDeviceId(model));
            break;
        case QuantType::kQ80:
            y = CudaQ80LinearDeviceInput(x, cuda_weight->buffer.data(),
                                         cuda_weight->block_count,
                                         cuda_weight->shape,
                                         CudaDeviceId(model));
            break;
        case QuantType::kQ40:
            y = CudaQ40LinearDeviceInput(x, cuda_weight->buffer.data(),
                                         cuda_weight->block_count,
                                         cuda_weight->shape,
                                         CudaDeviceId(model));
            break;
        case QuantType::kQ41:
            y = CudaQ41LinearDeviceInput(x, cuda_weight->buffer.data(),
                                         cuda_weight->block_count,
                                         cuda_weight->shape,
                                         CudaDeviceId(model));
            break;
    }
    RecordCudaLinearCall(model);
    return y;
}

Tensor ForwardLinear(const MiniLlamaModel& model, const std::string& name, const Tensor& x, const QuantizedTensor& weight){
    return Linear(x, weight);
}

void ForwardQkvProjection(const MiniLlamaModel& model, const std::string& layer_prefix
    , const Tensor& h, const LayerWeights& weight, Tensor& q_flat, Tensor& k_flat, Tensor& v_flat){
    const std::string q_name = layer_prefix + "wq";
    const std::string k_name = layer_prefix + "wk";
    const std::string v_name = layer_prefix + "wv";
    
    q_flat = AddOptionalBias(ForwardLinear(model, q_name, h, weight.wq), weight.bq, "forward_layer q");
    k_flat = AddOptionalBias(ForwardLinear(model, k_name, h, weight.wk), weight.bk, "forward_layer k");
    v_flat = AddOptionalBias(ForwardLinear(model, v_name, h, weight.wv), weight.bv, "forward_layer v");
}

CudaTensor AddOptionalBiasDevice(const MiniLlamaModel& model,
                                 const std::string& name, CudaTensor x,
                                 const Tensor& bias){
    if(bias.data.empty()){
        return x;
    }
    const CudaUploadedWeight* cuda_weight =
        FindCudaWeight(model, name, QuantType::kF32);
    if(cuda_weight == nullptr || cuda_weight->linear){
        throw std::runtime_error("CUDA forward bias missing uploaded weight: " +
                                 name);
    }
    if(cuda_weight->shape != bias.shape){
        throw std::runtime_error(
            "CUDA forward bias uploaded weight shape mismatch: " + name);
    }
    CudaTensor y = CudaElementwiseAddDeviceWeight(
        x, cuda_weight->buffer.data(), cuda_weight->shape, CudaDeviceId(model));
    RecordCudaActivationCall(model);
    return y;
}

CudaTensor AddOptionalBiasBatchDevice(const MiniLlamaModel& model,
                                      const std::string& name, CudaTensor x,
                                      const Tensor& bias){
    if(bias.data.empty()){
        return x;
    }
    const CudaUploadedWeight* cuda_weight =
        FindCudaWeight(model, name, QuantType::kF32);
    if(cuda_weight == nullptr || cuda_weight->linear){
        throw std::runtime_error(
            "CUDA forward batch bias missing uploaded weight: " + name);
    }
    if(cuda_weight->shape != bias.shape){
        throw std::runtime_error(
            "CUDA forward batch bias uploaded weight shape mismatch: " + name);
    }
    CudaTensor y = CudaElementwiseAddRowBiasDeviceWeight(
        x, cuda_weight->buffer.data(), cuda_weight->shape, CudaDeviceId(model));
    RecordCudaActivationCall(model);
    return y;
}

void ForwardQkvProjectionDevice(const MiniLlamaModel& model,
                                const std::string& layer_prefix,
                                const CudaTensor& h, const LayerWeights& lw,
                                CudaTensor& q, CudaTensor& k, CudaTensor& v){
    q = ForwardLinearDevice(model, layer_prefix + "wq", h, lw.wq);
    k = ForwardLinearDevice(model, layer_prefix + "wk", h, lw.wk);
    v = ForwardLinearDevice(model, layer_prefix + "wv", h, lw.wv);
    q = AddOptionalBiasDevice(model, layer_prefix + "bq", std::move(q), lw.bq);
    k = AddOptionalBiasDevice(model, layer_prefix + "bk", std::move(k), lw.bk);
    v = AddOptionalBiasDevice(model, layer_prefix + "bv", std::move(v), lw.bv);
}

void ForwardQkvProjectionBatchDevice(const MiniLlamaModel& model,
                                     const std::string& layer_prefix,
                                     const CudaTensor& h, const LayerWeights& lw,
                                     CudaTensor& q, CudaTensor& k,
                                     CudaTensor& v){
    q = ForwardLinearDevice(model, layer_prefix + "wq", h, lw.wq);
    k = ForwardLinearDevice(model, layer_prefix + "wk", h, lw.wk);
    v = ForwardLinearDevice(model, layer_prefix + "wv", h, lw.wv);
    q = AddOptionalBiasBatchDevice(model, layer_prefix + "bq", std::move(q),
                                   lw.bq);
    k = AddOptionalBiasBatchDevice(model, layer_prefix + "bk", std::move(k),
                                   lw.bk);
    v = AddOptionalBiasBatchDevice(model, layer_prefix + "bv", std::move(v),
                                   lw.bv);
}

CudaTensor ForwardRmsNormDevice(const MiniLlamaModel& model,
                                const std::string& name, const CudaTensor& x,
                                const Tensor& weight, float eps){
    const CudaUploadedWeight* cuda_weight =
        FindCudaWeight(model, name, QuantType::kF32);
    if(cuda_weight == nullptr || cuda_weight->linear){
        throw std::runtime_error(
            "CUDA forward RMSNorm missing uploaded weight: " + name);
    }
    if(cuda_weight->shape != weight.shape){
        throw std::runtime_error(
            "CUDA forward RMSNorm uploaded weight shape mismatch: " + name);
    }
    CudaTensor y = CudaRmsNormDeviceWeight(
        x, cuda_weight->buffer.data(), cuda_weight->shape, eps,
        CudaDeviceId(model));
    RecordCudaActivationCall(model);
    return y;
}

CudaTensor ForwardRmsNormBatchDevice(const MiniLlamaModel& model,
                                     const std::string& name,
                                     const CudaTensor& x, const Tensor& weight,
                                     float eps){
    const CudaUploadedWeight* cuda_weight =
        FindCudaWeight(model, name, QuantType::kF32);
    if(cuda_weight == nullptr || cuda_weight->linear){
        throw std::runtime_error(
            "CUDA forward batch RMSNorm missing uploaded weight: " + name);
    }
    if(cuda_weight->shape != weight.shape){
        throw std::runtime_error(
            "CUDA forward batch RMSNorm uploaded weight shape mismatch: " + name);
    }
    CudaTensor y = CudaRmsNormBatchDeviceWeight(
        x, cuda_weight->buffer.data(), cuda_weight->shape, eps,
        CudaDeviceId(model));
    RecordCudaActivationCall(model);
    return y;
}

Tensor ForwardRmsNorm(const MiniLlamaModel& model, const Tensor& x, const Tensor& weight, float eps){
    return RMSNorm(x, weight, eps);
}

Tensor ForwardSwiGlu(const MiniLlamaModel& model, const Tensor& gate, const Tensor& up){
    return SWiGLU(gate, up);
}

Tensor EmbedToken(const MiniLlamaModel& model, int token_id){
    int dim = model.config.dim;
    Tensor x({dim}, 0.0f);
    for(int i = 0;i < dim;i++){
        x.At1(i) = model.token_embedding.At2(token_id, i);
    }
    return x;
}

CudaTensor EmbedTokenDevice(const MiniLlamaModel& model, int token_id){
    const CudaUploadedWeight* cuda_weight =
        FindCudaWeight(model, "token_embedding", QuantType::kF32);
    if(cuda_weight == nullptr || cuda_weight->linear){
        throw std::runtime_error(
            "CUDA forward embedding missing uploaded weight: token_embedding");
    }
    if(cuda_weight->shape != model.token_embedding.shape){
        throw std::runtime_error(
            "CUDA forward embedding uploaded weight shape mismatch");
    }
    return CudaEmbeddingLookupDeviceWeight(cuda_weight->buffer.data(),
                                           cuda_weight->shape, token_id,
                                           CudaDeviceId(model));
}

CudaTensor EmbedTokensDeviceBatch(const MiniLlamaModel& model,
                                  const std::vector<int>& token_ids){
    const CudaUploadedWeight* cuda_weight =
        FindCudaWeight(model, "token_embedding", QuantType::kF32);
    if(cuda_weight == nullptr || cuda_weight->linear){
        throw std::runtime_error(
            "CUDA forward batch embedding missing uploaded weight: "
            "token_embedding");
    }
    if(cuda_weight->shape != model.token_embedding.shape){
        throw std::runtime_error(
            "CUDA forward batch embedding uploaded weight shape mismatch");
    }
    return CudaEmbeddingLookupBatchDeviceWeight(
        cuda_weight->buffer.data(), cuda_weight->shape, token_ids,
        CudaDeviceId(model));
}

Tensor ForwardSoftmax(const MiniLlamaModel& model, const Tensor& x){
    return softmax(x);
}

Tensor AttentionForward(const MiniLlamaModel& model, const Tensor& q, const Tensor& k, const Tensor& v
    , int pos, int layer, KVCache& kv_cache, int n_heads, int n_kv_heads, int head_dim){
    kv_cache.Write(layer, pos, k, v);
    float scale = 1.0f / sqrt(head_dim);
    Tensor attention_out({n_heads, head_dim}, 0.0f);
    //为每个位置的第h个Q头与对应的K头做点积
    auto compute = [&](int h){
        int kv_head = h / (n_heads / n_kv_heads);
        std::vector<float> score(pos + 1, 0.0f);
        for(int i=0;i < pos+1;i++){
            const float* k_ptr = kv_cache.KeyPtr(layer, i, kv_head);
            float dot = 0.0f;
            for(int d = 0;d < head_dim;d++){
                dot += q.At2(h, d) * k_ptr[d];
            }
            score[i] = dot * scale;
        }

        Tensor x({pos + 1}, 0.0f);
        x.data = std::move(score);
        Tensor prob;
        prob = ForwardSoftmax(model, x);

        for(int d=0;d<head_dim;d++){ 
            float val = 0.0f;
            for(int i=0;i<pos+1;i++){
                const float* v_ptr = kv_cache.ValuePtr(layer, i, kv_head);
                val += prob.At1(i) * v_ptr[d];
            }
            attention_out.At2(h, d) = val;
        }
    };
    
    for(int h = 0;h < n_heads;h++){
        compute(h);
    }
    return attention_out;
}

CudaTensor AttentionForwardDevice(const MiniLlamaModel& model,
                                  const CudaTensor& q, const CudaTensor& k,
                                  const CudaTensor& v, int pos, int layer,
                                  CudaKvCache& cuda_kv_cache, int n_heads,
                                  int n_kv_heads, int head_dim){
    cuda_kv_cache.WriteDevice(layer, pos, k, v);
    RecordCudaKvCacheWrite(model, (k.size() + v.size()) * sizeof(float));
    CudaTensor attn_out = CudaAttentionDecodeDeviceInput(
        q, cuda_kv_cache, layer, pos, n_heads, n_kv_heads, head_dim,
        CudaDeviceId(model));
    RecordCudaAttentionCall(model);
    RecordCudaKvCacheRead(
        model, static_cast<size_t>(pos + 1) * static_cast<size_t>(n_heads) *
                   static_cast<size_t>(head_dim) * sizeof(float) * 2);
    return attn_out;
}

Tensor FfnForward(const MiniLlamaModel& model, const std::string& layer_prefix, const Tensor& h, const LayerWeights& weights){
    const std::string gate_name = layer_prefix + "w_gate";
    const std::string up_name = layer_prefix + "w_up";
    const std::string down_name = layer_prefix + "w_down";

    Tensor gate = ForwardLinear(model, gate_name, h, weights.w_gate);
    Tensor up = ForwardLinear(model, up_name, h, weights.w_up);
    Tensor SWiGLU_tensor = ForwardSwiGlu(model, gate, up);
    Tensor down = ForwardLinear(model, down_name, SWiGLU_tensor, weights.w_down);

    return down;
}

CudaTensor FfnForwardDevice(const MiniLlamaModel& model,
                            const std::string& layer_prefix,
                            const CudaTensor& h, const LayerWeights& lw){
    CudaTensor gate =
        ForwardLinearDevice(model, layer_prefix + "w_gate", h, lw.w_gate);
    CudaTensor up = ForwardLinearDevice(model, layer_prefix + "w_up", h, lw.w_up);
    CudaTensor silu_gate = CudaSiluDeviceInput(gate, CudaDeviceId(model));
    CudaTensor ff =
        CudaElementwiseMulDeviceInput(silu_gate, up, CudaDeviceId(model));
    RecordCudaActivationCall(model, 2);
    return ForwardLinearDevice(model, layer_prefix + "w_down", ff, lw.w_down);
}

CudaTensor FfnForwardBatchDevice(const MiniLlamaModel& model,
                                 const std::string& layer_prefix,
                                 const CudaTensor& h, const LayerWeights& lw){
    CudaTensor gate =
        ForwardLinearDevice(model, layer_prefix + "w_gate", h, lw.w_gate);
    CudaTensor up = ForwardLinearDevice(model, layer_prefix + "w_up", h, lw.w_up);
    CudaTensor silu_gate = CudaSiluDeviceInput(gate, CudaDeviceId(model));
    CudaTensor ff =
        CudaElementwiseMulDeviceInput(silu_gate, up, CudaDeviceId(model));
    RecordCudaActivationCall(model, 2);
    return ForwardLinearDevice(model, layer_prefix + "w_down", ff, lw.w_down);
}

void ForwardRope(const MiniLlamaModel& model, Tensor& q, Tensor& k, int pos, float theta, RoPEtype ropetype){
    rope(q, k, pos, theta, ropetype);
}

Tensor ForwardAdd(const MiniLlamaModel& model, const Tensor& a, const Tensor& b){
    if(a.shape != b.shape){
        throw std::runtime_error("Forward add need the same shape tensors");
    }
    Tensor r(a.shape);
    for(size_t i = 0;i < a.TensorNum();i++){
        r.data[i] = a.data[i] + b.data[i];
    }
    return r;
}

CudaTensor ForwardAddDevice(const MiniLlamaModel& model,
                            const CudaTensor& a, const CudaTensor& b){
    CudaTensor y = CudaElementwiseAddDeviceInput(a, b, CudaDeviceId(model));
    RecordCudaActivationCall(model);
    return y;
}

Tensor ForwardLayer(Context& ctx, const MiniLlamaModel& model, const Tensor& embedding_vec
    , int layer, const LayerWeights& weights, const ModelConfig& config){
    int dim = config.dim;
    int n_heads = config.n_heads;
    int n_kv_heads = config.n_kv_heads;
    int head_dim = config.head_dim;   
    float eps = config.rms_norm_eps;
    int pos = ctx.pos;
    float theta = config.rope_theta;
    RoPEtype rope_type = config.rope_type;
    std::string layer_prefix = "layer." + std::to_string(layer);

    //QKV投影，分头
    Tensor h = ForwardRmsNorm(model, embedding_vec, weights.attention_norm, eps);
    Tensor q, k ,v;
    ForwardQkvProjection(model, layer_prefix, h, weights, q, k, v);
    q.ReshapeCheck({n_heads, head_dim});
    k.ReshapeCheck({n_kv_heads, head_dim});
    v.ReshapeCheck({n_kv_heads, head_dim});
    //旋转位置编码
    ForwardRope(model, q, k, pos, theta, rope_type);
    //Attention:QK得分，V加权
    Tensor att_out;
    att_out = AttentionForward(model, q, k, v, pos, layer, ctx.kv_cache, n_heads, n_kv_heads, head_dim);
    //合并头,输出投影
    att_out.ReshapeCheck({n_heads * head_dim});
    Tensor att_out_proj = ForwardLinear(model, layer_prefix + ":out_proj", att_out, weights.wo);
    RequireShape(att_out_proj, {dim}, layer_prefix + ": att_out_proj");
    //残差连接
    Tensor att_out_cnn = ForwardAdd(model, att_out_proj, embedding_vec);
    //FFN:门控激活，残差连接
    Tensor h_norm, h_out;
    h_norm = ForwardRmsNorm(model, att_out_cnn, weights.ffn_norm, eps);
    h_out = FfnForward(model, layer_prefix, h_norm, weights);
    Tensor ffn_out = ForwardAdd(model, h_out, att_out_cnn);

    return ffn_out;
}

Tensor ComputeLogits(const MiniLlamaModel& model, Tensor& x){
    Tensor norm = ForwardRmsNorm(model, x, model.final_norm, model.config.rms_norm_eps);
    Tensor logits = ForwardLinear(model, "lm_head", norm, model.lm_head);
    logits.ReshapeCheck({model.config.vocab_size});
    return logits;
}

//批量 embedding: 从 [vocab, dim] 行主序逐 token 拷贝得到 [B, dim]
Tensor EmbedBatch(const MiniLlamaModel& model, const std::vector<int>& tokens){
    const int dim = model.config.dim;
    const int batch = static_cast<int>(tokens.size());
    Tensor x({batch, dim}, 0.0f);
    const float* emb = model.token_embedding.data.data();
    for(int b = 0;b < batch;b++){
        memcpy(x.data.data() + static_cast<size_t>(b) * dim,
               emb + static_cast<size_t>(tokens[b]) * dim,
               static_cast<size_t>(dim) * sizeof(float));
    }
    return x;
}

//批量输入检查
void ValidateBatchInputs(const MiniLlamaModel& model, const Batch& batch){
    ValidateModelInputs(model);
    for(int b = 0;b < batch.num_tokens();b++){
        const int token = batch.tokens[b];
        const int pos = batch.positions[b];
        if(token < 0 || token >= model.config.vocab_size){
            throw std::out_of_range("can not find this token: " + std::to_string(token));
        }
        if(pos < 0 || pos >= model.config.max_seq_len){
            throw std::out_of_range("tokens are out of the context size");
        }
    }
}

//批量因果注意力（方案 A: 以 query 自身位置作为循环上界，隐式实现因果 mask）
//  q: [B, n_heads, head_dim]，K/V 已按 positions[b] 写入缓存
Tensor AttentionForwardBatch(const MiniLlamaModel& model, const Tensor& q,
    const std::vector<int>& positions, int layer, KVCache& kv_cache,
    int n_heads, int n_kv_heads, int head_dim){
    const int batch = static_cast<int>(positions.size());
    if(n_kv_heads <= 0 || n_heads % n_kv_heads != 0){
        throw std::runtime_error("n_heads must be divisible by n_kv_heads");
    }
    const int group = n_heads / n_kv_heads;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    Tensor attention_out({batch, n_heads, head_dim}, 0.0f);
    for(int b = 0;b < batch;b++){
        const int pos = positions[b];
        const float* q_base = q.data.data() + static_cast<size_t>(b) * n_heads * head_dim;
        for(int h = 0;h < n_heads;h++){
            const int kv_head = h / group;
            const float* q_head = q_base + static_cast<size_t>(h) * head_dim;
            //QK^T: 只累加 i <= pos，等价于因果 mask（看不见 pos 之后的 key）
            std::vector<float> score(pos + 1, 0.0f);
            for(int i = 0;i <= pos;i++){
                const float* k_ptr = kv_cache.KeyPtr(layer, i, kv_head);
                float dot = 0.0f;
                for(int d = 0;d < head_dim;d++){
                    dot += q_head[d] * k_ptr[d];
                }
                score[i] = dot * scale;
            }
            Tensor x({pos + 1}, 0.0f);
            x.data = std::move(score);
            Tensor prob = ForwardSoftmax(model, x);
            //V 加权
            float* out_head = attention_out.data.data()
                + (static_cast<size_t>(b) * n_heads + h) * head_dim;
            for(int d = 0;d < head_dim;d++){
                float val = 0.0f;
                for(int i = 0;i <= pos;i++){
                    const float* v_ptr = kv_cache.ValuePtr(layer, i, kv_head);
                    val += prob.data[i] * v_ptr[d];
                }
                out_head[d] = val;
            }
        }
    }
    return attention_out;
}

//批量单层前向: x [B, dim] -> [B, dim]
Tensor ForwardLayerBatch(Context& ctx, const MiniLlamaModel& model, const Tensor& embedding_vec,
    const std::vector<int>& positions, int layer, const LayerWeights& weights, const ModelConfig& config){
    const int dim = config.dim;
    const int n_heads = config.n_heads;
    const int n_kv_heads = config.n_kv_heads;
    const int head_dim = config.head_dim;
    const int kv_dim = n_kv_heads * head_dim;
    const int batch = static_cast<int>(positions.size());
    const float eps = config.rms_norm_eps;
    const float theta = config.rope_theta;
    const RoPEtype rope_type = config.rope_type;
    const std::string layer_prefix = "layer." + std::to_string(layer);

    //注意力前 RMSNorm（逐行）
    Tensor h = RMSNormRows(embedding_vec, weights.attention_norm, eps);

    //QKV 投影: [B, dim] -> [B, n_heads*head_dim] / [B, kv_dim]
    Tensor q = AddOptionalBiasRows(Linear(h, weights.wq), weights.bq, layer_prefix + " wq");
    Tensor k = AddOptionalBiasRows(Linear(h, weights.wk), weights.bk, layer_prefix + " wk");
    Tensor v = AddOptionalBiasRows(Linear(h, weights.wv), weights.bv, layer_prefix + " wv");
    RequireShape(q, {batch, n_heads * head_dim}, layer_prefix + ": q");
    RequireShape(k, {batch, kv_dim}, layer_prefix + ": k");
    RequireShape(v, {batch, kv_dim}, layer_prefix + ": v");

    //分头
    q.ReshapeCheck({batch, n_heads, head_dim});
    k.ReshapeCheck({batch, n_kv_heads, head_dim});
    v.ReshapeCheck({batch, n_kv_heads, head_dim});

    //逐 token 旋转位置编码（每个 token 用自己的 position）
    rope_batch(q, k, positions, theta, rope_type);

    //先把整个 batch 的 K/V 写入缓存，再逐 query 计算注意力
    for(int b = 0;b < batch;b++){
        const float* k_row = k.data.data() + static_cast<size_t>(b) * kv_dim;
        const float* v_row = v.data.data() + static_cast<size_t>(b) * kv_dim;
        ctx.kv_cache.WriteRow(layer, positions[b], k_row, v_row);
    }

    //因果注意力
    Tensor att_out = AttentionForwardBatch(model, q, positions, layer,
        ctx.kv_cache, n_heads, n_kv_heads, head_dim);
    att_out.ReshapeCheck({batch, dim});

    //输出投影 + 残差
    Tensor att_out_proj = Linear(att_out, weights.wo);
    RequireShape(att_out_proj, {batch, dim}, layer_prefix + ": att_out_proj");
    Tensor att_out_cnn = ForwardAdd(model, att_out_proj, embedding_vec);

    //FFN: 逐行 RMSNorm -> SwiGLU -> down -> 残差
    Tensor h_norm = RMSNormRows(att_out_cnn, weights.ffn_norm, eps);
    Tensor gate = Linear(h_norm, weights.w_gate);
    Tensor up = Linear(h_norm, weights.w_up);
    Tensor swiglu = SWiGLU(gate, up);
    Tensor down = Linear(swiglu, weights.w_down);
    RequireShape(down, {batch, dim}, layer_prefix + ": ffn_down");

    return ForwardAdd(model, down, att_out_cnn);
}

void ForwardRopeDevice(const MiniLlamaModel& model, CudaTensor& q, CudaTensor& k,
                       int n_heads, int n_kv_heads, int head_dim, int pos,
                       float theta, RoPEtype rope_type){
    CudaRopeDeviceInput(q, k, n_heads, n_kv_heads, head_dim, pos, theta,
                        rope_type, CudaDeviceId(model));
    RecordCudaActivationCall(model);
}

void ForwardRopeBatchDevice(const MiniLlamaModel& model, CudaTensor& q,
                            CudaTensor& k, int n_heads, int n_kv_heads,
                            int head_dim, int start_pos, float theta,
                            RoPEtype rope_type){
    CudaRopeBatchDeviceInput(q, k, n_heads, n_kv_heads, head_dim, start_pos,
                             theta, rope_type, CudaDeviceId(model));
    RecordCudaActivationCall(model);
}

CudaTensor AttentionForwardBatchDevice(
    const MiniLlamaModel& model, CudaKvCache& cuda_kv_cache,
    const CudaTensor& q, const CudaTensor& k, const CudaTensor& v,
    int start_pos, int rows, int layer, int n_heads, int n_kv_heads,
    int head_dim){
    cuda_kv_cache.WriteDeviceBatch(layer, start_pos, k, v);
    RecordCudaKvCacheWrite(model, (k.size() + v.size()) * sizeof(float));

    CudaTensor attn_out = CudaAttentionBatchDeviceInput(
        q, cuda_kv_cache, layer, start_pos, n_heads, n_kv_heads, head_dim,
        CudaDeviceId(model));
    RecordCudaAttentionCall(model);

    size_t visible_positions = 0;
    for(int row = 0;row < rows;row++){
        visible_positions += static_cast<size_t>(start_pos + row + 1);
    }
    RecordCudaKvCacheRead(model, visible_positions *
                                     static_cast<size_t>(n_heads) *
                                     static_cast<size_t>(head_dim) *
                                     sizeof(float) * 2);
    return attn_out;
}

Tensor ComputeLogitsDevice(const CudaTensor& x, const MiniLlamaModel& model,
                           const ModelConfig& c){
    CudaTensor normed = ForwardRmsNormDevice(model, "final_norm", x,
                                             model.final_norm, c.rms_norm_eps);
    CudaTensor logits_dev =
        ForwardLinearDevice(model, "lm_head", normed, model.lm_head);
    Tensor logits = DownloadCudaTensor(model, logits_dev);
    logits.ReshapeCheck({c.vocab_size});
    return logits;
}

CudaTensor ForwardLayerDevice(Context& ctx, const MiniLlamaModel& model,
                              const CudaTensor& x, int layer,
                              const LayerWeights& lw, const ModelConfig& c){
    const int dim = c.dim;
    const int n_heads = c.n_heads;
    const int n_kv_heads = c.n_kv_heads;
    const int head_dim = c.head_dim;
    const int pos = ctx.pos;
    const std::string layer_prefix = "layers." + std::to_string(layer) + ".";

    CudaTensor h = ForwardRmsNormDevice(
        model, layer_prefix + "attention_norm", x, lw.attention_norm,
        c.rms_norm_eps);

    CudaTensor q;
    CudaTensor k;
    CudaTensor v;
    ForwardQkvProjectionDevice(model, layer_prefix, h, lw, q, k, v);
    ForwardRopeDevice(model, q, k, n_heads, n_kv_heads, head_dim, pos,
                      c.rope_theta, c.rope_type);

    CudaTensor attn_out =
        AttentionForwardDevice(model, q, k, v, pos, layer, ctx.cuda_kv_cache,
                               n_heads, n_kv_heads, head_dim);

    CudaTensor attn_proj =
        ForwardLinearDevice(model, layer_prefix + "wo", attn_out, lw.wo);
    if(attn_proj.size() != static_cast<size_t>(dim)){
        throw std::runtime_error(
            "forward_layer_device: attention projection shape mismatch");
    }
    CudaTensor x_attn = ForwardAddDevice(model, x, attn_proj);

    CudaTensor h2 = ForwardRmsNormDevice(
        model, layer_prefix + "ffn_norm", x_attn, lw.ffn_norm, c.rms_norm_eps);
    CudaTensor ff = FfnForwardDevice(model, layer_prefix, h2, lw);
    if(ff.size() != static_cast<size_t>(dim)){
        throw std::runtime_error("forward_layer_device: FFN output shape mismatch");
    }
    return ForwardAddDevice(model, x_attn, ff);
}

bool BatchPositionsAreContiguous(const Batch& batch, const ModelConfig& c,
                                 int& start_pos){
    if(batch.tokens.size() != batch.positions.size() ||
       batch.positions.empty()){
        return false;
    }
    start_pos = batch.positions.front();
    if(start_pos < 0 || batch.num_tokens() > c.max_seq_len - start_pos){
        return false;
    }
    for(int i = 0;i < batch.num_tokens();i++){
        if(batch.positions[i] != start_pos + i){
            return false;
        }
        if(batch.tokens[i] < 0 || batch.tokens[i] >= c.vocab_size){
            return false;
        }
    }
    return true;
}

CudaTensor ForwardLayerBatchDevice(const MiniLlamaModel& model,
                                   CudaKvCache& cuda_kv_cache,
                                   const CudaTensor& x, int layer,
                                   const LayerWeights& lw, const ModelConfig& c,
                                   int start_pos, int rows){
    const int dim = c.dim;
    const int n_heads = c.n_heads;
    const int n_kv_heads = c.n_kv_heads;
    const int head_dim = c.head_dim;
    const std::string layer_prefix = "layers." + std::to_string(layer) + ".";

    CudaTensor h = ForwardRmsNormBatchDevice(
        model, layer_prefix + "attention_norm", x, lw.attention_norm,
        c.rms_norm_eps);

    CudaTensor q;
    CudaTensor k;
    CudaTensor v;
    ForwardQkvProjectionBatchDevice(model, layer_prefix, h, lw, q, k, v);
    if(q.size() != static_cast<size_t>(rows) * static_cast<size_t>(n_heads) *
                       static_cast<size_t>(head_dim)){
        throw std::runtime_error("forward_layer_batch_device: q shape mismatch");
    }
    if(k.size() != static_cast<size_t>(rows) *
                       static_cast<size_t>(n_kv_heads) *
                       static_cast<size_t>(head_dim) ||
       v.size() != k.size()){
        throw std::runtime_error("forward_layer_batch_device: k/v shape mismatch");
    }

    ForwardRopeBatchDevice(model, q, k, n_heads, n_kv_heads, head_dim,
                           start_pos, c.rope_theta, c.rope_type);

    CudaTensor attn_out = AttentionForwardBatchDevice(
        model, cuda_kv_cache, q, k, v, start_pos, rows, layer, n_heads,
        n_kv_heads, head_dim);
    CudaTensor attn_proj =
        ForwardLinearDevice(model, layer_prefix + "wo", attn_out, lw.wo);
    if(attn_proj.size() !=
       static_cast<size_t>(rows) * static_cast<size_t>(dim)){
        throw std::runtime_error(
            "forward_layer_batch_device: attention projection shape mismatch");
    }
    CudaTensor x_attn = ForwardAddDevice(model, x, attn_proj);

    CudaTensor h2 = ForwardRmsNormBatchDevice(
        model, layer_prefix + "ffn_norm", x_attn, lw.ffn_norm, c.rms_norm_eps);
    CudaTensor ff = FfnForwardBatchDevice(model, layer_prefix, h2, lw);
    if(ff.size() != static_cast<size_t>(rows) * static_cast<size_t>(dim)){
        throw std::runtime_error("forward_layer_batch_device: FFN shape mismatch");
    }
    return ForwardAddDevice(model, x_attn, ff);
}


}

bool CudaBatchPrefillSupported(const MiniLlamaModel& model,
                               const Batch& batch){
    if(!model.loaded || !model.cuda_weights ||
       model.layers.size() != static_cast<size_t>(model.config.n_layers)){
        return false;
    }
    int start_pos = 0;
    if(!BatchPositionsAreContiguous(batch, model.config, start_pos)){
        return false;
    }
    if(!HasUploadedWeight(model, "token_embedding", QuantType::kF32, false) ||
       !HasUploadedWeight(model, "final_norm", QuantType::kF32, false) ||
       !CudaLinearReady(model, "lm_head", model.lm_head)){
        return false;
    }
    for(int layer = 0;layer < model.config.n_layers;layer++){
        const LayerWeights& lw = model.layers[layer];
        const std::string prefix = "layers." + std::to_string(layer) + ".";
        if(!HasUploadedWeight(model, prefix + "attention_norm", QuantType::kF32,
                              false) ||
           !HasUploadedWeight(model, prefix + "ffn_norm", QuantType::kF32,
                              false) ||
           !CudaLinearReady(model, prefix + "wq", lw.wq) ||
           !CudaLinearReady(model, prefix + "wk", lw.wk) ||
           !CudaLinearReady(model, prefix + "wv", lw.wv) ||
           !CudaLinearReady(model, prefix + "wo", lw.wo) ||
           !CudaLinearReady(model, prefix + "w_gate", lw.w_gate) ||
           !CudaLinearReady(model, prefix + "w_up", lw.w_up) ||
           !CudaLinearReady(model, prefix + "w_down", lw.w_down)){
            return false;
        }
        if((!lw.bq.data.empty() &&
            !HasUploadedWeight(model, prefix + "bq", QuantType::kF32, false)) ||
           (!lw.bk.data.empty() &&
            !HasUploadedWeight(model, prefix + "bk", QuantType::kF32, false)) ||
           (!lw.bv.data.empty() &&
            !HasUploadedWeight(model, prefix + "bv", QuantType::kF32, false))){
            return false;
        }
    }
    return true;
}

//token ID -> logits
Tensor ForwardToken(Context& ctx, const MiniLlamaModel& model, int token){
    ValidateForwardInputs(ctx, model, token);
    if(model.cuda_weights){
        Batch single = Batch::Single(token, ctx.pos);
        if(!CudaBatchPrefillSupported(model, single)){
            throw std::runtime_error(
                "CUDA forward requires all model weights uploaded to the device");
        }
        return ForwardPrefill(ctx, model, single);
    }

    int layers = model.config.n_layers;
    //Embedding
    Tensor embedding_vec = EmbedToken(model, token);
    //n layers transformer block
    Tensor x = embedding_vec;
    for(int i = 0;i < layers;i++){
        x = ForwardLayer(ctx, model, x, i, model.layers[i], model.config);
    }
    return ComputeLogits(model, x);
}
//batch -> logits
Tensor ForwardBatch(Context& ctx, const MiniLlamaModel& model, const Batch& batch){
    if(batch.num_tokens() == 0){
        throw std::runtime_error("ForwardBatch error: batch is empty");
    }

    if(batch.num_tokens() != static_cast<int>(batch.positions.size())){
        throw std::runtime_error("ForwardBatch error: the quantity of tokens and positions is not equal");
    }

    ValidateBatchInputs(model, batch);

    //CUDA 模型：整批 token 在 GPU 上完成前向与 KV 缓存更新
    if(model.cuda_weights){
        if(!CudaBatchPrefillSupported(model, batch)){
            throw std::runtime_error(
                "CUDA forward requires contiguous positions and all model "
                "weights uploaded to the device");
        }
        Tensor logits = ForwardPrefill(ctx, model, batch);
        ctx.token_history.insert(ctx.token_history.end(), batch.tokens.begin(),
                                 batch.tokens.end());
        return logits;
    }

    const int batch_size = batch.num_tokens();
    const int dim = model.config.dim;

    //批量 embedding: [B, dim]
    Tensor x = EmbedBatch(model, batch.tokens);
    //n layers transformer block（整个 batch 一起前向）
    for(int layer = 0;layer < model.config.n_layers;layer++){
        x = ForwardLayerBatch(ctx, model, x, batch.positions, layer,
                              model.layers[layer], model.config);
    }

    //契约: 只返回最后一个 token 的 logits
    Tensor last({dim}, 0.0f);
    memcpy(last.data.data(),
           x.data.data() + static_cast<size_t>(batch_size - 1) * dim,
           static_cast<size_t>(dim) * sizeof(float));
    Tensor logits = ComputeLogits(model, last);

    //推进上下文状态
    ctx.token_history.insert(ctx.token_history.end(), batch.tokens.begin(), batch.tokens.end());
    ctx.pos = batch.positions.back();

    return logits;
}

Tensor ForwardPrefill(Context& ctx, const MiniLlamaModel& model,
                      const Batch& batch){
    if(batch.num_tokens() == 0){
        throw std::runtime_error("ForwardPrefill: empty batch");
    }
    if(batch.tokens.size() != batch.positions.size()){
        throw std::runtime_error(
            "ForwardPrefill: tokens and positions size mismatch");
    }
    if(!model.loaded){
        throw std::runtime_error("ForwardPrefill called with an unloaded model");
    }
    if(!CudaBatchPrefillSupported(model, batch)){
        throw std::runtime_error(
            "ForwardPrefill requires CUDA-resident model weights and contiguous "
            "positions");
    }

    const ModelConfig& c = model.config;
    const int rows = batch.num_tokens();
    const int start_pos = batch.positions.front();
    const int device_id = CudaDeviceId(model);
    if(ctx.cuda_kv_cache.empty()){
        ctx.cuda_kv_cache.Reset(c.n_layers, c.max_seq_len, c.n_kv_heads,
                                c.head_dim, device_id);
    }

    CudaTensor x = EmbedTokensDeviceBatch(model, batch.tokens);
    for(int layer = 0;layer < c.n_layers;layer++){
        x = ForwardLayerBatchDevice(model, ctx.cuda_kv_cache, x, layer,
                                    model.layers[layer], c, start_pos, rows);
    }

    CudaTensor normed = ForwardRmsNormBatchDevice(
        model, "final_norm", x, model.final_norm, c.rms_norm_eps);
    CudaTensor logits_dev =
        ForwardLinearDevice(model, "lm_head", normed, model.lm_head);
    Tensor all_logits = DownloadCudaTensor(model, logits_dev);
    if(all_logits.shape != std::vector<int>{rows, c.vocab_size}){
        throw std::runtime_error("ForwardPrefill: logits shape mismatch");
    }

    Tensor last_logits({c.vocab_size}, 0.0f);
    const size_t vocab = static_cast<size_t>(c.vocab_size);
    const float* src = all_logits.data.data() +
                       static_cast<size_t>(rows - 1) * vocab;
    for(size_t i = 0;i < vocab;i++){
        last_logits.data[i] = src[i];
    }

    ctx.pos = batch.positions.back();
    return last_logits;
}


}
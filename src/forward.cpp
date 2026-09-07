#include "../include/forward.h"
#include "../include/ops.h"
#include <cmath>

namespace LLaMA_Tensor{
namespace{
//输入检查
void ValidateForwardInputs(const Context& ctx, const MiniLlamaModel& model, int token){
    if(!model.loaded){
        throw std::runtime_error("model is not loaded");
    }
    if(token < 0 || token >= model.config.vocab_size){
        throw std::out_of_range("can not find this token: " + std::to_string(token));
    }
    if(ctx.pos < 0 || ctx.pos >= model.config.max_seq_len){
        throw std::out_of_range("tokens are out of the context size");
    }
    if(model.layers.size() != model.config.n_layers){
        throw std::runtime_error("model's layers are not mathed with config's, need: "
             + std::to_string(model.config.n_layers));
    }
    if(model.token_embedding.shape[0] < model.config.vocab_size){
        throw std::runtime_error("token embedding's size is not enough, it should be >= "
             + std::to_string(model.config.vocab_size));
    }
    if(model.token_embedding.shape[1] != model.config.dim){
        throw std::runtime_error("the token embedding vector dimension is not matched with the config's, it should be: "
             + std::to_string(model.config.dim));
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

Tensor ForwardLinear(const MiniLlamaModel& model, const std::string& name, const Tensor& x, const Tensor& weight){
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
    Tensor ffn_out = ForwardAdd(model, h_out, embedding_vec);

    return ffn_out;
}

Tensor ComputeLogits(const MiniLlamaModel& model, Tensor& x){
    Tensor norm = ForwardRmsNorm(model, x, model.final_norm, model.config.rms_norm_eps);
    Tensor logits = ForwardLinear(model, "lm_head", norm, model.lm_head);
    logits.ReshapeCheck({model.config.vocab_size});
    return logits;
}

}
//token ID -> logits
Tensor ForwardToken(Context& ctx, const MiniLlamaModel& model, int token){
    ValidateForwardInputs(ctx, model, token);
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

    if(batch.num_tokens() != batch.positions.size()){
        throw std::runtime_error("ForwardBatch error: the quantity of tokens and positions is not equal");
    }
    Tensor logits;
    for(size_t i = 0;i < batch.num_tokens();i++){
        ctx.pos = batch.positions[i];
        logits = ForwardToken(ctx, model, batch.tokens[i]);
        ctx.token_history.push_back(batch.tokens[i]);
    }
    return logits;
}

}
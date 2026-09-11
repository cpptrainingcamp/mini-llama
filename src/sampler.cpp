#include "../include/sampler.h"
#include "../include/ops.h"
#include <algorithm>

namespace LLaMA_Tensor{
namespace{
constexpr float kGreedyTemperature = 1e-6f;
//logits不能为空或存在无穷
void ValidateLogits(const Tensor& logits){
    if(logits.TensorNum() == 0){
        throw std::runtime_error("can not sample for empty logits");
    }
    for(size_t i = 0;i < logits.TensorNum();i++){
        if(!std::isfinite(logits.At1(i))){
            throw std::runtime_error("infinite value is in the logits: logits[" + std::to_string(i) + "], it can not be sampled");
        }
    }
}

void ValidateTemperature(float temperature){
    if(!std::isfinite(temperature) || temperature < 0.0f){
        throw std::runtime_error("temperature must be finite and non-negative");
    }
}

void ValidateTopK(int top_k){
    if(top_k < 0){
        throw std::runtime_error("top_k must be non-negetive");
    }
}
//随机数采样，若temperature极小，则退化为贪婪采样
int SampleFromCandidates(const std::vector<std::pair<float, int>>& candidates, float temperature, std::mt19937& rng){
    if(temperature < kGreedyTemperature){
        int best = candidates[0].second;
        float best_val = candidates[0].first;
        for(size_t i = 0;i < candidates.size();i++){
            if(candidates[i].first > best_val){
                best_val = candidates[i].first;
                best = candidates[i].second;
            }
        }
        return best;
    }

    float max_val = candidates[0].first;
    for(size_t i = 0;i < candidates.size();i++){
        if(candidates[i].first > max_val){
            max_val = candidates[i].first;
        }
    }

    std::vector<float> probs(candidates.size(), 0.0f);
    float sum = 0.0f;
    for(size_t i = 0;i < probs.size();i++){
        probs[i] = std::expf((candidates[i].first - max_val) / temperature);
        sum += probs[i];
    }
    for(float& num : probs){
        num /= sum;
    }

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cumulative = 0.0f;
    for(size_t i = 0;i < probs.size();i++){
        cumulative += probs[i];
        if(r <= cumulative){
            return candidates[i].second;
        }
    }
    return candidates.back().second;
}

}

Sampler::Sampler(unsigned int seed){
    if(seed != 0){
        rng_.seed(seed);
    }else{
        std::random_device rd;
        rng_.seed(rd());
    }
}

Sampler::Sampler(const SamplerParams& param)
    :Sampler(param.seed){
    ValidateTemperature(param.temperature);
    ValidateTopK(param.top_k);
    param_ = std::move(param);
}

int Sampler::SampleGreedy(const Tensor& logits){
    ValidateLogits(logits);
    return ArgsMax(logits);
}

int Sampler::SampleTemperature(const Tensor& logits){
    ValidateLogits(logits);
    std::vector<std::pair<float, int>> candidate;
    candidate.reserve(logits.TensorNum());
    for(int i = 0;i < logits.TensorNum();i++){
        candidate.push_back({logits.At1(i), i});
    }
    return SampleFromCandidates(candidate, param_.temperature, rng_);
}

int Sampler::SampleTopK(const Tensor& logits){
    ValidateLogits(logits);
    int topk = param_.top_k;
    if(topk == 0){
        throw std::runtime_error("SampleTopK: topk must be positive");
    }
    if(topk == 1){
        return Sampler::SampleGreedy(logits);
    }
    if(topk > logits.TensorNum()){
        topk = logits.TensorNum();
    }

    std::vector<std::pair<float, int>> candidate;
    candidate.reserve(logits.TensorNum());
    for(int i = 0;i < logits.TensorNum();i++){
        candidate.emplace_back(logits.At1(i), i);
    }
    auto cmp = [](const std::pair<float, int>& a, const std::pair<float, int>& b){
        if(a.first == b.first){
            return a.second < b.second;
        }
        return a.first > b.first;
    };
    std::sort(candidate.begin(), candidate.end(), cmp);
    candidate.resize(topk);
    return SampleFromCandidates(candidate, param_.temperature, rng_);
}

int Sampler::Sample(const Tensor& logits){
    if(param_.temperature < kGreedyTemperature || param_.top_k == 1){
        return Sampler::SampleGreedy(logits);
    }
    if(param_.top_k > 1){
        return SampleTopK(logits);
    }
    return SampleTemperature(logits);
}
}

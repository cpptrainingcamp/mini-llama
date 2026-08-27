#pragma once

#include <unordered_map>
#include <map>
#include <memory>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <simdjson.h>

namespace LLaMA_Tensor{
    using namespace simdjson;

    class Tokenizer{
    public:
        virtual ~Tokenizer() = default;
        //将text编码为id
        virtual std::vector<int> Encode(const std::string& text) const = 0;
        //将id解码为单个token
        virtual std::string DecodeToken(int token) const = 0;
        //将一系列id解码为一系列token
        virtual std::string Decode(const std::vector<int>& tokens) const = 0;
        virtual int vocab_size() const = 0;
        virtual int bos_id() const = 0;
        virtual int eos_id() const = 0;
        virtual int unk_id() const = 0;
    };

    class AsciiTokenizer : public Tokenizer{
    public:
        static constexpr int VocabSize = 128;
        AsciiTokenizer() = default;
        std::vector<int> Encode(const std::string& text) const override;
        std::string DecodeToken(int token_id) const override;
        std::string Decode(const std::vector<int>& tokens_id) const override;
        int vocab_size() const override{
            return VocabSize;
        }
        int bos_id() const override {
            return 1;
        }
        int eos_id() const override{
            return 2;
        }
        int unk_id() const override{
            return 0;
        }
    };

    class JosnVocabTokenizer : public Tokenizer{
    public:
        explicit JosnVocabTokenizer(const std::string& path);
        ~JosnVocabTokenizer() override = default;
        std::vector<int> Encode(const std::string& text) const override;
        std::string DecodeToken(int token_id) const override;
        std::string Decode(const std::vector<int>& tokens_id) const override;
        
        int vocab_size() const override{
            return vocab_size_;
        }
        int bos_id() const override {
            return bos_id_;
        }
        int eos_id() const override{
            return eos_id_;
        }
        int unk_id() const override{
            return unk_id_;
        }

    private:
        //词表里单个token的基本信息
        struct vocab_unit{
            int id = 0;
            std::string content;
            bool special = false;
        };
        //词表token数
        int vocab_size_ = 0;
        //特殊token ID
        int bos_id_ = 1;
        int eos_id_ = 2;
        int unk_id_ = 0;
        //解码检索：ID -> content
        std::vector<vocab_unit>id_2_content;
        //编码检索：content -> ID
        std::vector<std::pair<std::string, int>>content_2_id;
    };

    class BpeTokenizer : public Tokenizer{
    public:
        BpeTokenizer() = default;
        ~BpeTokenizer() override = default;  

        bool Load(const std::string& vocab_path, const std::string& merges_path,
             const std::string& special_path);
        
        std::vector<int> Encode(const std::string& text) const override;
        std::string DecodeToken(int token_id) const override;
        std::string Decode(const std::vector<int>& tokens_id) const override;
        
        int vocab_size() const override{
            return static_cast<int>(id_2_token_.size());
        }
        int bos_id() const override {
            return bos_id_;
        }
        int eos_id() const override{
            return eos_id_;
        }
        int unk_id() const override{
            return unk_id_;
        }

    private:
        //编码检索词表:content -> id
        std::unordered_map<std::string, int>token_2_id_;
        //解码检索:id -> content
        std::vector<std::string>id_2_token_;
        //token合并优先级映射，越小优先级越高
        std::map<std::pair<std::string, std::string>, int>merge_ranks_;

        int bos_id_ = -1;
        int eos_id_ = -1;
        int unk_id_ = -1;

        //特殊字符最长匹配
        std::vector<std::pair<std::string, int>>special_token_;
        //byte -> unicode
        std::vector<std::string>byte_2_unicode_;
        //unicode -> byte
        std::unordered_map<std::string, u_int8_t>unicode_2_byte_;
    };
    //JsonTokenizer工厂函数，如果找不到词表或JsonTokenizer出错就退化为AsciiTokenizer
    std::unique_ptr<Tokenizer> CreateJsonTokenizer(const std::string& vocab_path);
    //BpeTokenizer工厂函数
    std::unique_ptr<Tokenizer> CreateBpeTokenizer(const std::string& vocab_path, const std::string& merges_path,
             const std::string& special_path);

}
#include "../include/tokenizer.h"

// Simple JSON array parser for the vocab format.
// Only supports: [{"id": n, "content": "...", "special": true/false}, ...]

namespace LLaMA_Tensor{
    

    JosnVocabTokenizer::JosnVocabTokenizer(const std::string& path){
        ondemand::parser parser;
        padded_string json_data;
        auto error = padded_string::load(path).get(json_data);
        if(error){
            throw std::runtime_error("failed to load file: " + path);
        }
        auto doc = parser.iterate(json_data);
        if(doc.error()){
            throw std::runtime_error("json parse error");
        }
        for(auto obj : doc){
            int id;
            std::string_view content_v;
            bool special;
            auto err0 = obj["id"].get(id);
            auto err1 = obj["content"].get(content_v);
            auto err2 = obj["special"].get(special);
            if(err0 || err1 || err2){
                std::cerr << "get token error" << std::endl;
                continue;
            }
            std::string content = std::string(content_v);
            id_2_content.push_back({id, content, special});
            content_2_id.push_back({content, id});
            if(content == "<bos>"){
                bos_id_ = id;
            }else if(content == "<eos>"){
                eos_id_ = id;
            }else if(content == "<unk>"){
                unk_id_ = id;
            }
            vocab_size_ = (id + 1) > vocab_size_ ? (id + 1) : vocab_size_;
        }
        std::sort(content_2_id.begin(), content_2_id.end(), 
        [](const auto& a, const auto& b){
            return a.first.size() > b.first.size();
        });
    }

    std::vector<int> JosnVocabTokenizer::Encode(const std::string& text) const {
        std::vector<int> tokens_id;
        tokens_id.push_back(bos_id());
        int pos = 0;
        while(pos < text.size()){
            bool match = false;
            for(const auto& token : content_2_id){
                if(token.first.empty()) continue;
                std::string_view content = token.first;
                if(text.compare(pos, content.size(), content) == 0){
                    tokens_id.push_back(token.second);
                    match = true;
                    pos += content.empty() ? 1 : content.size();
                    break;
                }
            }
            if(!match){
                tokens_id.push_back(unk_id());
                pos++;
            }
        }
        return tokens_id;
    }

    std::string JosnVocabTokenizer::DecodeToken(int token_id) const {
        auto it = std::find_if(id_2_content.begin(), id_2_content.end(), [token_id](const auto& x){
            return x.id == token_id;
        });
        if(it == id_2_content.end()){
            return "<unk>";
        }else{
            return it->content;
        }
    }

    std::string JosnVocabTokenizer::Decode(const std::vector<int>& tokens_id) const {
        std::string result;
        for(const auto& num : tokens_id){
            result += DecodeToken(num);
        }
        return result;
    }

    std::unique_ptr<Tokenizer> CreateJsonTokenizer(const std::string& vocab_path){
        if(!vocab_path.empty()){
            std::ifstream f(vocab_path);
            if(f.good()){
                try{
                    return std::make_unique<JosnVocabTokenizer>(vocab_path);
                }catch (const std::exception& e){
                    std::cerr << "Someting wrong happened with JsonVocabTokenizer: " 
                    << e.what() << ",  fall back to AsciiTokenizer" << std::endl;
                }
            }
        }
        return std::make_unique<AsciiTokenizer>();
    }

}
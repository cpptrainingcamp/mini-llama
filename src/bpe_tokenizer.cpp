#include "../include/tokenizer.h"


namespace LLaMA_Tensor{
    
namespace{
    bool CheckHead(const std::string& str, const std::string& head){
        if(head.size() > str.size()) return false;
        int pos = 0;
        while(pos < head.size()){
            if(head[pos] != str[pos]){
                return false;
            }else{
                pos++;
            }
        }
        if(pos == head.size()) return true;
    }
    bool CheckTail(const std::string& str, const std::string& tail){
        if(tail.size() > str.size()) return false;
        int pos = 0;
        while(pos < tail.size()){
            if(tail[tail.size() - pos - 1] != str[str.size() - pos - 1]) return false;
            else pos++;
        }
        if(pos == tail.size()) return true;
    }

    std::string CodepointToUtf8(char32_t cp){
        std::string result;
        if(cp <= 0x7F){
            result += static_cast<char>(cp);
        }else if(cp <= 0x7FF){
            result += static_cast<char>(0xC0 | cp >> 6);
            result += static_cast<char>(0x80 | cp & 0x3F);
        }else if(cp <= 0xFFFF){
            result += static_cast<char>(0xE0 | cp >> 12);
            result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            result += static_cast<char>(0x80 | (cp & 0x3F));
        }else{
            result += static_cast<char>(0xF0 | (cp >> 18));
            result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            result += static_cast<char>(0x80 | (cp & 0x3F));
        }
        return result;
    }

    std::string NextUtf8Codepoint(const std::string& s, size_t& pos){
        if (pos >= s.size()) {
            return "";
        }
        unsigned char first = s[pos];
        size_t len = 1;
        if ((first & 0x80) == 0) {
            len = 1;
        } else if ((first & 0xE0) == 0xC0) {
            len = 2;
        } else if ((first & 0xF0) == 0xE0) {
            len = 3;
        } else if ((first & 0xF8) == 0xF0) {
            len = 4;
        } else {
            // Invalid UTF-8, skip one byte
            ++pos;
            return "";
        }
        if (pos + len > s.size()) {
            ++pos;
            return "";
        }
        std::string result = s.substr(pos, len);
        pos += len;
        return result;
    }

    std::vector<std::string> BuildBytesToUnicode(){
        std::vector<std::string> map(256);

        std::vector<int> bs;
        for (int c = '!'; c <= '~'; ++c) {
            bs.push_back(c);
        }
        for (int c = 0xA1; c <= 0xAC; ++c) {
            bs.push_back(c);
        }
        for (int c = 0xAE; c <= 0xFF; ++c) {
            bs.push_back(c);
        }
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) != bs.end()) {
                map[b] = CodepointToUtf8(b);
            }else{
                map[b] = CodepointToUtf8(256 + n++);
            }
        }
        return map;
    }

}

    bool BpeTokenizer::Load(const std::string& vocab_path, const std::string& merges_path,const std::string& special_path){
        //load vocab.json
        {
            ondemand::parser parser;
            padded_string json_data;
            auto error = padded_string::load(vocab_path).get(json_data);
            if(error){
                std::cerr << "failed to load " << vocab_path << std::endl;
                return false;
            }
            auto doc = parser.iterate(json_data);
            if(doc.error()){
                std::cerr << "json parse error" << std::endl;
                return false;
            }
            int max_id = 0;
            for(auto field : doc.get_object()){
                std::string_view token_view;
                auto key_err = field.unescaped_key().get(token_view);
                if (key_err) {
                    std::cerr << "Error getting key: " << key_err << std::endl;
                    continue;
                }
                int64_t id_value;
                auto val_err = field.value().get(id_value);
                if (val_err) {
                    std::cerr << "Error getting value for token '" << token_view << "': " << val_err << std::endl;
                    continue;
                }
                int id = static_cast<int>(id_value);
                max_id = id > max_id ? id : max_id;
                id_2_token_.resize(max_id + 1);
                id_2_token_[id] = token_view;
                std::string content(token_view);
                token_2_id_[content] = id;
                if(CheckHead(content, "<|") && CheckTail(content, "|>")){
                    special_token_.push_back({content, id});
                }
            }
            std::sort(special_token_.begin(), special_token_.end(), 
                [](const auto& a, const auto& b){
                    return a.first.size() > b.first.size();
                });
        }
        //load merges.txt
        {
            std::ifstream f(merges_path);
            if(!f.is_open()){
                std::cerr << "failed to open " << merges_path << std::endl;
                return false;
            }
            std::string line;
            int ranks = 0;
            while(std::getline(f, line)){
                std::stringstream ss(line);
                std::string a, b;
                if(ss>>a>>b){
                    merge_ranks_[{a, b}] = ranks++;
                }
            }
        }
        //load special.json
        {
            ondemand::parser parser;
            padded_string json_data;
            auto error = padded_string::load(special_path).get(json_data);
            if(error){
                std::cerr << "failed to load " << special_path << std::endl;
                return false;
            }
            auto doc = parser.iterate(json_data);
            if(doc.error()){
                std::cerr << "json parse failed" << std::endl;
                return false;
            }
            auto obj = doc.get_object();
            auto erro0 = obj["bos_id"].get(bos_id_);
            auto erro1 = obj["eos_id"].get(eos_id_);
            auto erro2 = obj["unk_id"].get(unk_id_);
            if(erro0){
                std::cerr << "failed to find bos_id" << std::endl;
                return false;
            }
            if(erro1){
                std::cerr << "failed to find eos_id" << std::endl;
                return false;
            }
            if(erro2){
                std::cerr << "failed to find unk_id" << std::endl;
                return false;
            }
        }

        //byte - unicode
        {
            byte_2_unicode_ = BuildBytesToUnicode();
            unicode_2_byte_.clear();
            for(int i = 0;i < byte_2_unicode_.size();i++){
                unicode_2_byte_[byte_2_unicode_[i]] = static_cast<u_int8_t>(i);
            }
        }

        std::cout << "BPE tokenizer loaded: vocab=" << token_2_id_.size()
            << " merges=" << merge_ranks_.size() << " bos=" << bos_id_
            << " eos=" << eos_id_ << " unk=" << unk_id_ << std::endl;
        return true;
    }

    std::vector<int> BpeTokenizer::Encode(const std::string& text) const{
        if(text.empty()) return {};
        std::vector<int> result;
        int pos = 0;
        //最长匹配特殊token
        for(const auto& x : special_token_){
            if(text.compare(0, x.first.size(), x.first) == 0){
                result.push_back(x.second);
                pos += x.first.size();
                break;
            }
        }
        while(pos < text.size()){
            int start = pos;
            int end;
            std::pair<std::string, int> another_sp;
            bool find_another = false;
            //找后一个特殊token或末尾
            for(;pos < text.size();pos++){
                if(CheckHead(text.substr(pos, 2), "<|")){
                    for(const auto& x : special_token_){
                        if(text.compare(pos, x.first.size(), x.first) == 0){
                            end = pos;
                            pos += x.first.size();
                            another_sp = x;
                            find_another = true;
                            break;
                        }
                    }
                }
                if(find_another) break;
            }

            if(!find_another){
                end = text.size();
                pos = end + 1;
            }
            //获取正常字符的unicode
            std::vector<std::string> unicode_word;
            std::string_view seg(text.data() + start, end - start);
            for(unsigned char c : seg){
                if(c >= byte_2_unicode_.size()){
                    std::cerr << "Invalid character" << std::endl;
                    return {};
                }
                unicode_word.push_back(byte_2_unicode_[c]);
            }
            //BPE merge
            while(unicode_word.size() > 1){
                int best_rank = std::numeric_limits<int>::max();
                int merge_index = unicode_word.size();
                for(int i = 0;i + 1 <unicode_word.size();i++){
                    auto it = merge_ranks_.find({unicode_word[i], unicode_word[i+1]});
                    if(it != merge_ranks_.end() && it->second < best_rank){
                        best_rank = it->second;
                        merge_index = i;
                    }
                }
                if(merge_index >= unicode_word.size()){
                    break;
                }
                unicode_word[merge_index] += unicode_word[merge_index + 1];
                unicode_word.erase(unicode_word.begin() + merge_index + 1);
            }
            //token -> id
            for(std::string& word : unicode_word){
                if(token_2_id_.find(word) != token_2_id_.end()){
                    result.push_back(token_2_id_.at(word));
                }else{
                    result.push_back(unk_id());
                }
            }
            if(find_another){
                result.push_back(another_sp.second);
            }
        }
        return result;
    }

    std::string BpeTokenizer::DecodeToken(int token_id) const {
        if(token_id >= 0 && token_id < id_2_token_.size()){
            return id_2_token_[token_id];
        }else{
            return "";
        }
    }

    std::string BpeTokenizer::Decode(const std::vector<int>& tokens_id) const {
        std::string result_unicode;
        for(int id : tokens_id){
            result_unicode += DecodeToken(id);
        }
        std::string result_bytes;
        size_t pos = 0;
        while(pos < result_unicode.size()){
            size_t start = pos;
            std::string cp = NextUtf8Codepoint(result_unicode, pos);
            if (cp.empty()) {
            // Invalid UTF-8, skip
            if (pos == start) {
                ++pos;
            }
            continue;
            }
            auto it = unicode_2_byte_.find(cp);
            if (it != unicode_2_byte_.end()) {
            result_bytes += static_cast<char>(it->second);
            } else {
            // Not a byte token; pass through as-is (shouldn't happen for standard
            // BPE)
            result_bytes += cp;
            }
        }
        return result_bytes;
    }

    std::unique_ptr<Tokenizer> CreateBpeTokenizer(const std::string& vocab_path, const std::string& merges_path,const std::string& special_path){
        auto bpe = std::make_unique<BpeTokenizer>();
        if(bpe->Load(vocab_path, merges_path, special_path)){
            return bpe;
        }else{
            return nullptr;
        }
    }
}
#include "../include/radixtree.h"

namespace LLaMA_Tensor{
void RadixTree::Clear(){
    root_.key.clear();
    root_.children.clear();
    root_.terminal = false;
    terminal_count_ = 0;
}

bool RadixTree::empty() const{
    return terminal_count_ == 0;
}

size_t RadixTree::size() const{
    return terminal_count_;
}

size_t RadixTree::CommonPrefixLength(const std::vector<int>& a, size_t a_offset, const std::vector<int>& b){
    size_t i = 0;
    while(a_offset + i < a.size() && i < b.size() && a[a_offset + i] == b[i]){
        i++;
    }
    return i;
}

void RadixTree::Insert(const std::vector<int>& tokens){
    if(tokens.empty()){
        if(!root_.terminal){
            root_.terminal = true;
            ++terminal_count_;
        }
        return;
    }
    InsertInto(&root_, tokens);
}

void RadixTree::InsertInto(Node* parent, std::vector<int> suffix){
    //suffix全新
    auto child_it = parent->children.find(suffix[0]);
    if(child_it == parent->children.end()){
        auto child = std::make_unique<Node>();
        child->key = std::move(suffix);
        child->terminal = true;
        parent->children[child->key[0]] = std::move(child);
        terminal_count_++;
        return;
    }
    //suffix全部与child前缀匹配
    std::unique_ptr<Node>& child = child_it->second;
    size_t common = CommonPrefixLength(suffix, 0, child->key);
    if(common == child->key.size()){
        //suffix与child前缀等长
        if(common == suffix.size()){
            if(!child->terminal){
                child->terminal = true;
                terminal_count_++;
            }
            return;
        //suffix长于child前缀
        }else{
            std::vector<int> suffix_tail(suffix.begin() + static_cast<long>(common), suffix.end());
            InsertInto(child.get(), suffix_tail);
            return;
        }
    }
    //suffix与child前缀部分匹配
    auto split = std::make_unique<Node>();
    split->key.assign(child->key.begin(), child->key.begin() + static_cast<long>(common));
    child->key.erase(child->key.begin(), child->key.begin() + static_cast<long>(common));
    int old_child_first = child->key[0];
    split->children[old_child_first] = std::move(child);
    //suffix全部匹配child部分前缀
    if(common == suffix.size()){
        split->terminal = true;
        terminal_count_++;
    //suffix与child部分前缀匹配
    }else{
        auto new_child = std::make_unique<Node>();
        new_child->key.assign(suffix.begin() + static_cast<long>(common), suffix.end());
        new_child->terminal = true;
        split->children[new_child->key[0]] = std::move(new_child);
        terminal_count_++;
    }
    child_it->second = std::move(split);
}

size_t RadixTree::LongestPrefix(const std::vector<int>& tokens) const{
    if(tokens.empty()) return 0;

    const Node* node = &root_;
    size_t pos = 0;
    while(pos < tokens.size()){
        auto child_t = node->children.find(tokens[pos]);
        if(child_t == node->children.end()) return pos;

        const Node* child = child_t->second.get();
        size_t common = CommonPrefixLength(tokens, pos, child->key);
        pos += common;
        if(common < child->key.size()) return pos;
        node = child;
    }
    return pos;
}
}

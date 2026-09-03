#pragma once

#include <vector>
#include <memory>
#include <map>

namespace LLaMA_Tensor{
class RadixTree{
public:
    RadixTree() = default;
    ~RadixTree() = default;

    RadixTree(const RadixTree&) = delete;
    RadixTree& operator=(const RadixTree&) = delete;
    RadixTree(RadixTree&&) noexcept = default;
    RadixTree& operator=(RadixTree&&) noexcept = default;

    void Clear();
    bool empty() const;
    size_t size() const;

    void Insert(const std::vector<int>& tokens);
    size_t LongestPrefix(const std::vector<int>& tokens) const;
private:
    struct Node{
        std::vector<int> key;
        bool terminal = false;
        std::map<int, std::unique_ptr<Node>> children;
    };
    static size_t CommonPrefixLength(const std::vector<int>& a, size_t a_offset,
                                   const std::vector<int>& b);
    void InsertInto(Node* parent, std::vector<int> suffix);

    Node root_;
    size_t terminal_count_ = 0;
};
}
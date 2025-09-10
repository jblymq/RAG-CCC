#pragma once
#include "chunk.h"
#include "config.h"
#include "tokenizer.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <shared_mutex>
#include <sstream>
#include <memory>

namespace rag {

class BM25Indexer {
public:
    BM25Indexer(const BM25Config& config = BM25Config{});
    BM25Indexer(double k1 = 1.5, double b = 0.75);  // Keep backward compatibility

    // 设置Tokenizer
    void set_tokenizer(std::shared_ptr<Tokenizer> tokenizer);
    void set_tokenizer_config(const TokenizerConfig& config);

    void fit(const std::vector<Chunk>& chunks);
    std::vector<std::pair<size_t, double>> query(const std::vector<std::string>& terms, size_t topK);

    // 使用文本查询（自动分词）
    std::vector<std::pair<size_t, double>> query_text(const std::string& query_text, size_t topK, Language lang = Language::AUTO);

private:
    double idf(const std::string& term) const;

    // 分词函数
    std::vector<std::string> tokenize(const std::string& text, Language lang = Language::AUTO) const;

    double k1_;
    double b_;
    double avgdl_ = 0.0;
    size_t N_ = 0;
    std::unordered_map<std::string, size_t> df_;
    std::vector<std::unordered_map<std::string, size_t>> tfs_;
    std::shared_mutex mutex_;

    // Tokenizer
    std::shared_ptr<Tokenizer> tokenizer_;
};

} // namespace rag

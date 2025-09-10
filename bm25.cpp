#include "bm25.h"
#include <cmath>
#include <algorithm>

namespace rag {

BM25Indexer::BM25Indexer(const BM25Config& config) : k1_(config.k1), b_(config.b) {
    // 创建默认tokenizer
    TokenizerConfig tokenizer_config;
    tokenizer_ = std::make_shared<Tokenizer>(tokenizer_config);
}

BM25Indexer::BM25Indexer(double k1, double b) : k1_(k1), b_(b) {
    // 创建默认tokenizer
    TokenizerConfig tokenizer_config;
    tokenizer_ = std::make_shared<Tokenizer>(tokenizer_config);
}

void BM25Indexer::set_tokenizer(std::shared_ptr<Tokenizer> tokenizer) {
    tokenizer_ = tokenizer;
}

void BM25Indexer::set_tokenizer_config(const TokenizerConfig& config) {
    tokenizer_ = std::make_shared<Tokenizer>(config);
}

std::vector<std::string> BM25Indexer::tokenize(const std::string& text, Language lang) const {
    if (tokenizer_) {
        return tokenizer_->tokenize(text, lang);
    } else {
        // 回退到简单分词
        std::vector<std::string> tokens;
        std::istringstream iss(text);
        std::string token;
        while (iss >> token) {
            tokens.push_back(token);
        }
        return tokens;
    }
}

void BM25Indexer::fit(const std::vector<Chunk>& chunks) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    N_ = chunks.size();
    tfs_.clear();
    tfs_.reserve(N_);
    df_.clear();
    double total_len = 0.0;

    for (size_t i = 0; i < N_; ++i) {
        const auto &c = chunks[i];
        std::unordered_map<std::string, size_t> tf;

        // 使用新的tokenizer进行分词
        auto tokens = tokenize(c.text);

        for (const auto& token : tokens) {
            ++tf[token];
        }

        for (auto &p : tf) df_[p.first]++;
        tfs_.push_back(std::move(tf));
        total_len += tokens.size();
    }
    avgdl_ = N_ ? (total_len / (double)N_) : 0.0;
}

double BM25Indexer::idf(const std::string& term) const {
    auto it = df_.find(term);
    double df = (it == df_.end()) ? 0.0 : (double)it->second;
    return std::log(1.0 + (N_ - df + 0.5) / (df + 0.5));
}

std::vector<std::pair<size_t, double>> BM25Indexer::query(const std::vector<std::string>& terms, size_t topK) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    std::vector<std::pair<size_t, double>> scores;
    scores.reserve(N_);
    for (size_t i = 0; i < N_; ++i) {
        double score = 0.0;
        double doclen = 0.0;
        for (auto &p : tfs_[i]) doclen += p.second;
        for (const auto &term : terms) {
            double f = 0.0;
            auto it = tfs_[i].find(term);
            if (it != tfs_[i].end()) f = (double)it->second;
            double idf_v = idf(term);
            double denom = f + k1_ * (1.0 - b_ + b_ * (doclen / (avgdl_ > 0 ? avgdl_ : 1.0)));
            if (denom > 0) score += idf_v * (f * (k1_ + 1.0)) / denom;
        }
        scores.emplace_back(i, score);
    }
    std::sort(scores.begin(), scores.end(), [](auto &a, auto &b){return a.second > b.second;});
    if (scores.size() > topK) scores.resize(topK);
    return scores;
}

std::vector<std::pair<size_t, double>> BM25Indexer::query_text(const std::string& query_text, size_t topK, Language lang) {
    // 使用tokenizer对查询文本进行分词
    auto terms = tokenize(query_text, lang);
    return query(terms, topK);
}

} // namespace rag

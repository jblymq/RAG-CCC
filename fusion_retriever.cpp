#include "fusion_retriever.h"
#include <algorithm>
#include <unordered_map>
#include <set>
#include <sstream>
#include <cmath>  // 添加数学函数

// 包含具体的实现头文件
namespace humanus {
    // 简单的mock类，避免依赖复杂的humanus框架
    enum class EmbeddingType { DOCUMENT, QUERY };

    struct VectorStoreConfig {
        int vector_dim = 768;
        int max_elements = 10000;
        int ef_construction = 200;
        int M = 16;
    };

    struct EmbeddingModelConfig {
        std::string provider = "tfidf";
    };

    struct MemoryItem {
        size_t id;
        std::string content;
        std::unordered_map<std::string, std::string> metadata;
        double similarity = 0.0;
    };

    class VectorStore {
    public:
        virtual ~VectorStore() = default;
        virtual void reset() = 0;
        virtual void insert(const std::vector<float>& vector, size_t vector_id, const MemoryItem& metadata) = 0;
        virtual std::vector<MemoryItem> search(const std::vector<float>& query, size_t limit) = 0;
    };

    class EmbeddingModel {
    public:
        virtual ~EmbeddingModel() = default;
        virtual std::vector<float> embed(const std::string& text, EmbeddingType type) = 0;
        static std::shared_ptr<EmbeddingModel> get_instance(const std::string& name, std::shared_ptr<EmbeddingModelConfig> config);
    };

    // 简单的mock实现
    class MockVectorStore : public VectorStore {
    private:
        std::vector<std::pair<std::vector<float>, MemoryItem>> data_;

    public:
        void reset() override { data_.clear(); }

        void insert(const std::vector<float>& vector, size_t vector_id, const MemoryItem& metadata) override {
            data_.emplace_back(vector, metadata);
        }

        std::vector<MemoryItem> search(const std::vector<float>& query, size_t limit) override {
            std::vector<std::pair<double, MemoryItem>> scored_results;

            for (const auto& item : data_) {
                // 简单的余弦相似度计算
                double dot_product = 0.0;
                double norm_query = 0.0;
                double norm_doc = 0.0;

                for (size_t i = 0; i < std::min(query.size(), item.first.size()); ++i) {
                    dot_product += query[i] * item.first[i];
                    norm_query += query[i] * query[i];
                    norm_doc += item.first[i] * item.first[i];
                }

                double similarity = 0.0;
                if (norm_query > 0 && norm_doc > 0) {
                    similarity = dot_product / (std::sqrt(norm_query) * std::sqrt(norm_doc));
                }

                auto memory_item = item.second;
                memory_item.similarity = similarity;
                scored_results.emplace_back(similarity, memory_item);
            }

            // 按相似度排序
            std::sort(scored_results.begin(), scored_results.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });

            std::vector<MemoryItem> results;
            for (size_t i = 0; i < std::min(limit, scored_results.size()); ++i) {
                results.push_back(scored_results[i].second);
            }

            return results;
        }
    };

    class MockEmbeddingModel : public EmbeddingModel {
    public:
        std::vector<float> embed(const std::string& text, EmbeddingType type) override {
            // 简单的TF-IDF式embedding
            std::vector<float> embedding(768, 0.0f);

            // 基于文本hash生成特征向量
            std::hash<std::string> hasher;
            auto hash_val = hasher(text);

            for (size_t i = 0; i < embedding.size(); ++i) {
                embedding[i] = static_cast<float>((hash_val + i) % 1000) / 1000.0f;
            }

            // 归一化
            float norm = 0.0f;
            for (float val : embedding) {
                norm += val * val;
            }
            norm = std::sqrt(norm);

            if (norm > 0) {
                for (float& val : embedding) {
                    val /= norm;
                }
            }

            return embedding;
        }
    };

    std::shared_ptr<EmbeddingModel> EmbeddingModel::get_instance(const std::string& name, std::shared_ptr<EmbeddingModelConfig> config) {
        return std::make_shared<MockEmbeddingModel>();
    }
}

namespace rag {

FusionRetriever::FusionRetriever(const FusionRetrieverConfig& config,
                               std::shared_ptr<humanus::VectorStore> vector_store,
                               std::shared_ptr<humanus::EmbeddingModel> embedding_model)
    : config_(config), vector_store_(vector_store), embedding_model_(embedding_model) {

    // 如果没有提供vector_store，创建默认的Mock实现
    if (!vector_store_) {
        vector_store_ = std::make_shared<humanus::MockVectorStore>();
    }

    // 如果没有提供embedding_model，创建Mock模型
    if (!embedding_model_) {
        auto embed_config = std::make_shared<humanus::EmbeddingModelConfig>();
        embed_config->provider = "tfidf";
        embedding_model_ = humanus::EmbeddingModel::get_instance("tfidf", embed_config);
    }
}

std::shared_ptr<FusionRetriever> FusionRetriever::from_config(const RAGConfig& config) {
    auto fusion_config = FusionRetrieverConfig::from_rag_config(config);

    // 创建mock vector store和embedding model
    auto vector_store = std::make_shared<humanus::MockVectorStore>();
    auto embedding_model = humanus::EmbeddingModel::get_instance("fusion_tfidf", nullptr);

    return std::make_shared<FusionRetriever>(fusion_config, vector_store, embedding_model);
}

void FusionRetriever::fit(const std::vector<Chunk>& chunks) {
    chunks_ = chunks;

    // 构建BM25索引
    bm25_indexer_ = std::make_shared<BM25Indexer>(BM25Config());
    bm25_indexer_->fit(chunks);

    // 构建向量索引
    vector_store_->reset();
    doc_to_vector_id_.clear();

    for (size_t i = 0; i < chunks.size(); ++i) {
        const auto& chunk = chunks[i];

        // 生成embedding
        auto embedding = embedding_model_->embed(chunk.text, humanus::EmbeddingType::DOCUMENT);

        // 插入向量存储
        humanus::MemoryItem memory_item;
        memory_item.id = i;
        memory_item.content = chunk.text;
        memory_item.metadata["doc_id"] = chunk.doc_id;
        memory_item.metadata["seq_no"] = std::to_string(chunk.seq_no);

        vector_store_->insert(embedding, i, memory_item);
        doc_to_vector_id_[get_doc_key(chunk.doc_id, chunk.seq_no)] = i;
    }
}

std::vector<RetrievalResult> FusionRetriever::query(const std::string& query_text, int top_k) {
    switch (config_.strategy) {
        case FusionStrategy::BM25_ONLY:
            return bm25_retrieve(query_text, top_k);

        case FusionStrategy::VECTOR_ONLY:
            return vector_retrieve(query_text, top_k);

        case FusionStrategy::HYBRID:
        case FusionStrategy::RRF:
        case FusionStrategy::WEIGHTED: {
            // 并行检索
            auto bm25_future = std::async(std::launch::async, [this, query_text, top_k]() {
                return bm25_retrieve(query_text, config_.max_candidates);
            });

            auto vector_future = std::async(std::launch::async, [this, query_text, top_k]() {
                return vector_retrieve(query_text, config_.max_candidates);
            });

            auto bm25_results = bm25_future.get();
            auto vector_results = vector_future.get();

            return fuse_results(bm25_results, vector_results, top_k);
        }
    }

    return {};
}

std::future<std::vector<RetrievalResult>> FusionRetriever::query_async(const std::string& query_text, int top_k) {
    return std::async(std::launch::async, [this, query_text, top_k]() {
        return query(query_text, top_k);
    });
}

std::vector<RetrievalResult> FusionRetriever::bm25_retrieve(const std::string& query_text, int top_k) {
    if (!bm25_indexer_) {
        return {};
    }

    // 简单的查询分词
    std::vector<std::string> terms;
    std::istringstream iss(query_text);
    std::string term;
    while (iss >> term) {
        terms.push_back(term);
    }

    auto bm25_scores = bm25_indexer_->query(terms, top_k);

    std::vector<RetrievalResult> results;
    for (const auto& score_pair : bm25_scores) {
        size_t chunk_idx = score_pair.first;
        double score = score_pair.second;

        if (chunk_idx < chunks_.size()) {
            const auto& chunk = chunks_[chunk_idx];
            results.emplace_back(chunk.doc_id, chunk.seq_no, score, chunk.text);
        }
    }

    return results;
}

std::vector<RetrievalResult> FusionRetriever::vector_retrieve(const std::string& query_text, int top_k) {
    if (!vector_store_ || !embedding_model_) {
        return {};
    }

    // 生成查询向量
    auto query_embedding = embedding_model_->embed(query_text, humanus::EmbeddingType::QUERY);

    // 向量检索
    auto memory_items = vector_store_->search(query_embedding, top_k);

    std::vector<RetrievalResult> results;
    for (const auto& item : memory_items) {
        std::string doc_id = item.metadata.at("doc_id");
        int seq_no = std::stoi(item.metadata.at("seq_no"));
        double score = item.similarity;  // 相似度分数

        results.emplace_back(doc_id, seq_no, score, item.content);
    }

    return results;
}

std::vector<RetrievalResult> FusionRetriever::fuse_results(
    const std::vector<RetrievalResult>& bm25_results,
    const std::vector<RetrievalResult>& vector_results,
    int top_k) {

    switch (config_.strategy) {
        case FusionStrategy::RRF:
            return rrf_fusion(bm25_results, vector_results, top_k);

        case FusionStrategy::WEIGHTED:
        case FusionStrategy::HYBRID:
        default:
            return weighted_fusion(bm25_results, vector_results, top_k);
    }
}

std::vector<RetrievalResult> FusionRetriever::rrf_fusion(
    const std::vector<RetrievalResult>& bm25_results,
    const std::vector<RetrievalResult>& vector_results,
    int top_k) {

    std::unordered_map<std::string, double> doc_scores;
    std::unordered_map<std::string, RetrievalResult> doc_map;

    // BM25结果的RRF分数
    for (size_t i = 0; i < bm25_results.size(); ++i) {
        const auto& result = bm25_results[i];
        std::string doc_key = get_doc_key(result.doc_id, result.seq_no);
        double rrf_score = 1.0 / (config_.rrf_k + i + 1);
        doc_scores[doc_key] += config_.bm25_weight * rrf_score;
        doc_map[doc_key] = result;
    }

    // 向量结果的RRF分数
    for (size_t i = 0; i < vector_results.size(); ++i) {
        const auto& result = vector_results[i];
        std::string doc_key = get_doc_key(result.doc_id, result.seq_no);
        double rrf_score = 1.0 / (config_.rrf_k + i + 1);
        doc_scores[doc_key] += config_.vector_weight * rrf_score;

        if (doc_map.find(doc_key) == doc_map.end()) {
            doc_map[doc_key] = result;
        }
    }

    // 排序并返回top_k
    std::vector<std::pair<std::string, double>> sorted_docs(doc_scores.begin(), doc_scores.end());
    std::sort(sorted_docs.begin(), sorted_docs.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    std::vector<RetrievalResult> results;
    for (size_t i = 0; i < std::min(static_cast<size_t>(top_k), sorted_docs.size()); ++i) {
        const std::string& doc_key = sorted_docs[i].first;
        double score = sorted_docs[i].second;

        auto result = doc_map[doc_key];
        result.score = score;  // 更新为融合分数
        results.push_back(result);
    }

    return results;
}

std::vector<RetrievalResult> FusionRetriever::weighted_fusion(
    const std::vector<RetrievalResult>& bm25_results,
    const std::vector<RetrievalResult>& vector_results,
    int top_k) {

    // 归一化分数
    auto norm_bm25 = bm25_results;
    auto norm_vector = vector_results;
    normalize_scores(norm_bm25);
    normalize_scores(norm_vector);

    std::unordered_map<std::string, double> doc_scores;
    std::unordered_map<std::string, RetrievalResult> doc_map;

    // BM25加权分数
    for (const auto& result : norm_bm25) {
        std::string doc_key = get_doc_key(result.doc_id, result.seq_no);
        doc_scores[doc_key] += config_.bm25_weight * result.score;
        doc_map[doc_key] = result;
    }

    // 向量加权分数
    for (const auto& result : norm_vector) {
        std::string doc_key = get_doc_key(result.doc_id, result.seq_no);
        doc_scores[doc_key] += config_.vector_weight * result.score;

        if (doc_map.find(doc_key) == doc_map.end()) {
            doc_map[doc_key] = result;
        }
    }

    // 排序并返回top_k
    std::vector<std::pair<std::string, double>> sorted_docs(doc_scores.begin(), doc_scores.end());
    std::sort(sorted_docs.begin(), sorted_docs.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    std::vector<RetrievalResult> results;
    for (size_t i = 0; i < std::min(static_cast<size_t>(top_k), sorted_docs.size()); ++i) {
        const std::string& doc_key = sorted_docs[i].first;
        double score = sorted_docs[i].second;

        auto result = doc_map[doc_key];
        result.score = score;  // 更新为融合分数
        results.push_back(result);
    }

    return results;
}

std::vector<RetrievalResult> FusionRetriever::deduplicate_results(const std::vector<RetrievalResult>& results) {
    std::set<std::string> seen;
    std::vector<RetrievalResult> dedup_results;

    for (const auto& result : results) {
        std::string doc_key = get_doc_key(result.doc_id, result.seq_no);
        if (seen.find(doc_key) == seen.end()) {
            seen.insert(doc_key);
            dedup_results.push_back(result);
        }
    }

    return dedup_results;
}

void FusionRetriever::normalize_scores(std::vector<RetrievalResult>& results) {
    if (results.empty()) return;

    // 找到最大和最小分数
    double max_score = results[0].score;
    double min_score = results[0].score;

    for (const auto& result : results) {
        max_score = std::max(max_score, result.score);
        min_score = std::min(min_score, result.score);
    }

    // 归一化到[0,1]
    double range = max_score - min_score;
    if (range > 0) {
        for (auto& result : results) {
            result.score = (result.score - min_score) / range;
        }
    }
}

} // namespace rag

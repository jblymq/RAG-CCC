#ifndef RAG_FUSION_RETRIEVER_H
#define RAG_FUSION_RETRIEVER_H

#include "chunk.h"
#include "bm25.h"
#include "config.h"
#include <vector>
#include <memory>
#include <future>
#include <unordered_set>

// 前向声明，避免完整包含
namespace humanus {
    class VectorStore;
    class EmbeddingModel;
    enum class EmbeddingType;
}

namespace rag {

// 检索结果结构
struct RetrievalResult {
    std::string doc_id;
    int seq_no;
    double score;
    std::string text;

    // 默认构造函数
    RetrievalResult() : doc_id(""), seq_no(0), score(0.0), text("") {}

    RetrievalResult(const std::string& id, int seq, double s, const std::string& t)
        : doc_id(id), seq_no(seq), score(s), text(t) {}
};

// 融合策略枚举
enum class FusionStrategy {
    BM25_ONLY,      // 仅使用BM25
    VECTOR_ONLY,    // 仅使用向量检索
    HYBRID,         // 混合检索
    RRF,           // Reciprocal Rank Fusion
    WEIGHTED       // 加权融合
};

// 融合检索器配置
struct FusionRetrieverConfig {
    FusionStrategy strategy = FusionStrategy::HYBRID;
    double bm25_weight = 0.5;           // BM25权重
    double vector_weight = 0.5;         // 向量权重
    int max_candidates = 100;           // 候选结果最大数量
    double rrf_k = 60.0;               // RRF参数k值
    bool enable_rerank = true;          // 是否启用重排序

    // 从RAGConfig读取
    static FusionRetrieverConfig from_rag_config(const RAGConfig& config) {
        FusionRetrieverConfig fusion_config;

        // 根据配置决定策略
        if (config.fusion.bm25_weight > 0 && config.fusion.vector_weight > 0) {
            fusion_config.strategy = FusionStrategy::HYBRID;
        } else if (config.fusion.bm25_weight > 0) {
            fusion_config.strategy = FusionStrategy::BM25_ONLY;
        } else {
            fusion_config.strategy = FusionStrategy::VECTOR_ONLY;
        }

        fusion_config.bm25_weight = config.fusion.bm25_weight;
        fusion_config.vector_weight = config.fusion.vector_weight;
        fusion_config.max_candidates = config.fusion.max_candidates;
        fusion_config.rrf_k = config.fusion.rrf_k;
        fusion_config.enable_rerank = config.fusion.enable_rerank;

        return fusion_config;
    }
};

// BM25+HNSW融合检索器
class FusionRetriever {
private:
    std::shared_ptr<BM25Indexer> bm25_indexer_;
    std::shared_ptr<humanus::VectorStore> vector_store_;
    std::shared_ptr<humanus::EmbeddingModel> embedding_model_;
    FusionRetrieverConfig config_;

    std::vector<Chunk> chunks_;  // 保存所有chunks的引用
    std::unordered_map<std::string, size_t> doc_to_vector_id_;  // 文档ID到向量ID的映射

public:
    // 构造函数
    FusionRetriever(const FusionRetrieverConfig& config,
                   std::shared_ptr<humanus::VectorStore> vector_store = nullptr,
                   std::shared_ptr<humanus::EmbeddingModel> embedding_model = nullptr);

    // 从RAGConfig构造
    static std::shared_ptr<FusionRetriever> from_config(const RAGConfig& config);

    // 构建索引
    void fit(const std::vector<Chunk>& chunks);

    // 查询接口
    std::vector<RetrievalResult> query(const std::string& query_text, int top_k = 10);

    // 异步查询
    std::future<std::vector<RetrievalResult>> query_async(const std::string& query_text, int top_k = 10);

private:
    // BM25检索
    std::vector<RetrievalResult> bm25_retrieve(const std::string& query_text, int top_k);

    // 向量检索
    std::vector<RetrievalResult> vector_retrieve(const std::string& query_text, int top_k);

    // 结果融合
    std::vector<RetrievalResult> fuse_results(
        const std::vector<RetrievalResult>& bm25_results,
        const std::vector<RetrievalResult>& vector_results,
        int top_k);

    // RRF融合
    std::vector<RetrievalResult> rrf_fusion(
        const std::vector<RetrievalResult>& bm25_results,
        const std::vector<RetrievalResult>& vector_results,
        int top_k);

    // 加权融合
    std::vector<RetrievalResult> weighted_fusion(
        const std::vector<RetrievalResult>& bm25_results,
        const std::vector<RetrievalResult>& vector_results,
        int top_k);

    // 去重
    std::vector<RetrievalResult> deduplicate_results(const std::vector<RetrievalResult>& results);

    // 归一化分数
    void normalize_scores(std::vector<RetrievalResult>& results);

    // 生成文档唯一键
    std::string get_doc_key(const std::string& doc_id, int seq_no) const {
        return doc_id + "_" + std::to_string(seq_no);
    }
};

} // namespace rag

#endif // RAG_FUSION_RETRIEVER_H

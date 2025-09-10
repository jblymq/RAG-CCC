#pragma once
#include <string>
#include <memory>

namespace rag {

struct ChunkConfig {
    int size = 512;
    int overlap = 128;
    int min_size = 64;
};

struct BM25Config {
    double k1 = 1.5;
    double b = 0.75;
};

struct HNSWConfig {
    int M = 16;
    int ef_construction = 200;
    int ef_query = 50;
    int vector_dim = 768;
    int max_elements = 10000;
};

struct FusionConfig {
    double bm25_weight = 0.5;
    double vector_weight = 0.5;
    int max_candidates = 100;
    double rrf_k = 60.0;
    bool enable_rerank = true;
    std::string strategy = "hybrid";  // "bm25_only", "vector_only", "hybrid", "rrf", "weighted"
};

struct CacheConfig {
    size_t capacity = 1024;
    int ttl_seconds = 3600;
};

struct ThreadPoolConfig {
    size_t num_workers = 8;
};

struct TunerConfig {
    double latency_max_ms = 200.0;
    double recall_min_pct = 0.8;
    int ef_delta = 5;
    int topk_delta = 2;
    bool enable = true;
    int check_interval_seconds = 10;
};

struct SQLiteConfig {
    std::string db_path = "rag_store.db";           // 数据库文件路径
    std::string vector_extension = "sqlite_vec";    // 向量扩展名
    int vector_dimension = 768;                     // 向量维度
    bool enable_fts5 = true;                        // 启用 FTS5
    bool enable_wal = true;                         // 启用 WAL 模式
    int cache_size = 10000;                         // 缓存页数
    int busy_timeout = 30000;                       // 忙等待超时（ms）
    int fts5_limit = 50;                           // FTS5 检索数量
    int vector_limit = 50;                         // 向量检索数量
};

struct RAGConfig {
    ChunkConfig chunk;
    BM25Config bm25;
    HNSWConfig hnsw;
    FusionConfig fusion;
    CacheConfig cache;
    ThreadPoolConfig threadpool;
    TunerConfig tuner;
    SQLiteConfig sqlite;
};

class ConfigLoader {
public:
    static std::shared_ptr<RAGConfig> load(const std::string& config_path = "rag/rag_config.toml");
    static std::shared_ptr<RAGConfig> get_instance();

private:
    static std::shared_ptr<RAGConfig> instance_;
};

} // namespace rag

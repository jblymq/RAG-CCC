/**
 * SQLite RAG 检索器 - 基于 SQLite 的统一检索接口
 *
 * 集成 FTS5 文本检索和向量检索，提供统一的 RAG 检索服务
 */

#pragma once

#include "sqlite_db.h"
#include "config.h"
#include "chunk.h"
#include "lru_cache.h"
#include "thread_pool.h"
#include <memory>
#include <vector>
#include <string>
#include <future>
#include <functional>

namespace rag {

/**
 * 检索策略枚举
 */
enum class SQLiteRetrievalStrategy {
    FTS5_ONLY,      // 仅使用 FTS5 文本检索
    VECTOR_ONLY,    // 仅使用向量检索
    HYBRID,         // 混合检索（FTS5 + 向量）
    ADAPTIVE        // 自适应策略（根据查询类型选择）
};

/**
 * SQLite RAG 检索器配置
 */
struct SQLiteRetrieverConfig {
    SQLiteRetrievalStrategy strategy = SQLiteRetrievalStrategy::HYBRID;
    double fts5_weight = 0.6;          // FTS5 权重
    double vector_weight = 0.4;        // 向量权重
    int max_results = 10;              // 最大返回结果数
    bool enable_cache = true;          // 启用结果缓存
    bool enable_parallel = true;       // 启用并行检索

    // 从 RAGConfig 创建配置
    static SQLiteRetrieverConfig from_rag_config(const RAGConfig& config);
};

/**
 * SQLite RAG 检索器
 *
 * 提供基于 SQLite 的统一文档检索服务，支持：
 * - FTS5 全文检索
 * - 向量语义检索
 * - 混合检索策略
 * - 结果缓存
 * - 并行查询
 */
class SQLiteRetriever {
public:
    /**
     * 构造函数
     * @param config 检索器配置
     * @param embed_func 文本嵌入函数
     */
    SQLiteRetriever(
        const SQLiteRetrieverConfig& config,
        std::function<std::vector<float>(const std::string&)> embed_func = nullptr
    );

    /**
     * 构造函数（从 RAGConfig）
     * @param rag_config RAG 配置
     * @param embed_func 文本嵌入函数
     */
    SQLiteRetriever(
        const RAGConfig& rag_config,
        std::function<std::vector<float>(const std::string&)> embed_func = nullptr
    );

    ~SQLiteRetriever();

    /**
     * 初始化数据库和索引
     * @return 是否初始化成功
     */
    bool initialize();

    /**
     * 插入文档集合
     * @param chunks 文档块列表
     * @return 成功插入的文档数量
     */
    size_t insert_documents(const std::vector<Chunk>& chunks);

    /**
     * 查询文档
     * @param query 查询文本
     * @param limit 返回结果数量限制
     * @return 检索结果列表
     */
    std::vector<SQLiteSearchResult> query(
        const std::string& query,
        int limit = -1  // -1 表示使用配置的默认值
    );

    /**
     * 异步查询文档
     * @param query 查询文本
     * @param limit 返回结果数量限制
     * @return 检索结果 future
     */
    std::future<std::vector<SQLiteSearchResult>> query_async(
        const std::string& query,
        int limit = -1
    );

    /**
     * 仅文本检索（FTS5）
     * @param query 查询文本
     * @param limit 返回结果数量限制
     * @return 检索结果列表
     */
    std::vector<SQLiteSearchResult> query_text_only(
        const std::string& query,
        int limit = -1
    );

    /**
     * 仅向量检索
     * @param query 查询文本（会转换为向量）
     * @param limit 返回结果数量限制
     * @return 检索结果列表
     */
    std::vector<SQLiteSearchResult> query_vector_only(
        const std::string& query,
        int limit = -1
    );

    /**
     * 混合检索（FTS5 + 向量）
     * @param query 查询文本
     * @param limit 返回结果数量限制
     * @return 检索结果列表
     */
    std::vector<SQLiteSearchResult> query_hybrid(
        const std::string& query,
        int limit = -1
    );

    /**
     * 根据 chunk IDs 获取文档
     */
    std::vector<SQLiteSearchResult> get_documents_by_ids(
        const std::vector<size_t>& chunk_ids);

    /**
     * 清空所有数据
     * @return 是否成功
     */
    bool clear_all_data();

    /**
     * 获取数据库统计信息
     * @return 统计信息
     */
    SQLiteDB::DBStats get_stats();

    /**
     * 更新检索配置
     * @param new_config 新配置
     */
    void update_config(const SQLiteRetrieverConfig& new_config);

    /**
     * 获取当前配置
     * @return 当前配置
     */
    const SQLiteRetrieverConfig& get_config() const { return config_; }

    /**
     * 检查检索器是否可用
     * @return 是否可用
     */
    bool is_available() const;

    /**
     * 设置嵌入函数
     * @param embed_func 新的嵌入函数
     */
    void set_embedding_function(
        std::function<std::vector<float>(const std::string&)> embed_func
    );

    /**
     * 预热查询（用于性能测试）
     * @param sample_queries 样本查询列表
     */
    void warmup(const std::vector<std::string>& sample_queries = {});

private:
    SQLiteRetrieverConfig config_;
    std::unique_ptr<SQLiteDB> db_;
    std::unique_ptr<LRUCache> cache_;
    std::unique_ptr<ThreadPool> thread_pool_;
    std::function<std::vector<float>(const std::string&)> embed_func_;

    bool initialized_;

    /**
     * 初始化缓存
     */
    void init_cache(const CacheConfig& cache_config);

    /**
     * 初始化线程池
     */
    void init_thread_pool(const ThreadPoolConfig& tp_config);

    /**
     * 默认嵌入函数（随机向量，用于测试）
     */
    std::vector<float> default_embedding(const std::string& text);

    /**
     * 生成缓存键
     */
    std::string generate_cache_key(
        const std::string& query,
        SQLiteRetrievalStrategy strategy,
        int limit
    );

    /**
     * 从缓存获取结果
     */
    bool get_from_cache(
        const std::string& cache_key,
        std::vector<SQLiteSearchResult>& results
    );

    /**
     * 将结果存入缓存
     */
    void put_to_cache(
        const std::string& cache_key,
        const std::vector<SQLiteSearchResult>& results
    );

    /**
     * 根据查询自动选择检索策略
     */
    SQLiteRetrievalStrategy choose_strategy(const std::string& query);

    /**
     * 合并和排序检索结果
     */
    std::vector<SQLiteSearchResult> merge_and_rank_results(
        const std::vector<SQLiteSearchResult>& fts5_results,
        const std::vector<SQLiteSearchResult>& vector_results,
        int limit
    );

    /**
     * 日志输出
     */
    void log_info(const std::string& message);
    void log_error(const std::string& message);
};

/**
 * SQLite RAG 系统管理器
 *
 * 高级封装，提供完整的 RAG 系统管理功能
 */
class SQLiteRAGSystem {
public:
    /**
     * 构造函数
     * @param config_path 配置文件路径
     */
    explicit SQLiteRAGSystem(const std::string& config_path = "rag_config.toml");

    /**
     * 初始化系统
     * @return 是否初始化成功
     */
    bool initialize();

    /**
     * 加载文档集合
     * @param documents 文档列表
     * @return 成功加载的文档数量
     */
    size_t load_documents(const std::vector<Chunk>& documents);

    /**
     * 从文件加载文档
     * @param file_path 文件路径
     * @return 成功加载的文档数量
     */
    size_t load_documents_from_file(const std::string& file_path);

    /**
     * 查询文档
     * @param query 查询文本
     * @param limit 结果数量限制
     * @return 检索结果
     */
    std::vector<SQLiteSearchResult> search(
        const std::string& query,
        int limit = 10
    );

    /**
     * 获取系统统计信息
     */
    SQLiteDB::DBStats get_system_stats();

    /**
     * 获取检索器
     */
    std::shared_ptr<SQLiteRetriever> get_retriever() { return retriever_; }

private:
    std::shared_ptr<RAGConfig> config_;
    std::shared_ptr<SQLiteRetriever> retriever_;
    bool initialized_;

    /**
     * 默认文档分块函数
     */
    std::vector<Chunk> chunk_text(
        const std::string& text,
        const std::string& doc_id = "default"
    );

    /**
     * 简单的嵌入函数（可替换为实际的模型）
     */
    std::vector<float> simple_embedding(const std::string& text);
};

} // namespace rag

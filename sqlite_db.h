/**
 * SQLite 数据库管理器 - 支持 FTS5 和向量扩展
 *
 * 功能特性：
 * - SQLite 连接管理和事务控制
 * - FTS5 全文检索（BM25）
 * - 向量扩展支持（sqlite-vec/sqlite-vss）
 * - 自动 schema 初始化
 * - 线程安全的数据库操作
 */

#pragma once

#include <sqlite3.h>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <functional>
#include "chunk.h"
#include "config.h"

namespace rag {

/**
 * 检索结果结构
 */
struct SQLiteSearchResult {
    int chunk_id;
    double score;
    std::string doc_id;
    std::string content;
    std::string topic;
};

/**
 * SQLite 数据库管理器
 *
 * 提供统一的数据库访问接口，支持文档存储、FTS5 检索和向量检索
 */
class SQLiteDB {
public:
    explicit SQLiteDB(const SQLiteConfig& config);
    ~SQLiteDB();

    // 禁用拷贝构造和赋值
    SQLiteDB(const SQLiteDB&) = delete;
    SQLiteDB& operator=(const SQLiteDB&) = delete;

    /**
     * 初始化数据库 schema
     * 创建必要的表和索引
     */
    bool initialize_schema();

    /**
     * 插入文档块
     * @param chunks 文档块列表
     * @param embed_func 嵌入计算函数
     * @return 成功插入的文档数量
     */
    size_t insert_chunks(
        const std::vector<Chunk>& chunks,
        std::function<std::vector<float>(const std::string&)> embed_func
    );

    /**
     * FTS5 全文检索
     * @param query 查询字符串
     * @param limit 返回结果数量
     * @return 检索结果列表，按 BM25 分数排序
     */
    std::vector<SQLiteSearchResult> search_fts5(
        const std::string& query,
        int limit = 10
    );

    /**
     * 向量检索
     * @param query_embedding 查询向量
     * @param limit 返回结果数量
     * @return 检索结果列表，按向量相似度排序
     */
    std::vector<SQLiteSearchResult> search_vector(
        const std::vector<float>& query_embedding,
        int limit = 10
    );

    /**
     * 混合检索（FTS5 + 向量）
     * @param query_text 查询文本
     * @param query_embedding 查询向量
     * @param fts5_limit FTS5 结果数量
     * @param vector_limit 向量结果数量
     * @param fts5_weight FTS5 权重
     * @param vector_weight 向量权重
     * @return 融合后的检索结果
     */
    std::vector<SQLiteSearchResult> search_hybrid(
        const std::string& query_text,
        const std::vector<float>& query_embedding,
        int fts5_limit = 50,
        int vector_limit = 50,
        double fts5_weight = 0.6,
        double vector_weight = 0.4
    );

    /**
     * 获取文档块信息
     * @param chunk_ids 文档块ID列表
     * @return 文档块详细信息
     */
    std::vector<SQLiteSearchResult> get_chunks_by_ids(
        const std::vector<int>& chunk_ids
    );

    /**
     * 清空所有数据
     */
    bool clear_all_data();

    /**
     * 获取数据库统计信息
     */
    struct DBStats {
        int total_chunks;
        int total_embeddings;
        double db_size_mb;
        std::string last_update;
    };
    DBStats get_stats();

    /**
     * 执行 SQL 语句（用于调试和维护）
     * @param sql SQL 语句
     * @param callback 结果回调函数
     * @return 是否执行成功
     */
    bool execute_sql(
        const std::string& sql,
        std::function<void(sqlite3_stmt*)> callback = nullptr
    );

    /**
     * 开始事务
     */
    bool begin_transaction();

    /**
     * 提交事务
     */
    bool commit_transaction();

    /**
     * 回滚事务
     */
    bool rollback_transaction();

    /**
     * 获取原始数据库句柄（谨慎使用）
     */
    sqlite3* handle() const { return db_; }

    /**
     * 检查数据库是否可用
     */
    bool is_valid() const { return db_ != nullptr; }

private:
    sqlite3* db_;
    SQLiteConfig config_;
    mutable std::mutex db_mutex_;
    bool schema_initialized_;

    /**
     * 加载向量扩展
     */
    bool load_vector_extension();

    /**
     * 创建表结构
     */
    bool create_tables();

    /**
     * 创建索引
     */
    bool create_indexes();

    /**
     * 优化数据库配置
     */
    bool optimize_database();

    /**
     * 归一化分数到 [0, 1] 范围
     */
    double normalize_score(double score, double min_score, double max_score);

    /**
     * 执行 prepared statement
     */
    bool execute_prepared(
        const std::string& sql,
        std::function<void(sqlite3_stmt*)> bind_func,
        std::function<void(sqlite3_stmt*)> result_func = nullptr
    );

    /**
     * 错误处理
     */
    void log_error(const std::string& operation, int error_code = -1);
};

/**
 * RAII 事务管理器
 */
class SQLiteTransaction {
public:
    explicit SQLiteTransaction(SQLiteDB& db);
    ~SQLiteTransaction();

    bool commit();
    void rollback();

private:
    SQLiteDB& db_;
    bool committed_;
    bool active_;
};

} // namespace rag

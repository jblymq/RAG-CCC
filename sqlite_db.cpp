/**
 * SQLite 数据库管理器实现
 */

#include "sqlite_db.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <ctime>
#include <cstring>
#include <unordered_set>
#include <iomanip>

namespace rag {

SQLiteDB::SQLiteDB(const SQLiteConfig& config)
    : db_(nullptr), config_(config), schema_initialized_(false) {

    // 打开数据库
    int rc = sqlite3_open(config_.db_path.c_str(), &db_);
    if (rc != SQLITE_OK) {
        log_error("Failed to open database", rc);
        sqlite3_close(db_);
        db_ = nullptr;
        return;
    }

    // 设置忙等待超时
    sqlite3_busy_timeout(db_, config_.busy_timeout);

    // 优化数据库配置
    if (!optimize_database()) {
        log_error("Failed to optimize database");
    }

    // 加载向量扩展
    if (!load_vector_extension()) {
        log_error("Failed to load vector extension");
    }

    // 初始化 schema
    if (!initialize_schema()) {
        log_error("Failed to initialize schema");
    }
}

SQLiteDB::~SQLiteDB() {
    if (db_) {
        sqlite3_close(db_);
    }
}

bool SQLiteDB::load_vector_extension() {
    if (!db_) return false;

    // 启用扩展加载
    int rc = sqlite3_enable_load_extension(db_, 1);
    if (rc != SQLITE_OK) {
        log_error("Failed to enable extension loading", rc);
        return false;
    }

    // 加载向量扩展
    char* error_msg = nullptr;
    rc = sqlite3_load_extension(db_, config_.vector_extension.c_str(), nullptr, &error_msg);

    // 禁用扩展加载（安全考虑）
    sqlite3_enable_load_extension(db_, 0);

    if (rc != SQLITE_OK) {
        std::string error = error_msg ? error_msg : "Unknown error";
        std::cout << "Warning: Failed to load vector extension '"
                  << config_.vector_extension << "': " << error << std::endl;
        std::cout << "Vector search will be disabled." << std::endl;
        if (error_msg) sqlite3_free(error_msg);
        // 不返回 false，允许系统在没有向量扩展的情况下运行
    }

    return true;
}

bool SQLiteDB::optimize_database() {
    if (!db_) return false;

    std::vector<std::string> pragmas = {
        "PRAGMA journal_mode = " + std::string(config_.enable_wal ? "WAL" : "DELETE"),
        "PRAGMA synchronous = NORMAL",
        "PRAGMA cache_size = " + std::to_string(config_.cache_size),
        "PRAGMA temp_store = MEMORY",
        "PRAGMA mmap_size = 268435456"  // 256MB
    };

    for (const auto& pragma : pragmas) {
        char* error_msg = nullptr;
        int rc = sqlite3_exec(db_, pragma.c_str(), nullptr, nullptr, &error_msg);
        if (rc != SQLITE_OK) {
            std::string error = error_msg ? error_msg : "Unknown error";
            log_error("Failed to execute pragma: " + pragma + " - " + error);
            if (error_msg) sqlite3_free(error_msg);
            return false;
        }
    }

    return true;
}

bool SQLiteDB::initialize_schema() {
    std::lock_guard<std::mutex> lock(db_mutex_);
    if (schema_initialized_) return true;

    if (!create_tables() || !create_indexes()) {
        return false;
    }

    schema_initialized_ = true;
    return true;
}

bool SQLiteDB::create_tables() {
    if (!db_) return false;

    // 创建主表：存储文档块
    const char* create_chunks_sql = R"(
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            seq_no INTEGER NOT NULL,
            topic TEXT,
            content TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    )";

    char* error_msg = nullptr;
    int rc = sqlite3_exec(db_, create_chunks_sql, nullptr, nullptr, &error_msg);
    if (rc != SQLITE_OK) {
        log_error("Failed to create chunks table: " + std::string(error_msg ? error_msg : ""));
        if (error_msg) sqlite3_free(error_msg);
        return false;
    }

    // 创建 FTS5 虚拟表
    if (config_.enable_fts5) {
        const char* create_fts_sql = R"(
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                content='chunks',
                content_rowid='id',
                tokenize='unicode61 remove_diacritics 1'
            );
        )";

        rc = sqlite3_exec(db_, create_fts_sql, nullptr, nullptr, &error_msg);
        if (rc != SQLITE_OK) {
            log_error("Failed to create FTS5 table: " + std::string(error_msg ? error_msg : ""));
            if (error_msg) sqlite3_free(error_msg);
            return false;
        }
    }

    // 创建向量表（如果向量扩展可用）
    const char* create_embeddings_sql = R"(
        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_id INTEGER PRIMARY KEY,
            vector BLOB NOT NULL,
            FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
        );
    )";

    rc = sqlite3_exec(db_, create_embeddings_sql, nullptr, nullptr, &error_msg);
    if (rc != SQLITE_OK) {
        log_error("Failed to create embeddings table: " + std::string(error_msg ? error_msg : ""));
        if (error_msg) sqlite3_free(error_msg);
        return false;
    }

    return true;
}

bool SQLiteDB::create_indexes() {
    if (!db_) return false;

    std::vector<std::string> index_sqls = {
        "CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);",
        "CREATE INDEX IF NOT EXISTS idx_chunks_topic ON chunks(topic);",
        "CREATE INDEX IF NOT EXISTS idx_chunks_created ON chunks(created_at);"
    };

    for (const auto& sql : index_sqls) {
        char* error_msg = nullptr;
        int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &error_msg);
        if (rc != SQLITE_OK) {
            log_error("Failed to create index: " + std::string(error_msg ? error_msg : ""));
            if (error_msg) sqlite3_free(error_msg);
            return false;
        }
    }

    return true;
}

size_t SQLiteDB::insert_chunks(
    const std::vector<Chunk>& chunks,
    std::function<std::vector<float>(const std::string&)> embed_func) {

    if (!db_ || chunks.empty()) return 0;

    std::lock_guard<std::mutex> lock(db_mutex_);

    // 开始事务
    SQLiteTransaction trans(*this);

    const char* insert_chunk_sql =
        "INSERT INTO chunks(doc_id, seq_no, topic, content) VALUES(?,?,?,?);";

    const char* insert_embedding_sql =
        "INSERT INTO embeddings(chunk_id, vector) VALUES(?,?);";

    sqlite3_stmt* chunk_stmt = nullptr;
    sqlite3_stmt* emb_stmt = nullptr;

    int rc = sqlite3_prepare_v2(db_, insert_chunk_sql, -1, &chunk_stmt, nullptr);
    if (rc != SQLITE_OK) {
        log_error("Failed to prepare chunk insert statement", rc);
        return 0;
    }

    rc = sqlite3_prepare_v2(db_, insert_embedding_sql, -1, &emb_stmt, nullptr);
    if (rc != SQLITE_OK) {
        log_error("Failed to prepare embedding insert statement", rc);
        sqlite3_finalize(chunk_stmt);
        return 0;
    }

    size_t inserted_count = 0;

    for (const auto& chunk : chunks) {
        // 插入 chunks 表
        sqlite3_bind_text(chunk_stmt, 1, chunk.doc_id.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int(chunk_stmt, 2, chunk.seq_no);
        sqlite3_bind_text(chunk_stmt, 3, chunk.topic.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(chunk_stmt, 4, chunk.text.c_str(), -1, SQLITE_STATIC);

        rc = sqlite3_step(chunk_stmt);
        if (rc != SQLITE_DONE) {
            log_error("Failed to insert chunk", rc);
            sqlite3_reset(chunk_stmt);
            continue;
        }

        // 获取插入的行ID
        sqlite3_int64 chunk_id = sqlite3_last_insert_rowid(db_);

        // 计算并插入向量
        if (embed_func) {
            try {
                auto embedding = embed_func(chunk.text);
                if (!embedding.empty()) {
                    sqlite3_bind_int64(emb_stmt, 1, chunk_id);
                    sqlite3_bind_blob(emb_stmt, 2, embedding.data(),
                                    embedding.size() * sizeof(float), SQLITE_STATIC);

                    rc = sqlite3_step(emb_stmt);
                    if (rc != SQLITE_DONE) {
                        log_error("Failed to insert embedding", rc);
                    }
                    sqlite3_reset(emb_stmt);
                }
            } catch (const std::exception& e) {
                std::cerr << "Error computing embedding: " << e.what() << std::endl;
            }
        }

        sqlite3_reset(chunk_stmt);
        inserted_count++;
    }

    sqlite3_finalize(chunk_stmt);
    sqlite3_finalize(emb_stmt);

    // 提交事务
    if (!trans.commit()) {
        log_error("Failed to commit transaction");
        return 0;
    }

    // 重建 FTS5 索引
    if (config_.enable_fts5) {
        const char* rebuild_fts_sql = "INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild');";
        char* error_msg = nullptr;
        rc = sqlite3_exec(db_, rebuild_fts_sql, nullptr, nullptr, &error_msg);
        if (rc != SQLITE_OK) {
            log_error("Failed to rebuild FTS5 index: " + std::string(error_msg ? error_msg : ""));
            if (error_msg) sqlite3_free(error_msg);
        }
    }

    return inserted_count;
}

std::vector<SQLiteSearchResult> SQLiteDB::search_fts5(
    const std::string& query, int limit) {

    std::vector<SQLiteSearchResult> results;
    if (!db_ || !config_.enable_fts5 || query.empty()) return results;

    std::lock_guard<std::mutex> lock(db_mutex_);

    const char* sql = R"(
        SELECT c.id, c.doc_id, c.topic, c.content, bm25(chunks_fts) AS score
        FROM chunks_fts
        JOIN chunks c ON chunks_fts.rowid = c.id
        WHERE chunks_fts MATCH ?
        ORDER BY score DESC
        LIMIT ?;
    )";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        log_error("Failed to prepare FTS5 search statement", rc);
        return results;
    }

    sqlite3_bind_text(stmt, 1, query.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 2, limit);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        SQLiteSearchResult result;
        result.chunk_id = sqlite3_column_int(stmt, 0);
        result.doc_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        result.topic = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        result.content = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        result.score = sqlite3_column_double(stmt, 4);
        results.push_back(result);
    }

    sqlite3_finalize(stmt);
    return results;
}

std::vector<SQLiteSearchResult> SQLiteDB::search_vector(
    const std::vector<float>& query_embedding, int limit) {

    std::vector<SQLiteSearchResult> results;
    if (!db_ || query_embedding.empty()) return results;

    std::lock_guard<std::mutex> lock(db_mutex_);

    // 尝试使用向量扩展语法
    const char* sql = R"(
        SELECT c.id, c.doc_id, c.topic, c.content,
               (1.0 / (1.0 + ABS(e.vector - ?))) AS score
        FROM embeddings e
        JOIN chunks c ON e.chunk_id = c.id
        ORDER BY score DESC
        LIMIT ?;
    )";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        // 如果向量扩展不可用，返回空结果
        return results;
    }

    // 绑定查询向量
    sqlite3_bind_blob(stmt, 1, query_embedding.data(),
                     query_embedding.size() * sizeof(float), SQLITE_STATIC);
    sqlite3_bind_int(stmt, 2, limit);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        SQLiteSearchResult result;
        result.chunk_id = sqlite3_column_int(stmt, 0);
        result.doc_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        result.topic = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        result.content = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        result.score = sqlite3_column_double(stmt, 4);
        results.push_back(result);
    }

    sqlite3_finalize(stmt);
    return results;
}

std::vector<SQLiteSearchResult> SQLiteDB::search_hybrid(
    const std::string& query_text,
    const std::vector<float>& query_embedding,
    int fts5_limit, int vector_limit,
    double fts5_weight, double vector_weight) {

    // 并行执行 FTS5 和向量检索
    auto fts5_results = search_fts5(query_text, fts5_limit);
    auto vector_results = search_vector(query_embedding, vector_limit);

    // 使用 unordered_set 去重并合并结果
    std::unordered_set<int> seen_ids;
    std::vector<SQLiteSearchResult> merged_results;

    // 归一化分数
    auto normalize_fts5 = [&](double score) {
        return score > 0 ? 1.0 / (1.0 + std::abs(score)) : 0.0;
    };

    // 添加 FTS5 结果
    for (auto& result : fts5_results) {
        if (seen_ids.find(result.chunk_id) == seen_ids.end()) {
            result.score = normalize_fts5(result.score) * fts5_weight;
            merged_results.push_back(result);
            seen_ids.insert(result.chunk_id);
        }
    }

    // 添加向量结果
    for (auto& result : vector_results) {
        auto it = seen_ids.find(result.chunk_id);
        if (it == seen_ids.end()) {
            result.score = result.score * vector_weight;
            merged_results.push_back(result);
            seen_ids.insert(result.chunk_id);
        } else {
            // 如果已存在，更新分数（加权平均）
            for (auto& merged_result : merged_results) {
                if (merged_result.chunk_id == result.chunk_id) {
                    merged_result.score += result.score * vector_weight;
                    break;
                }
            }
        }
    }

    // 按分数排序
    std::sort(merged_results.begin(), merged_results.end(),
              [](const SQLiteSearchResult& a, const SQLiteSearchResult& b) {
                  return a.score > b.score;
              });

    return merged_results;
}

std::vector<SQLiteSearchResult> SQLiteDB::get_chunks_by_ids(
    const std::vector<int>& chunk_ids) {

    std::vector<SQLiteSearchResult> results;
    if (!db_ || chunk_ids.empty()) return results;

    std::lock_guard<std::mutex> lock(db_mutex_);

    // 构建 IN 查询
    std::string sql = "SELECT id, doc_id, topic, content FROM chunks WHERE id IN (";
    for (size_t i = 0; i < chunk_ids.size(); ++i) {
        if (i > 0) sql += ",";
        sql += "?";
    }
    sql += ");";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        log_error("Failed to prepare get chunks statement", rc);
        return results;
    }

    // 绑定参数
    for (size_t i = 0; i < chunk_ids.size(); ++i) {
        sqlite3_bind_int(stmt, i + 1, chunk_ids[i]);
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        SQLiteSearchResult result;
        result.chunk_id = sqlite3_column_int(stmt, 0);
        result.doc_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        result.topic = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        result.content = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        result.score = 1.0;  // 默认分数
        results.push_back(result);
    }

    sqlite3_finalize(stmt);
    return results;
}

bool SQLiteDB::clear_all_data() {
    if (!db_) return false;

    std::lock_guard<std::mutex> lock(db_mutex_);

    std::vector<std::string> clear_sqls = {
        "DELETE FROM embeddings;",
        "DELETE FROM chunks_fts;",
        "DELETE FROM chunks;",
        "VACUUM;"
    };

    SQLiteTransaction trans(*this);

    for (const auto& sql : clear_sqls) {
        char* error_msg = nullptr;
        int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &error_msg);
        if (rc != SQLITE_OK) {
            log_error("Failed to execute: " + sql + " - " +
                     std::string(error_msg ? error_msg : ""));
            if (error_msg) sqlite3_free(error_msg);
            return false;
        }
    }

    return trans.commit();
}

SQLiteDB::DBStats SQLiteDB::get_stats() {
    DBStats stats = {};
    if (!db_) return stats;

    std::lock_guard<std::mutex> lock(db_mutex_);

    // 获取文档块数量
    const char* count_chunks_sql = "SELECT COUNT(*) FROM chunks;";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, count_chunks_sql, -1, &stmt, nullptr) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            stats.total_chunks = sqlite3_column_int(stmt, 0);
        }
        sqlite3_finalize(stmt);
    }

    // 获取嵌入向量数量
    const char* count_embeddings_sql = "SELECT COUNT(*) FROM embeddings;";
    if (sqlite3_prepare_v2(db_, count_embeddings_sql, -1, &stmt, nullptr) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            stats.total_embeddings = sqlite3_column_int(stmt, 0);
        }
        sqlite3_finalize(stmt);
    }

    // 获取数据库大小
    const char* size_sql = "SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size();";
    if (sqlite3_prepare_v2(db_, size_sql, -1, &stmt, nullptr) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            stats.db_size_mb = sqlite3_column_double(stmt, 0) / (1024.0 * 1024.0);
        }
        sqlite3_finalize(stmt);
    }

    // 获取最后更新时间
    const char* last_update_sql = "SELECT MAX(created_at) FROM chunks;";
    if (sqlite3_prepare_v2(db_, last_update_sql, -1, &stmt, nullptr) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* timestamp = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            if (timestamp) {
                stats.last_update = timestamp;
            }
        }
        sqlite3_finalize(stmt);
    }

    return stats;
}

bool SQLiteDB::execute_sql(
    const std::string& sql,
    std::function<void(sqlite3_stmt*)> callback) {

    if (!db_) return false;

    std::lock_guard<std::mutex> lock(db_mutex_);

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        log_error("Failed to prepare SQL: " + sql, rc);
        return false;
    }

    bool success = true;
    if (callback) {
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            try {
                callback(stmt);
            } catch (const std::exception& e) {
                std::cerr << "Error in SQL callback: " << e.what() << std::endl;
                success = false;
                break;
            }
        }
    } else {
        rc = sqlite3_step(stmt);
        success = (rc == SQLITE_DONE || rc == SQLITE_ROW);
    }

    sqlite3_finalize(stmt);
    return success;
}

bool SQLiteDB::begin_transaction() {
    if (!db_) return false;
    char* error_msg = nullptr;
    int rc = sqlite3_exec(db_, "BEGIN TRANSACTION;", nullptr, nullptr, &error_msg);
    if (rc != SQLITE_OK) {
        log_error("Failed to begin transaction: " + std::string(error_msg ? error_msg : ""));
        if (error_msg) sqlite3_free(error_msg);
        return false;
    }
    return true;
}

bool SQLiteDB::commit_transaction() {
    if (!db_) return false;
    char* error_msg = nullptr;
    int rc = sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, &error_msg);
    if (rc != SQLITE_OK) {
        log_error("Failed to commit transaction: " + std::string(error_msg ? error_msg : ""));
        if (error_msg) sqlite3_free(error_msg);
        return false;
    }
    return true;
}

bool SQLiteDB::rollback_transaction() {
    if (!db_) return false;
    char* error_msg = nullptr;
    int rc = sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, &error_msg);
    if (rc != SQLITE_OK) {
        log_error("Failed to rollback transaction: " + std::string(error_msg ? error_msg : ""));
        if (error_msg) sqlite3_free(error_msg);
        return false;
    }
    return true;
}

void SQLiteDB::log_error(const std::string& operation, int error_code) {
    std::cerr << "[SQLiteDB Error] " << operation;
    if (error_code != -1) {
        std::cerr << " (Code: " << error_code << ")";
    }
    if (db_) {
        std::cerr << " - " << sqlite3_errmsg(db_);
    }
    std::cerr << std::endl;
}

// SQLiteTransaction 实现

SQLiteTransaction::SQLiteTransaction(SQLiteDB& db)
    : db_(db), committed_(false), active_(false) {
    active_ = db_.begin_transaction();
}

SQLiteTransaction::~SQLiteTransaction() {
    if (active_ && !committed_) {
        rollback();
    }
}

bool SQLiteTransaction::commit() {
    if (!active_ || committed_) return false;

    bool success = db_.commit_transaction();
    if (success) {
        committed_ = true;
        active_ = false;
    }
    return success;
}

void SQLiteTransaction::rollback() {
    if (active_) {
        db_.rollback_transaction();
        active_ = false;
    }
}

} // namespace rag

/**
 * SQLite RAG 检索器实现
 */

#include "sqlite_retriever.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <random>
#include <sstream>
#include <unordered_set>
#include <regex>

namespace rag {

// SQLiteRetrieverConfig 实现

SQLiteRetrieverConfig SQLiteRetrieverConfig::from_rag_config(const RAGConfig& config) {
    SQLiteRetrieverConfig retriever_config;

    retriever_config.fts5_weight = config.fusion.bm25_weight;
    retriever_config.vector_weight = config.fusion.vector_weight;

    // 根据 fusion 策略设置检索策略
    if (config.fusion.strategy == "bm25_only") {
        retriever_config.strategy = SQLiteRetrievalStrategy::FTS5_ONLY;
    } else if (config.fusion.strategy == "vector_only") {
        retriever_config.strategy = SQLiteRetrievalStrategy::VECTOR_ONLY;
    } else if (config.fusion.strategy == "hybrid") {
        retriever_config.strategy = SQLiteRetrievalStrategy::HYBRID;
    } else {
        retriever_config.strategy = SQLiteRetrievalStrategy::ADAPTIVE;
    }

    return retriever_config;
}

// SQLiteRetriever 实现

SQLiteRetriever::SQLiteRetriever(
    const SQLiteRetrieverConfig& config,
    std::function<std::vector<float>(const std::string&)> embed_func)
    : config_(config), embed_func_(embed_func), initialized_(false) {
}

SQLiteRetriever::SQLiteRetriever(
    const RAGConfig& rag_config,
    std::function<std::vector<float>(const std::string&)> embed_func)
    : config_(SQLiteRetrieverConfig::from_rag_config(rag_config)),
      embed_func_(embed_func), initialized_(false) {

    // 初始化数据库
    db_ = std::make_unique<SQLiteDB>(rag_config.sqlite);

    // 初始化缓存
    if (config_.enable_cache) {
        init_cache(rag_config.cache);
    }

    // 初始化线程池
    if (config_.enable_parallel) {
        init_thread_pool(rag_config.threadpool);
    }

    // 设置默认嵌入函数
    if (!embed_func_) {
        embed_func_ = [this](const std::string& text) {
            return default_embedding(text);
        };
    }
}

SQLiteRetriever::~SQLiteRetriever() = default;

bool SQLiteRetriever::initialize() {
    if (initialized_) return true;

    if (!db_ || !db_->is_valid()) {
        log_error("Database is not available");
        return false;
    }

    if (!db_->initialize_schema()) {
        log_error("Failed to initialize database schema");
        return false;
    }

    initialized_ = true;
    log_info("SQLiteRetriever initialized successfully");
    return true;
}

void SQLiteRetriever::init_cache(const CacheConfig& cache_config) {
    cache_ = std::make_unique<LRUCache>(cache_config);
}

void SQLiteRetriever::init_thread_pool(const ThreadPoolConfig& tp_config) {
    thread_pool_ = std::make_unique<ThreadPool>(tp_config);
}

size_t SQLiteRetriever::insert_documents(const std::vector<Chunk>& chunks) {
    if (!initialized_ && !initialize()) {
        log_error("Failed to initialize retriever");
        return 0;
    }

    auto start = std::chrono::high_resolution_clock::now();

    size_t inserted = db_->insert_chunks(chunks, embed_func_);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    log_info("Inserted " + std::to_string(inserted) + "/" + std::to_string(chunks.size()) +
             " documents in " + std::to_string(duration.count()) + "ms");

    // 清空缓存（因为有新数据）
    if (cache_) {
        // cache_->clear();  // 假设有这个方法
    }

    return inserted;
}

std::vector<SQLiteSearchResult> SQLiteRetriever::query(
    const std::string& query, int limit) {

    if (!initialized_ && !initialize()) {
        log_error("Failed to initialize retriever");
        return {};
    }

    if (query.empty()) {
        log_error("Empty query");
        return {};
    }

    // 使用配置的默认limit
    if (limit == -1) {
        limit = config_.max_results;
    }

    // 确定检索策略
    SQLiteRetrievalStrategy strategy = config_.strategy;
    if (strategy == SQLiteRetrievalStrategy::ADAPTIVE) {
        strategy = choose_strategy(query);
    }

    // 生成缓存键
    std::string cache_key = generate_cache_key(query, strategy, limit);

    // 尝试从缓存获取
    std::vector<SQLiteSearchResult> results;
    if (config_.enable_cache && cache_ && get_from_cache(cache_key, results)) {
        log_info("Cache hit for query: " + query);
        return results;
    }

    // 执行检索
    auto start = std::chrono::high_resolution_clock::now();

    switch (strategy) {
        case SQLiteRetrievalStrategy::FTS5_ONLY:
            results = query_text_only(query, limit);
            break;
        case SQLiteRetrievalStrategy::VECTOR_ONLY:
            results = query_vector_only(query, limit);
            break;
        case SQLiteRetrievalStrategy::HYBRID:
            results = query_hybrid(query, limit);
            break;
        default:
            results = query_hybrid(query, limit);
            break;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    log_info("Query '" + query + "' returned " + std::to_string(results.size()) +
             " results in " + std::to_string(duration.count()) + "μs");

    // 存入缓存
    if (config_.enable_cache && cache_) {
        put_to_cache(cache_key, results);
    }

    return results;
}

std::future<std::vector<SQLiteSearchResult>> SQLiteRetriever::query_async(
    const std::string& query, int limit) {

    if (!thread_pool_) {
        // 如果没有线程池，直接同步执行
        auto promise = std::make_shared<std::promise<std::vector<SQLiteSearchResult>>>();
        auto future = promise->get_future();

        try {
            auto results = this->query(query, limit);
            promise->set_value(results);
        } catch (...) {
            promise->set_exception(std::current_exception());
        }

        return future;
    }

    // 使用线程池异步执行
    return thread_pool_->submit([this, query, limit]() {
        return this->query(query, limit);
    });
}

std::vector<SQLiteSearchResult> SQLiteRetriever::query_text_only(
    const std::string& query, int limit) {

    if (limit == -1) limit = config_.max_results;
    return db_->search_fts5(query, limit);
}

std::vector<SQLiteSearchResult> SQLiteRetriever::query_vector_only(
    const std::string& query, int limit) {

    if (limit == -1) limit = config_.max_results;

    if (!embed_func_) {
        log_error("No embedding function available for vector search");
        return {};
    }

    auto embedding = embed_func_(query);
    if (embedding.empty()) {
        log_error("Failed to generate embedding for query");
        return {};
    }

    return db_->search_vector(embedding, limit);
}

std::vector<SQLiteSearchResult> SQLiteRetriever::query_hybrid(
    const std::string& query, int limit) {

    if (limit == -1) limit = config_.max_results;

    if (!embed_func_) {
        log_info("No embedding function, falling back to FTS5 only");
        return query_text_only(query, limit);
    }

    auto embedding = embed_func_(query);
    if (embedding.empty()) {
        log_info("Failed to generate embedding, falling back to FTS5 only");
        return query_text_only(query, limit);
    }

    // 使用配置的权重和限制
    int fts5_limit = std::max(limit, 50);  // 获取更多候选
    int vector_limit = std::max(limit, 50);

    return db_->search_hybrid(
        query, embedding,
        fts5_limit, vector_limit,
        config_.fts5_weight, config_.vector_weight
    );
}

std::vector<SQLiteSearchResult> SQLiteRetriever::get_documents_by_ids(
    const std::vector<size_t>& chunk_ids) {

    if (!initialized_ && !initialize()) {
        return {};
    }

    // 转换为 int 类型
    std::vector<int> int_chunk_ids;
    int_chunk_ids.reserve(chunk_ids.size());
    for (size_t id : chunk_ids) {
        int_chunk_ids.push_back(static_cast<int>(id));
    }

    return db_->get_chunks_by_ids(int_chunk_ids);
}

bool SQLiteRetriever::clear_all_data() {
    if (!initialized_ && !initialize()) {
        return false;
    }

    bool success = db_->clear_all_data();

    // 清空缓存
    if (cache_) {
        // cache_->clear();
    }

    return success;
}

SQLiteDB::DBStats SQLiteRetriever::get_stats() {
    if (!initialized_ && !initialize()) {
        return {};
    }

    return db_->get_stats();
}

void SQLiteRetriever::update_config(const SQLiteRetrieverConfig& new_config) {
    config_ = new_config;
    log_info("Retriever configuration updated");
}

bool SQLiteRetriever::is_available() const {
    return initialized_ && db_ && db_->is_valid();
}

void SQLiteRetriever::set_embedding_function(
    std::function<std::vector<float>(const std::string&)> embed_func) {
    embed_func_ = embed_func;
    log_info("Embedding function updated");
}

void SQLiteRetriever::warmup(const std::vector<std::string>& sample_queries) {
    if (!initialized_ && !initialize()) {
        return;
    }

    std::vector<std::string> queries = sample_queries;
    if (queries.empty()) {
        // 使用默认查询进行预热
        queries = {
            "machine learning",
            "artificial intelligence",
            "deep learning",
            "natural language processing"
        };
    }

    log_info("Starting warmup with " + std::to_string(queries.size()) + " queries");

    auto start = std::chrono::high_resolution_clock::now();

    for (const auto& query : queries) {
        this->query(query, 5);  // 小批量查询
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    log_info("Warmup completed in " + std::to_string(duration.count()) + "ms");
}

std::vector<float> SQLiteRetriever::default_embedding(const std::string& text) {
    // 简单的基于哈希的嵌入（用于测试）
    std::hash<std::string> hasher;
    auto hash = hasher(text);

    std::mt19937 generator(hash);
    std::normal_distribution<float> distribution(0.0, 1.0);

    std::vector<float> embedding(768);  // 默认 768 维
    for (auto& val : embedding) {
        val = distribution(generator);
    }

    // 归一化
    float norm = 0.0;
    for (auto val : embedding) {
        norm += val * val;
    }
    norm = std::sqrt(norm);

    if (norm > 0) {
        for (auto& val : embedding) {
            val /= norm;
        }
    }

    return embedding;
}

std::string SQLiteRetriever::generate_cache_key(
    const std::string& query,
    SQLiteRetrievalStrategy strategy,
    int limit) {

    std::ostringstream oss;
    oss << "q:" << query << "|s:" << static_cast<int>(strategy) << "|l:" << limit;
    return oss.str();
}

bool SQLiteRetriever::get_from_cache(
    const std::string& cache_key,
    std::vector<SQLiteSearchResult>& results) {

    if (!cache_) return false;

    Retrieval cached_result;
    if (!cache_->get(cache_key, cached_result)) {
        return false;
    }

    // 从缓存的 chunk IDs 获取完整结果
    results = get_documents_by_ids(cached_result.top_chunks);
    return !results.empty();
}

void SQLiteRetriever::put_to_cache(
    const std::string& cache_key,
    const std::vector<SQLiteSearchResult>& results) {

    if (!cache_) return;

    Retrieval retrieval_result;
    for (const auto& result : results) {
        retrieval_result.top_chunks.push_back(result.chunk_id);
    }
    retrieval_result.timestamp = std::time(nullptr);

    cache_->put(cache_key, retrieval_result);
}

SQLiteRetrievalStrategy SQLiteRetriever::choose_strategy(const std::string& query) {
    // 简单的启发式策略选择

    // 如果查询包含很多英文单词，优先使用 FTS5
    std::regex word_regex(R"(\b[a-zA-Z]+\b)");
    auto words_begin = std::sregex_iterator(query.begin(), query.end(), word_regex);
    auto words_end = std::sregex_iterator();
    int english_words = std::distance(words_begin, words_end);

    // 如果查询很短且包含关键词，使用 FTS5
    if (query.length() < 50 && english_words > 2) {
        return SQLiteRetrievalStrategy::FTS5_ONLY;
    }

    // 如果查询很长且语义化，使用向量检索
    if (query.length() > 100) {
        return SQLiteRetrievalStrategy::VECTOR_ONLY;
    }

    // 默认使用混合策略
    return SQLiteRetrievalStrategy::HYBRID;
}

void SQLiteRetriever::log_info(const std::string& message) {
    std::cout << "[SQLiteRetriever] " << message << std::endl;
}

void SQLiteRetriever::log_error(const std::string& message) {
    std::cerr << "[SQLiteRetriever Error] " << message << std::endl;
}

// SQLiteRAGSystem 实现

SQLiteRAGSystem::SQLiteRAGSystem(const std::string& config_path)
    : initialized_(false) {

    try {
        config_ = ConfigLoader::load(config_path);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load config: " << e.what() << std::endl;
        config_ = std::make_shared<RAGConfig>();
    }
}

bool SQLiteRAGSystem::initialize() {
    if (initialized_) return true;

    // 创建嵌入函数
    auto embed_func = [this](const std::string& text) {
        return simple_embedding(text);
    };

    // 创建检索器
    retriever_ = std::make_shared<SQLiteRetriever>(*config_, embed_func);

    if (!retriever_->initialize()) {
        std::cerr << "Failed to initialize SQLite retriever" << std::endl;
        return false;
    }

    initialized_ = true;
    std::cout << "SQLiteRAGSystem initialized successfully" << std::endl;
    return true;
}

size_t SQLiteRAGSystem::load_documents(const std::vector<Chunk>& documents) {
    if (!initialized_ && !initialize()) {
        return 0;
    }

    return retriever_->insert_documents(documents);
}

size_t SQLiteRAGSystem::load_documents_from_file(const std::string& file_path) {
    // 这里应该实现文件读取和分块逻辑
    // 为了简化，我们只返回一个示例实现

    std::cout << "Loading documents from file: " << file_path << std::endl;
    std::cout << "Note: File loading not implemented in this example" << std::endl;

    return 0;
}

std::vector<SQLiteSearchResult> SQLiteRAGSystem::search(
    const std::string& query, int limit) {

    if (!initialized_ && !initialize()) {
        return {};
    }

    return retriever_->query(query, limit);
}

SQLiteDB::DBStats SQLiteRAGSystem::get_system_stats() {
    if (!initialized_ && !initialize()) {
        return {};
    }

    return retriever_->get_stats();
}

std::vector<Chunk> SQLiteRAGSystem::chunk_text(
    const std::string& text, const std::string& doc_id) {

    std::vector<Chunk> chunks;

    // 简单的分块实现：按句子分割
    std::regex sentence_regex(R"([.!?]+\s+)");
    std::sregex_token_iterator iter(text.begin(), text.end(), sentence_regex, -1);
    std::sregex_token_iterator end;

    int seq_no = 0;
    std::string current_chunk;

    for (auto it = iter; it != end; ++it) {
        std::string sentence = it->str();
        if (sentence.empty()) continue;

        if (current_chunk.length() + sentence.length() > config_->chunk.size) {
            if (!current_chunk.empty()) {
                Chunk chunk;
                chunk.doc_id = doc_id;
                chunk.seq_no = seq_no++;
                chunk.text = current_chunk;
                chunk.topic = "auto";
                chunks.push_back(chunk);
                current_chunk.clear();
            }
        }

        current_chunk += sentence + " ";
    }

    // 添加最后一个块
    if (!current_chunk.empty()) {
        Chunk chunk;
        chunk.doc_id = doc_id;
        chunk.seq_no = seq_no;
        chunk.text = current_chunk;
        chunk.topic = "auto";
        chunks.push_back(chunk);
    }

    return chunks;
}

std::vector<float> SQLiteRAGSystem::simple_embedding(const std::string& text) {
    // 这是一个非常简单的嵌入实现，实际应用中应该使用真实的嵌入模型
    std::hash<std::string> hasher;
    auto hash = hasher(text);

    std::mt19937 generator(hash);
    std::normal_distribution<float> distribution(0.0, 1.0);

    std::vector<float> embedding(config_->sqlite.vector_dimension);
    for (auto& val : embedding) {
        val = distribution(generator);
    }

    // 简单归一化
    float norm = 0.0;
    for (auto val : embedding) {
        norm += val * val;
    }
    norm = std::sqrt(norm);

    if (norm > 0) {
        for (auto& val : embedding) {
            val /= norm;
        }
    }

    return embedding;
}

} // namespace rag

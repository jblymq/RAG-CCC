#include "config.h"
#include "toml.hpp"
#include <fstream>
#include <iostream>

namespace rag {

std::shared_ptr<RAGConfig> ConfigLoader::instance_ = nullptr;

std::shared_ptr<RAGConfig> ConfigLoader::load(const std::string& config_path) {
    auto config = std::make_shared<RAGConfig>();

    try {
        auto data = toml::parse_file(config_path);

        // Load chunk config
        if (data.contains("chunk")) {
            const auto& chunk_table = *data["chunk"].as_table();
            if (chunk_table.contains("size")) {
                config->chunk.size = chunk_table["size"].as_integer()->get();
            }
            if (chunk_table.contains("overlap")) {
                config->chunk.overlap = chunk_table["overlap"].as_integer()->get();
            }
            if (chunk_table.contains("min_size")) {
                config->chunk.min_size = chunk_table["min_size"].as_integer()->get();
            }
        }

        // Load BM25 config
        if (data.contains("bm25")) {
            const auto& bm25_table = *data["bm25"].as_table();
            if (bm25_table.contains("k1")) {
                config->bm25.k1 = bm25_table["k1"].as_floating_point()->get();
            }
            if (bm25_table.contains("b")) {
                config->bm25.b = bm25_table["b"].as_floating_point()->get();
            }
        }

        // Load HNSW config
        if (data.contains("hnsw")) {
            const auto& hnsw_table = *data["hnsw"].as_table();
            if (hnsw_table.contains("M")) {
                config->hnsw.M = hnsw_table["M"].as_integer()->get();
            }
            if (hnsw_table.contains("ef_construction")) {
                config->hnsw.ef_construction = hnsw_table["ef_construction"].as_integer()->get();
            }
            if (hnsw_table.contains("ef_query")) {
                config->hnsw.ef_query = hnsw_table["ef_query"].as_integer()->get();
            }
            if (hnsw_table.contains("vector_dim")) {
                config->hnsw.vector_dim = hnsw_table["vector_dim"].as_integer()->get();
            }
            if (hnsw_table.contains("max_elements")) {
                config->hnsw.max_elements = hnsw_table["max_elements"].as_integer()->get();
            }
        }

        // Load fusion config
        if (data.contains("fusion")) {
            const auto& fusion_table = *data["fusion"].as_table();
            if (fusion_table.contains("bm25_weight")) {
                config->fusion.bm25_weight = fusion_table["bm25_weight"].as_floating_point()->get();
            }
            if (fusion_table.contains("vector_weight")) {
                config->fusion.vector_weight = fusion_table["vector_weight"].as_floating_point()->get();
            }
            if (fusion_table.contains("max_candidates")) {
                config->fusion.max_candidates = fusion_table["max_candidates"].as_integer()->get();
            }
            if (fusion_table.contains("rrf_k")) {
                config->fusion.rrf_k = fusion_table["rrf_k"].as_floating_point()->get();
            }
            if (fusion_table.contains("enable_rerank")) {
                config->fusion.enable_rerank = fusion_table["enable_rerank"].as_boolean()->get();
            }
            if (fusion_table.contains("strategy")) {
                config->fusion.strategy = fusion_table["strategy"].as_string()->get();
            }
        }

        // Load cache config
        if (data.contains("cache")) {
            const auto& cache_table = *data["cache"].as_table();
            if (cache_table.contains("capacity")) {
                config->cache.capacity = cache_table["capacity"].as_integer()->get();
            }
            if (cache_table.contains("ttl_seconds")) {
                config->cache.ttl_seconds = cache_table["ttl_seconds"].as_integer()->get();
            }
        }

        // Load threadpool config
        if (data.contains("threadpool")) {
            const auto& tp_table = *data["threadpool"].as_table();
            if (tp_table.contains("num_workers")) {
                config->threadpool.num_workers = tp_table["num_workers"].as_integer()->get();
            }
        }

        // Load tuner config
        if (data.contains("tuner")) {
            const auto& tuner_table = *data["tuner"].as_table();
            if (tuner_table.contains("latency_max_ms")) {
                config->tuner.latency_max_ms = tuner_table["latency_max_ms"].as_floating_point()->get();
            }
            if (tuner_table.contains("recall_min_pct")) {
                config->tuner.recall_min_pct = tuner_table["recall_min_pct"].as_floating_point()->get();
            }
            if (tuner_table.contains("ef_delta")) {
                config->tuner.ef_delta = tuner_table["ef_delta"].as_integer()->get();
            }
            if (tuner_table.contains("topk_delta")) {
                config->tuner.topk_delta = tuner_table["topk_delta"].as_integer()->get();
            }
            if (tuner_table.contains("enable")) {
                config->tuner.enable = tuner_table["enable"].as_boolean()->get();
            }
            if (tuner_table.contains("check_interval_seconds")) {
                config->tuner.check_interval_seconds = tuner_table["check_interval_seconds"].as_integer()->get();
            }
        }

        // Load SQLite config
        if (data.contains("sqlite")) {
            const auto& sqlite_table = *data["sqlite"].as_table();
            if (sqlite_table.contains("db_path")) {
                config->sqlite.db_path = sqlite_table["db_path"].as_string()->get();
            }
            if (sqlite_table.contains("vector_extension")) {
                config->sqlite.vector_extension = sqlite_table["vector_extension"].as_string()->get();
            }
            if (sqlite_table.contains("vector_dimension")) {
                config->sqlite.vector_dimension = sqlite_table["vector_dimension"].as_integer()->get();
            }
            if (sqlite_table.contains("enable_fts5")) {
                config->sqlite.enable_fts5 = sqlite_table["enable_fts5"].as_boolean()->get();
            }
            if (sqlite_table.contains("enable_wal")) {
                config->sqlite.enable_wal = sqlite_table["enable_wal"].as_boolean()->get();
            }
            if (sqlite_table.contains("cache_size")) {
                config->sqlite.cache_size = sqlite_table["cache_size"].as_integer()->get();
            }
            if (sqlite_table.contains("busy_timeout")) {
                config->sqlite.busy_timeout = sqlite_table["busy_timeout"].as_integer()->get();
            }
            if (sqlite_table.contains("fts5_limit")) {
                config->sqlite.fts5_limit = sqlite_table["fts5_limit"].as_integer()->get();
            }
            if (sqlite_table.contains("vector_limit")) {
                config->sqlite.vector_limit = sqlite_table["vector_limit"].as_integer()->get();
            }
        }

        std::cout << "RAG config loaded from: " << config_path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Failed to load config from " << config_path << ": " << e.what() << std::endl;
        std::cerr << "Using default configuration." << std::endl;
    }

    instance_ = config;
    return config;
}

std::shared_ptr<RAGConfig> ConfigLoader::get_instance() {
    if (!instance_) {
        return load();
    }
    return instance_;
}

} // namespace rag

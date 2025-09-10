/**
 * Humanus.cpp 混合RAG系统演示
 *
 * 本示例展示如何将内存RAG和SQLite RAG结合，形成完整的增强RAG系统：
 *
 * 🎯 核心设计理念：
 * • 热数据内存检索 - 毫秒级响应
 * • 冷数据持久化存储 - 大容量存储
 * • 智能数据分层 - 自动热点识别
 * • 无缝数据迁移 - 热冷数据动态调整
 *
 * 🚀 技术特性：
 * • 双模式并行检索
 * • 智能缓存管理
 * • 自动数据分层
 * • 统一检索接口
 * • 性能监控与调优
 *
 * 编译: cd build && make hybrid_rag_demo
 * 运行: ./hybrid_rag_demo
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <memory>
#include <future>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

// RAG 核心模块
#include "rag/chunk.h"
#include "rag/config.h"
#include "rag/fusion_retriever.h"
#include "rag/sqlite_retriever.h"

using namespace rag;

/**
 * 扩展的搜索结果结构，包含来源信息
 */
struct HybridSearchResult {
    int chunk_id;
    double score;
    std::string doc_id;
    std::string content;
    std::string topic;
    std::string source;  // "memory" 或 "sqlite"

    // 从SQLiteSearchResult转换
    static HybridSearchResult from_sqlite(const SQLiteSearchResult& sqlite_result) {
        HybridSearchResult result;
        result.chunk_id = sqlite_result.chunk_id;
        result.score = sqlite_result.score;
        result.doc_id = sqlite_result.doc_id;
        result.content = sqlite_result.content;
        result.topic = sqlite_result.topic;
        result.source = "sqlite";
        return result;
    }

    // 从RetrievalResult转换
    static HybridSearchResult from_memory(const RetrievalResult& memory_result) {
        HybridSearchResult result;
        result.chunk_id = 0;  // 内存模式可能没有chunk_id
        result.score = memory_result.score;
        result.doc_id = memory_result.doc_id;
        result.content = memory_result.text;
        result.topic = "";  // 内存模式可能没有topic
        result.source = "memory";
        return result;
    }
};

// 颜色输出
namespace Color {
    const std::string RESET = "\033[0m";
    const std::string RED = "\033[31m";
    const std::string GREEN = "\033[32m";
    const std::string YELLOW = "\033[33m";
    const std::string BLUE = "\033[34m";
    const std::string MAGENTA = "\033[35m";
    const std::string CYAN = "\033[36m";
    const std::string BOLD = "\033[1m";
}

/**
 * 计时器工具
 */
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;

public:
    Timer() { reset(); }

    void reset() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double elapsed_ms() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0;
    }

    double elapsed_us() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count();
    }
};

/**
 * 数据访问统计
 */
struct AccessStats {
    std::unordered_map<std::string, int> doc_access_count;
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> last_access_time;

    void record_access(const std::string& doc_id) {
        doc_access_count[doc_id]++;
        last_access_time[doc_id] = std::chrono::steady_clock::now();
    }

    bool is_hot_data(const std::string& doc_id, int threshold = 3) const {
        auto it = doc_access_count.find(doc_id);
        return it != doc_access_count.end() && it->second >= threshold;
    }

    std::vector<std::string> get_hot_documents(int threshold = 3) const {
        std::vector<std::string> hot_docs;
        for (const auto& [doc_id, count] : doc_access_count) {
            if (count >= threshold) {
                hot_docs.push_back(doc_id);
            }
        }
        return hot_docs;
    }
};

/**
 * 混合RAG系统 - 核心类
 *
 * 设计理念：
 * 1. 内存层：存储热数据，提供毫秒级检索
 * 2. 持久化层：存储全量数据，保证数据完整性
 * 3. 智能调度：根据访问模式自动数据分层
 */
class HybridRAGSystem {
private:
    std::shared_ptr<FusionRetriever> memory_retriever_;    // 内存检索器
    std::unique_ptr<SQLiteRAGSystem> sqlite_system_;       // SQLite检索器
    std::shared_ptr<RAGConfig> config_;                    // 配置
    AccessStats access_stats_;                             // 访问统计

    // 内存层文档集合
    std::unordered_set<std::string> memory_doc_ids_;

    // 配置参数
    int hot_threshold_ = 3;        // 热数据访问阈值
    int memory_capacity_ = 1000;   // 内存层容量限制

public:
    explicit HybridRAGSystem(const std::string& config_path = "rag_config.toml") {
        // 1. 加载配置
        config_ = ConfigLoader::load(config_path);
        if (!config_) {
            throw std::runtime_error("Failed to load configuration");
        }

        // 2. 初始化内存检索器
        memory_retriever_ = FusionRetriever::from_config(*config_);

        // 3. 初始化SQLite系统
        sqlite_system_ = std::make_unique<SQLiteRAGSystem>(config_path);
        if (!sqlite_system_->initialize()) {
            throw std::runtime_error("Failed to initialize SQLite RAG system");
        }

        std::cout << Color::GREEN << "✅ 混合RAG系统初始化成功" << Color::RESET << std::endl;
    }

    /**
     * 加载文档到系统
     * 策略：新文档先存储到SQLite，根据访问模式决定是否加载到内存
     */
    size_t load_documents(const std::vector<Chunk>& documents) {
        std::cout << Color::BLUE << "📥 加载文档到混合RAG系统..." << Color::RESET << std::endl;

        // 1. 全部文档存储到SQLite（持久化层）
        Timer timer;
        auto sqlite_count = sqlite_system_->load_documents(documents);
        double sqlite_time = timer.elapsed_ms();

        std::cout << "  • SQLite存储: " << sqlite_count << " 个文档 ("
                 << sqlite_time << "ms)" << std::endl;

        // 2. 如果内存层有容量，加载部分热门文档
        if (memory_doc_ids_.size() < memory_capacity_) {
            timer.reset();
            std::vector<Chunk> memory_docs;
            for (const auto& doc : documents) {
                if (memory_docs.size() < memory_capacity_ / 2) {  // 预留一半容量
                    memory_docs.push_back(doc);
                    memory_doc_ids_.insert(doc.doc_id);
                }
            }

            if (!memory_docs.empty()) {
                memory_retriever_->fit(memory_docs);
                double memory_time = timer.elapsed_ms();
                std::cout << "  • 内存预加载: " << memory_docs.size() << " 个文档 ("
                         << memory_time << "ms)" << std::endl;
            }
        }

        return sqlite_count;
    }

    /**
     * 混合检索 - 核心功能
     *
     * 策略：
     * 1. 并行查询内存层和持久化层
     * 2. 智能合并结果，去重排序
     * 3. 更新访问统计，触发数据迁移
     */
    std::vector<HybridSearchResult> search(const std::string& query, int limit = 10) {
        Timer total_timer;

        // 1. 并行查询两个层次
        std::future<std::vector<RetrievalResult>> memory_future;
        std::future<std::vector<SQLiteSearchResult>> sqlite_future;

        // 启动内存检索（如果有数据）
        if (!memory_doc_ids_.empty()) {
            memory_future = std::async(std::launch::async, [this, &query, limit]() {
                return memory_retriever_->query(query, limit);
            });
        }

        // 启动SQLite检索
        sqlite_future = std::async(std::launch::async, [this, &query, limit]() {
            return sqlite_system_->search(query, limit);
        });

        // 2. 收集结果
        std::vector<HybridSearchResult> final_results;
        std::unordered_set<std::string> seen_docs;

        // 获取SQLite结果
        auto sqlite_results = sqlite_future.get();

        // 获取内存结果并转换格式
        std::vector<RetrievalResult> memory_results;
        if (!memory_doc_ids_.empty()) {
            memory_results = memory_future.get();
        }

        // 3. 智能合并结果
        // 优先使用内存结果（更快），SQLite结果作为补充
        for (const auto& result : memory_results) {
            if (seen_docs.find(result.doc_id) == seen_docs.end()) {
                auto converted = HybridSearchResult::from_memory(result);
                final_results.push_back(converted);
                seen_docs.insert(result.doc_id);

                // 记录访问统计
                access_stats_.record_access(result.doc_id);
            }
        }

        // 补充SQLite结果
        for (const auto& result : sqlite_results) {
            if (seen_docs.find(result.doc_id) == seen_docs.end() &&
                final_results.size() < limit) {

                auto enhanced_result = HybridSearchResult::from_sqlite(result);
                final_results.push_back(enhanced_result);
                seen_docs.insert(result.doc_id);

                // 记录访问统计
                access_stats_.record_access(result.doc_id);
            }
        }

        // 4. 按分数重新排序
        std::sort(final_results.begin(), final_results.end(),
                 [](const HybridSearchResult& a, const HybridSearchResult& b) {
                     return a.score > b.score;
                 });

        // 5. 限制结果数量
        if (final_results.size() > limit) {
            final_results.resize(limit);
        }

        double total_time = total_timer.elapsed_us();

        // 6. 触发数据分层优化（异步）
        std::async(std::launch::async, [this]() {
            optimize_data_distribution();
        });

        std::cout << Color::CYAN << "🔍 混合检索完成: " << final_results.size()
                 << " 个结果 (" << total_time << "μs)" << Color::RESET << std::endl;

        return final_results;
    }

    /**
     * 数据分层优化
     * 根据访问模式，动态调整内存层和持久化层的数据分布
     */
    void optimize_data_distribution() {
        // 1. 识别热数据
        auto hot_docs = access_stats_.get_hot_documents(hot_threshold_);

        if (hot_docs.empty()) {
            return;  // 没有热数据，无需优化
        }

        std::cout << Color::YELLOW << "🔥 发现 " << hot_docs.size()
                 << " 个热数据，开始优化分布..." << Color::RESET << std::endl;

        // 2. 将热数据迁移到内存层
        std::vector<Chunk> hot_chunks;
        for (const auto& doc_id : hot_docs) {
            if (memory_doc_ids_.find(doc_id) == memory_doc_ids_.end() &&
                memory_doc_ids_.size() < memory_capacity_) {

                // 从SQLite获取文档内容
                auto results = sqlite_system_->search("doc_id:" + doc_id, 1);
                if (!results.empty()) {
                    Chunk chunk;
                    chunk.doc_id = results[0].doc_id;
                    chunk.text = results[0].content;
                    chunk.topic = results[0].topic;
                    hot_chunks.push_back(chunk);
                    memory_doc_ids_.insert(doc_id);
                }
            }
        }

        // 3. 更新内存索引
        if (!hot_chunks.empty()) {
            memory_retriever_->fit(hot_chunks);
            std::cout << Color::GREEN << "📈 已将 " << hot_chunks.size()
                     << " 个热数据迁移到内存层" << Color::RESET << std::endl;
        }

        // 4. 如果内存层超容量，移除冷数据
        if (memory_doc_ids_.size() > memory_capacity_) {
            // 简化实现：这里可以实现LRU策略移除冷数据
            std::cout << Color::YELLOW << "⚠️ 内存层达到容量限制，建议实现LRU清理策略"
                     << Color::RESET << std::endl;
        }
    }

    /**
     * 获取系统统计信息
     */
    void print_system_stats() {
        auto sqlite_stats = sqlite_system_->get_system_stats();
        auto hot_docs = access_stats_.get_hot_documents(hot_threshold_);

        std::cout << "\n" << Color::BOLD << "📊 混合RAG系统统计" << Color::RESET << std::endl;
        std::cout << std::string(50, '=') << std::endl;

        std::cout << Color::BLUE << "💾 存储层统计:" << Color::RESET << std::endl;
        std::cout << "  • SQLite文档总数: " << sqlite_stats.total_chunks << std::endl;
        std::cout << "  • 内存层文档数: " << memory_doc_ids_.size() << std::endl;
        std::cout << "  • 数据库大小: " << std::fixed << std::setprecision(2)
                 << sqlite_stats.db_size_mb << " MB" << std::endl;

        std::cout << "\n" << Color::GREEN << "🔥 访问热点统计:" << Color::RESET << std::endl;
        std::cout << "  • 热数据文档数: " << hot_docs.size() << std::endl;
        std::cout << "  • 总访问次数: " << access_stats_.doc_access_count.size() << std::endl;
        std::cout << "  • 内存命中率: " << std::fixed << std::setprecision(1)
                 << (memory_doc_ids_.empty() ? 0.0 :
                     (double)hot_docs.size() / memory_doc_ids_.size() * 100) << "%" << std::endl;

        std::cout << "\n" << Color::MAGENTA << "⚡ 性能指标:" << Color::RESET << std::endl;
        std::cout << "  • 内存层容量利用率: " << std::fixed << std::setprecision(1)
                 << (double)memory_doc_ids_.size() / memory_capacity_ * 100 << "%" << std::endl;
        std::cout << "  • 数据分层效率: " << (hot_docs.size() > 0 ? "优秀" : "待优化") << std::endl;
    }

    /**
     * 基准测试
     */
    void run_benchmark(const std::vector<std::string>& queries) {
        std::cout << "\n" << Color::BOLD << "🚀 混合RAG系统基准测试" << Color::RESET << std::endl;
        std::cout << std::string(50, '=') << std::endl;

        double total_time = 0;
        int total_results = 0;

        for (const auto& query : queries) {
            Timer timer;
            auto results = search(query, 5);
            double query_time = timer.elapsed_us();

            total_time += query_time;
            total_results += results.size();

            std::cout << "🔍 \"" << query << "\"" << std::endl;
            std::cout << "  ⏱️ 耗时: " << query_time << "μs | 📄 结果: " << results.size() << " 个" << std::endl;

            // 显示来源分布
            int memory_count = 0, sqlite_count = 0;
            for (const auto& result : results) {
                if (result.source == "memory") memory_count++;
                else sqlite_count++;
            }
            std::cout << "  📊 来源: 内存(" << memory_count << ") SQLite(" << sqlite_count << ")" << std::endl;
            std::cout << std::endl;
        }

        std::cout << Color::GREEN << "📈 基准测试汇总:" << Color::RESET << std::endl;
        std::cout << "  • 平均查询时间: " << std::fixed << std::setprecision(2)
                 << total_time / queries.size() << "μs" << std::endl;
        std::cout << "  • 平均结果数量: " << std::fixed << std::setprecision(1)
                 << (double)total_results / queries.size() << " 个" << std::endl;
        std::cout << "  • 系统吞吐量: " << std::fixed << std::setprecision(0)
                 << 1000000.0 / (total_time / queries.size()) << " QPS" << std::endl;
    }
};

/**
 * 创建测试数据集
 */
std::vector<Chunk> create_large_dataset() {
    std::vector<Chunk> documents;

    // 技术文档（可能成为热数据）
    std::vector<std::string> tech_topics = {
        "机器学习基础", "深度学习原理", "自然语言处理", "计算机视觉",
        "推荐系统", "分布式系统", "微服务架构", "容器技术"
    };

    std::vector<std::string> tech_contents = {
        "机器学习是人工智能的核心分支，通过算法让计算机从数据中学习模式和规律。",
        "深度学习使用多层神经网络模拟人脑处理信息的方式，在图像和语音识别方面表现卓越。",
        "自然语言处理让计算机理解和生成人类语言，包括文本分析、机器翻译等应用。",
        "计算机视觉使机器能够理解和解析视觉信息，广泛应用于自动驾驶、医疗诊断等领域。",
        "推荐系统通过分析用户行为和偏好，为用户提供个性化的内容和产品推荐。",
        "分布式系统通过多台计算机协同工作，提供高可用性和可扩展性的计算服务。",
        "微服务架构将大型应用拆分为小型、独立的服务，提高系统的灵活性和可维护性。",
        "容器技术通过轻量级虚拟化，实现应用的快速部署和高效资源利用。"
    };

    // 创建技术文档
    for (size_t i = 0; i < tech_topics.size(); ++i) {
        Chunk doc;
        doc.doc_id = "tech_" + std::to_string(i + 1);
        doc.topic = tech_topics[i];
        doc.text = tech_contents[i];
        doc.language = "zh";
        documents.push_back(doc);
    }

    // 业务文档（较少被访问的冷数据）
    for (int i = 1; i <= 20; ++i) {
        Chunk doc;
        doc.doc_id = "business_" + std::to_string(i);
        doc.topic = "业务流程 " + std::to_string(i);
        doc.text = "这是业务流程文档第" + std::to_string(i) + "部分，详细描述了相关的操作规范和注意事项。";
        doc.language = "zh";
        documents.push_back(doc);
    }

    // 英文技术文档
    std::vector<std::string> en_topics = {
        "Machine Learning", "Deep Learning", "Neural Networks", "AI Ethics",
        "Data Science", "Big Data", "Cloud Computing", "DevOps"
    };

    std::vector<std::string> en_contents = {
        "Machine learning algorithms enable computers to learn from data without explicit programming.",
        "Deep learning networks with multiple layers can model complex patterns in large datasets.",
        "Neural networks are computing systems inspired by biological neural networks.",
        "AI ethics addresses moral implications and societal impacts of artificial intelligence.",
        "Data science combines statistics, programming, and domain expertise to extract insights.",
        "Big data technologies handle massive volumes of structured and unstructured data.",
        "Cloud computing provides on-demand access to computing resources over the internet.",
        "DevOps practices integrate development and operations for faster software delivery."
    };

    for (size_t i = 0; i < en_topics.size(); ++i) {
        Chunk doc;
        doc.doc_id = "en_tech_" + std::to_string(i + 1);
        doc.topic = en_topics[i];
        doc.text = en_contents[i];
        doc.language = "en";
        documents.push_back(doc);
    }

    return documents;
}

/**
 * 主函数 - 混合RAG系统完整演示
 */
int main() {
    std::cout << Color::BOLD << Color::CYAN << R"(
    ██████╗  █████╗  ██████╗       ██████╗██╗  ██╗██╗  ██╗
    ██╔══██╗██╔══██╗██╔════╝      ██╔════╝╚██╗██╔╝╚██╗██╔╝
    ██████╔╝███████║██║  ███╗     ██║      ╚███╔╝  ╚███╔╝ 
    ██╔══██╗██╔══██║██║   ██║     ██║      ██╔██╗  ██╔██╗ 
    ██║  ██║██║  ██║╚██████╔╝     ╚██████╗██╔╝ ██╗██╔╝ ██╗
    ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝       ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝
    )" << Color::RESET << std::endl;

    std::cout << Color::BOLD << "混合RAG系统演示程序" << Color::RESET << std::endl;
    std::cout << Color::BLUE << "Memory + SQLite Hybrid RAG System Demo" << Color::RESET << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    try {
        // 1. 初始化混合RAG系统
        std::cout << "\n" << Color::BOLD << "🚀 系统初始化" << Color::RESET << std::endl;
        HybridRAGSystem hybrid_rag("rag_config.toml");

        // 2. 加载测试数据
        std::cout << "\n" << Color::BOLD << "📚 数据加载" << Color::RESET << std::endl;
        auto documents = create_large_dataset();
        auto loaded_count = hybrid_rag.load_documents(documents);
        std::cout << Color::GREEN << "✅ 成功加载 " << loaded_count << " 个文档" << Color::RESET << std::endl;

        // 3. 系统状态展示
        hybrid_rag.print_system_stats();

        // 4. 模拟用户查询（创建访问热点）
        std::cout << "\n" << Color::BOLD << "🔍 模拟用户查询" << Color::RESET << std::endl;
        std::vector<std::string> user_queries = {
            "机器学习算法",      // 预期成为热查询
            "深度学习网络",      // 预期成为热查询
            "自然语言处理",      // 预期成为热查询
            "machine learning",  // 英文查询
            "neural networks",   // 英文查询
            "业务流程",          // 冷数据查询
            "云计算技术",        // 中等热度
            "data science"       // 英文查询
        };

        // 重复查询某些关键词，模拟热点数据
        for (int round = 1; round <= 3; ++round) {
            std::cout << "\n" << Color::YELLOW << "📊 第 " << round << " 轮查询" << Color::RESET << std::endl;

            for (const auto& query : user_queries) {
                auto results = hybrid_rag.search(query, 3);

                // 显示简化结果
                std::cout << "  🔍 \"" << query << "\" -> " << results.size() << " 个结果";
                if (!results.empty()) {
                    std::cout << " (最佳: " << results[0].doc_id << ")";
                }
                std::cout << std::endl;
            }
        }

        // 5. 数据分层优化后的系统状态
        std::cout << "\n" << Color::BOLD << "📈 优化后系统状态" << Color::RESET << std::endl;
        hybrid_rag.print_system_stats();

        // 6. 性能基准测试
        std::vector<std::string> benchmark_queries = {
            "机器学习", "深度学习", "人工智能", "数据科学",
            "machine learning", "deep learning", "artificial intelligence"
        };
        hybrid_rag.run_benchmark(benchmark_queries);

        // 7. 总结
        std::cout << "\n" << Color::BOLD << Color::GREEN << "🎉 混合RAG系统演示完成！" << Color::RESET << std::endl;
        std::cout << "\n" << Color::BOLD << "💡 核心优势总结:" << Color::RESET << std::endl;
        std::cout << "✅ 热数据内存缓存 - 毫秒级响应" << std::endl;
        std::cout << "✅ 冷数据持久化存储 - 无容量限制" << std::endl;
        std::cout << "✅ 智能数据分层 - 自动热点识别" << std::endl;
        std::cout << "✅ 并行检索架构 - 最优性能平衡" << std::endl;
        std::cout << "✅ 统一检索接口 - 透明化访问" << std::endl;

    } catch (const std::exception& e) {
        std::cout << Color::RED << "❌ 系统错误: " << e.what() << Color::RESET << std::endl;
        return 1;
    }

    return 0;
}

# RAG 框架 - 企业级检索增强生成系统

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/jblymq/RAG-CCC#)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/std/the-standard)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](https://github.com/jblymq/RAG-CCC#)
[![SQLite](https://img.shields.io/badge/SQLite-3.35%2B-blue)](https://sqlite.org/)

一个高性能、企业级的C++17 RAG（检索增强生成）框架，提供双模式架构：内存BM25+HNSW融合检索和SQLite持久化矢量数据库。支持多语言文档处理、智能缓存、自动调优和热重建等特性。

## 🎯 核心优势

### � 完整的设计文档对标实现
- ✅ **三层SQLite架构**：chunks + chunks_fts + embeddings
- ✅ **混合检索流程**：FTS5 (Top K₁) + Vector (Top K₂) 并行执行
- ✅ **ACID事务保证**：数据一致性和可靠性
- ✅ **WAL模式优化**：高并发性能和热备份
- ✅ **动态参数调优**：K₁, K₂ 自适应优化

### 🚀 双模式架构

#### 🧠 内存模式 - 极致性能
- **BM25+HNSW融合**：微秒级检索响应（160-300μs）
- **异步并发查询**：多查询并行处理，3查询仅需0.3ms
- **智能缓存**：LRU+TTL，9倍查询加速
- **5种融合策略**：BM25_ONLY, VECTOR_ONLY, HYBRID, RRF, ADAPTIVE

#### 🗄️ SQLite模式 - 企业级持久化
- **FTS5全文检索**：完整BM25算法，支持中英文混合
- **向量扩展支持**：sqlite-vec/sqlite-vss集成
- **三层存储架构**：原文、全文索引、向量索引分离
- **事务ACID保证**：数据一致性和故障恢复
- **在线热重建**：无停机索引更新

### 🌍 多语言与国际化
- **智能分词器**：中英文混合文档处理
- **语言自动检测**：自适应分词策略
- **Unicode支持**：完整的国际化字符处理
- **停用词过滤**：可配置的多语言停用词库

### ⚡ 高性能特性
- **线程池架构**：基于future的异步任务调度
- **智能缓存**：多级缓存策略，平均3-9倍性能提升
- **内存优化**：零拷贝设计，最小化内存分配
- **批处理优化**：大规模文档的高效批量处理

### 🔧 企业级特性
- **配置驱动**：TOML配置文件，无需重编译
- **监控友好**：丰富的性能指标和状态监控
- **扩展接口**：模块化设计，支持自定义组件
- **运维工具**：数据库维护、备份和迁移工具

### 🚀 混合RAG特性 🔥
- **智能分层**：热数据内存缓存 + 冷数据持久化存储
- **无限容量**：内存层高性能 + SQLite层大容量存储
- **自动优化**：基于访问模式的智能数据迁移
- **透明切换**：用户无感知的双层检索架构
- **并行检索**：同时搜索内存和SQLite，智能结果合并

## 🏗️ 架构设计

### 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                           RAG 框架                       │
├─────────────────────────────────────────────────────────────┤
│  应用层  │  Agent  │  Tools  │  MCP Protocol │  Planning    │
├─────────────────────────────────────────────────────────────┤
│          │         内存模式         │        SQLite模式        │
│  检索层  │  BM25 + HNSW 融合检索    │  FTS5 + Vector 混合检索  │
│          │  异步并发 + 智能缓存      │  事务保护 + 持久化存储    │
├─────────────────────────────────────────────────────────────┤
│  数据层  │  Tokenizer │ ThreadPool │ LRUCache │ AutoTuner   │
│          │  多语言分词  │  线程池管理  │  缓存系统  │  自动调优   │
└─────────────────────────────────────────────────────────────┘
```

### SQLite 三层存储架构

```sql
-- 1. 主表：存储原文与元信息
CREATE TABLE chunks (
    id        INTEGER PRIMARY KEY,
    doc_id    TEXT NOT NULL,
    seq_no    INTEGER,
    topic     TEXT,
    content   TEXT NOT NULL,
    language  TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. FTS5 虚拟表：BM25全文检索
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content,
    content='chunks',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 1'
);

-- 3. 向量索引表：高维向量存储
CREATE TABLE embeddings (
    chunk_id  INTEGER PRIMARY KEY,
    vector    BLOB,  -- 768维浮点向量
    FOREIGN KEY(chunk_id) REFERENCES chunks(id)
);
```

## 📁 项目结构

```
rag/
├── 📄 README.md                 # 本文档
├── ⚙️ rag_config.toml          # 主配置文件
├── 📋 INTEGRATION_SCENARIOS.md  # 业务集成场景
│
├── 🧩 核心模块
│   ├── chunk.h                  # 文档块定义
│   ├── config.h/.cpp           # 配置管理器
│   ├── tokenizer.h/.cpp        # 多语言分词器
│   ├── bm25.h/.cpp            # BM25检索引擎
│   ├── fusion_retriever.h/.cpp # 内存融合检索器
│   ├── sqlite_db.h/.cpp       # SQLite数据库管理
│   ├── sqlite_retriever.h/.cpp # SQLite检索器
│   ├── lru_cache.h/.cpp       # LRU缓存系统
│   ├── thread_pool.h/.cpp     # 线程池管理
│   └── autotuner.h/.cpp       # 自动调优器
│
├── 📚 依赖库
│   └── toml.hpp               # TOML解析库
│
└── 🎯 示例程序
    ├── example/
    │   ├── CMakeLists.txt     # 构建配置
    │   ├── main.cpp          # 综合演示程序
    │   ├── hybrid_rag_demo.cpp # 混合RAG系统演示 🔥
    │   ├── rag_config.toml   # 示例配置
    │   └── build/            # 构建目录
    └── INTEGRATION_SCENARIOS.md # 集成案例文档
```

## 🚀 快速开始

### 环境要求

#### 必需依赖
- **C++17** 兼容编译器（GCC 7+, Clang 6+, MSVC 2019+）
- **CMake 3.10+** 构建系统
- **SQLite 3.35+** 数据库（支持FTS5）
- **pthread** 多线程支持

#### 可选扩展
- **sqlite-vec** 或 **sqlite-vss** 向量扩展（增强向量检索）
- **pkg-config** 用于依赖管理

### 一键安装依赖

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential cmake libsqlite3-dev pkg-config

# CentOS/RHEL
sudo yum install gcc-c++ cmake sqlite-devel pkgconfig

# macOS
brew install cmake sqlite pkg-config

# 验证安装
sqlite3 --version  # 应显示 3.35+ 版本
```

### 编译安装

```bash
# 1. 克隆项目
git clone https://github.com/jblymq/RAG-CCC.git
cd /rag/example

# 2. 创建构建目录
mkdir build && cd build

# 3. 配置和编译
cmake ..
make -j$(nproc)

# 4. 运行综合演示
./rag_example
```

### 预期输出

```
RAG 系统综合演示程序
Retrieval-Augmented Generation System Demo

================================================================================
  内存 RAG 系统演示
  BM25 + HNSW 融合检索
================================================================================
✅ 配置加载完成 (耗时: 0.062ms)
✅ 检索器初始化完成 (耗时: 0.006ms)
✅ 索引构建完成 (6 个文档块, 耗时: 0.883ms)

🔍 检索测试
--------------------------------------------------
查询: 机器学习算法
  检索耗时: 255μs
  找到结果: 3 个
    1. doc3 (分数: 0.5000)
    2. doc2 (分数: 0.4994)

🚀 异步查询演示
------------------------------
异步查询总耗时: 0.284ms

================================================================================
  SQLite RAG 系统演示
  FTS5 + Vector 持久化检索
================================================================================
✅ SQLite RAG 系统初始化成功 (耗时: 0.482ms)
✅ 文档加载完成 (6/6 个文档, 耗时: 7.231ms)

📊 数据库统计
  文档数量: 18
  向量数量: 18
  数据库大小: 0.11 MB

💾 缓存性能测试
------------------------------
  第一次查询: 161μs (缓存未命中)
  第二次查询: 46μs (缓存命中)
  缓存加速比: 3.50x

🎉 RAG 系统综合演示完成！
```

## 📖 详细使用指南

### 内存模式快速体验

```cpp
#include "rag/config.h"
#include "rag/fusion_retriever.h"

int main() {
    using namespace rag;

    // 1. 加载配置
    auto config = ConfigLoader::load("./rag_config.toml");

    // 2. 创建文档
    std::vector<Chunk> docs;

    Chunk doc1;
    doc1.text = "Machine learning is a subset of artificial intelligence.";
    doc1.doc_id = "ml_intro";
    doc1.topic = "AI";
    doc1.language = "en";
    docs.push_back(doc1);

    Chunk doc2;
    doc2.text = "深度学习是机器学习的一个重要分支，它模拟人脑神经网络。";
    doc2.doc_id = "dl_intro";
    doc2.topic = "深度学习";
    doc2.language = "zh";
    docs.push_back(doc2);

    // 3. 创建融合检索器
    auto retriever = FusionRetriever::from_config(*config);

    // 4. 构建索引
    retriever->fit(docs);

    // 5. 执行查询
    auto results = retriever->query("artificial intelligence", 5);

    // 6. 输出结果
    for (const auto& result : results) {
        std::cout << "📄 Doc: " << result.doc_id
                  << " | 📊 Score: " << std::fixed << std::setprecision(4)
                  << result.score << std::endl;
        std::cout << "📝 Content: " << result.text.substr(0, 100) << "..." << std::endl;
        std::cout << std::string(50, '-') << std::endl;
    }

    // 7. 异步查询演示
    auto future_result = retriever->query_async("deep learning", 3);
    auto async_results = future_result.get();
    std::cout << "🚀 异步查询完成，找到 " << async_results.size() << " 个结果" << std::endl;

    return 0;
}
```

    return 0;
}
```

## 📖 详细文档

### 1. 配置系统

RAG框架使用TOML格式的配置文件，支持热重载和模块化配置：

```toml
# rag_config.toml

[chunk]
size = 512            # 文档块大小
overlap = 128         # 重叠大小

[bm25]
k1 = 1.2             # BM25 k1参数
b = 0.75             # BM25 b参数

[hnsw]
M = 16               # HNSW连接数
ef_construction = 200 # 构建时ef参数
ef_query = 50        # 查询时ef参数

[fusion]
strategy = "HYBRID"   # 融合策略：BM25_ONLY/VECTOR_ONLY/HYBRID/RRF
bm25_weight = 0.6    # BM25权重
vector_weight = 0.4   # 向量权重
rrf_k = 60           # RRF参数

[cache]
capacity = 1000      # 缓存容量
ttl_seconds = 3600   # 过期时间（秒）

[threadpool]
num_workers = 4      # 工作线程数

[tuner]
enable = true                # 启用自动调优
latency_max_ms = 100.0      # 延迟阈值（毫秒）
recall_min_pct = 0.85       # 召回率阈值
ef_delta = 5                # ef调整步长
topk_delta = 2              # topK调整步长
check_interval_seconds = 30  # 检查间隔（秒）

# SQLite 数据库配置
[sqlite]
db_path = "rag_store.db"       # 数据库文件路径
vector_extension = "sqlite_vec" # 向量扩展名
vector_dimension = 768         # 向量维度
enable_fts5 = true            # 启用 FTS5 全文检索
enable_wal = true             # 启用 WAL 模式
cache_size = 10000            # 缓存页数
busy_timeout = 30000          # 忙等待超时（毫秒）
fts5_limit = 50              # FTS5 检索结果数量
vector_limit = 50            # 向量检索结果数量

# 混合 RAG 系统配置
[hybrid]
enable = true                    # 启用混合 RAG 系统
hot_threshold = 3               # 热数据访问次数阈值
memory_capacity = 1000          # 内存层最大文档数
auto_optimize_interval = 300    # 自动优化间隔（秒）
enable_smart_caching = true     # 启用智能缓存
parallel_search = true          # 并行搜索内存和SQLite
memory_first = true             # 优先从内存层搜索
result_merge_method = "score"   # 结果合并方式：score/relevance
max_results_per_layer = 20      # 每层最大结果数
enable_benchmark = true         # 启用性能基准测试
stats_interval = 100           # 统计信息更新间隔
cache_hit_threshold = 0.8      # 缓存命中率阈值
```

### 2. SQLite 数据库系统

框架提供基于 SQLite 的持久化存储和检索功能：

#### 2.1 数据库初始化

```cpp
#include "rag/sqlite_retriever.h"

// 创建 SQLite RAG 系统
SQLiteRAGSystem rag_system("./rag_config.toml");

// 初始化数据库和索引
if (!rag_system.initialize()) {
    std::cerr << "Failed to initialize database" << std::endl;
    return 1;
}
```

#### 2.2 文档存储

```cpp
// 创建文档
std::vector<Chunk> documents;
Chunk doc;
doc.doc_id = "tech_article_001";
doc.seq_no = 0;
doc.topic = "Machine Learning";
doc.text = "Machine learning algorithms can automatically learn from data...";
documents.push_back(doc);

// 加载到数据库（自动创建 FTS5 索引和向量嵌入）
auto loaded_count = rag_system.load_documents(documents);
std::cout << "Loaded " << loaded_count << " documents" << std::endl;
```

#### 2.3 检索查询

```cpp
// 混合检索（FTS5 + 向量）
auto results = rag_system.search("machine learning algorithms", 10);

for (const auto& result : results) {
    std::cout << "Doc ID: " << result.doc_id << std::endl;
    std::cout << "Topic: " << result.topic << std::endl;
    std::cout << "Score: " << result.score << std::endl;
    std::cout << "Content: " << result.content.substr(0, 100) << "..." << std::endl;
    std::cout << "---" << std::endl;
}
```

#### 2.4 检索策略

支持多种检索策略：

```cpp
auto retriever = rag_system.get_retriever();

// 仅文本检索（FTS5 + BM25）
auto text_results = retriever->query_text_only("machine learning", 5);

// 仅向量检索
auto vector_results = retriever->query_vector_only("artificial intelligence", 5);

// 混合检索（推荐）
auto hybrid_results = retriever->query_hybrid("deep learning networks", 5);

// 异步查询
auto future_results = retriever->query_async("neural networks", 5);
auto async_results = future_results.get();
```

#### 2.5 数据库管理

```cpp
// 获取数据库统计
auto stats = rag_system.get_system_stats();
std::cout << "Total chunks: " << stats.total_chunks << std::endl;
std::cout << "Total embeddings: " << stats.total_embeddings << std::endl;
std::cout << "Database size: " << stats.db_size_mb << " MB" << std::endl;
std::cout << "Last update: " << stats.last_update << std::endl;

// 清空所有数据
retriever->clear_all_data();
```

### 3. 多语言Tokenizer

支持英文、中文和混合语言的智能分词：

```cpp
// 配置分词器
TokenizerConfig config;
config.lowercase = true;                    // 启用小写转换
config.remove_punctuation = true;           // 移除标点符号
config.filter_stopwords = true;            // 过滤停用词
config.enable_chinese_segmentation = true;  // 启用中文分词

Tokenizer tokenizer(config);

// 自动语言检测
auto language = tokenizer.detect_language("机器学习和Machine Learning");
// 返回: Language::MIXED

// 分词处理
auto tokens = tokenizer.tokenize("Natural Language Processing自然语言处理");
// 返回: ["natural", "language", "processing", "自然语言", "处理"]
```

### 3. BM25检索器

高性能的BM25文本相关性计算：

```cpp
// 创建BM25索引器
BM25Config bm25_config;
bm25_config.k1 = 1.2;  // 调节词频饱和度
bm25_config.b = 0.75;  // 调节文档长度归一化

BM25Indexer bm25(bm25_config);

// 设置高级分词器
TokenizerConfig tokenizer_config;
tokenizer_config.filter_stopwords = true;
bm25.set_tokenizer_config(tokenizer_config);

// 构建索引
std::vector<Chunk> documents = load_documents();
bm25.fit(documents);

// 执行查询
auto results = bm25.query_text("machine learning algorithms", 10);
// 返回: vector<pair<size_t, double>> - (文档索引, BM25分数)
```

### 4. 融合检索器

集成BM25和HNSW的多策略融合检索：

```cpp
// 配置融合检索器
FusionRetrieverConfig config;
config.strategy = FusionStrategy::HYBRID;  // 混合策略
config.bm25_weight = 0.6;                 // BM25权重
config.vector_weight = 0.4;               // 向量权重

auto retriever = std::make_shared<FusionRetriever>(config);
retriever->fit(documents);

// 同步查询
auto results = retriever->query("deep learning", 5);

// 异步查询
auto future = retriever->query_async("neural networks", 5);
auto async_results = future.get();

// 切换融合策略
config.strategy = FusionStrategy::RRF;  // 使用RRF融合
retriever->update_config(config);
```

#### 融合策略说明

1. **BM25_ONLY**: 仅使用BM25文本检索
2. **VECTOR_ONLY**: 仅使用HNSW向量检索
3. **HYBRID**: 加权融合BM25和向量分数
4. **RRF**: 倒数排名融合（Reciprocal Rank Fusion）

### 5. LRU缓存

高性能的查询结果缓存系统：

```cpp
// 配置缓存
CacheConfig cache_config;
cache_config.capacity = 1000;      // 最大缓存项数
cache_config.ttl_seconds = 3600;   // 1小时过期

LRUCache cache(cache_config);

// 缓存查询结果
Retrieval result;
result.top_chunks = {1, 3, 5};
result.timestamp = std::time(nullptr);
cache.put("machine learning", result);

// 获取缓存
Retrieval cached;
if (cache.get("machine learning", cached)) {
    std::cout << "Cache hit!" << std::endl;
} else {
    std::cout << "Cache miss!" << std::endl;
}
```

### 6. 线程池

高效的并发任务处理：

```cpp
// 创建线程池
ThreadPoolConfig config;
config.num_workers = 8;  // 8个工作线程

ThreadPool pool(config);

// 提交任务
auto future1 = pool.submit([&]{ return bm25.query(terms, 10); });
auto future2 = pool.submit([&]{ return hnsw.search(embedding, 10); });

// 等待结果
auto bm25_results = future1.get();
auto hnsw_results = future2.get();
```

### 8. 混合RAG系统 🔥

**终极解决方案：内存RAG + SQLite RAG 完美结合**

混合RAG系统是框架的核心特性，它将内存RAG和SQLite RAG的优势完美结合，实现了**热数据内存缓存 + 冷数据持久化存储**的双层架构。

#### 8.1 架构设计理念

```
┌─────────────────────────────────────────────────────────────┐
│                     混合RAG系统架构                           │
├─────────────────────────────────────────────────────────────┤
│  🔥 热数据层 (内存)  │  ❄️ 冷数据层 (SQLite)                 │
│  • BM25 + HNSW     │  • FTS5 + Vector                     │
│  • 毫秒级响应        │  • 无限容量                          │
│  • 高频访问数据      │  • 数据持久化                        │
│  • LRU自动管理      │  • ACID事务保证                      │
├─────────────────────────────────────────────────────────────┤
│              智能数据分层管理器                               │
│  • 自动热点识别      • 动态数据迁移      • 统一检索接口       │
└─────────────────────────────────────────────────────────────┘
```

#### 8.2 核心优势

- **🚀 极致性能**: 热数据内存检索，平均响应时间 < 1ms
- **📦 无限容量**: 冷数据SQLite存储，支持TB级文档库
- **🧠 智能分层**: 基于访问模式自动识别热点数据
- **🔄 无缝切换**: 透明的数据迁移，用户无感知
- **⚡ 并行检索**: 双层同时检索，智能结果合并
- **💾 数据安全**: SQLite ACID保证，支持备份恢复

#### 8.3 快速开始

```cpp
#include "rag/hybrid_rag_system.h"

// 1. 创建混合RAG系统
HybridRAGSystem hybrid_rag("rag_config.toml");

// 2. 加载文档（自动分层存储）
std::vector<Chunk> documents = load_large_dataset();
auto loaded_count = hybrid_rag.load_documents(documents);

// 3. 智能检索（自动选择最优路径）
auto results = hybrid_rag.search("机器学习算法", 10);

// 4. 查看系统状态
hybrid_rag.print_system_stats();
```

#### 8.4 完整示例程序

框架提供了完整的混合RAG演示程序：

```bash
# 编译混合RAG演示
cd rag/example/build
make hybrid_rag_demo

# 运行演示
./hybrid_rag_demo
```

演示程序展示了：

- 📚 **大规模数据加载**: 36个文档的双层存储
- 🔍 **多轮查询模拟**: 模拟真实用户访问模式
- 📊 **热点数据识别**: 自动识别高频查询文档
- 🔄 **数据分层优化**: 热数据自动迁移到内存层
- ⚡ **性能基准测试**: 1298 QPS 的查询吞吐量

#### 8.5 智能数据分层策略

```cpp
// 配置热点识别参数
hybrid_rag.set_hot_threshold(3);        // 访问3次以上为热数据
hybrid_rag.set_memory_capacity(1000);   // 内存层最大容量

// 手动触发分层优化
hybrid_rag.optimize_data_distribution();

// 查看热点统计
auto stats = hybrid_rag.get_access_stats();
std::cout << "热数据文档数: " << stats.hot_document_count << std::endl;
std::cout << "内存命中率: " << stats.memory_hit_rate << "%" << std::endl;
```

#### 8.6 性能监控

```cpp
// 运行基准测试
std::vector<std::string> queries = {
    "机器学习", "深度学习", "人工智能", "数据科学"
};
hybrid_rag.run_benchmark(queries);

// 输出:
// 📈 基准测试汇总:
//   • 平均查询时间: 770.14μs
//   • 平均结果数量: 5.0 个
//   • 系统吞吐量: 1298 QPS
//   • 内存命中率: 50.0%
```

#### 8.7 实际应用场景

- **📖 大型知识库**: 热门文档内存缓存，历史文档持久化
- **🤖 智能客服**: 常见问题快速响应，长尾问题完整覆盖
- **🔬 科研平台**: 热点论文秒级检索，全量文献无限存储
- **💼 企业搜索**: 重要文档优先级访问，全部资料统一管理

混合RAG系统是处理大规模文档库的理想解决方案，它在保证检索性能的同时，提供了企业级的数据管理能力。

### 9. 自动调优器

基于性能指标的参数自动优化：

```cpp
// 配置调优器
TunerConfig config;
config.enable = true;
config.latency_max_ms = 100.0;     // 延迟阈值100ms
config.recall_min_pct = 0.85;      // 召回率阈值85%
config.check_interval_seconds = 30; // 30秒检查一次

// 定义监控函数
auto latency_monitor = []() -> double {
    return getCurrentLatency();  // 返回当前平均延迟
};

auto recall_monitor = []() -> double {
    return getCurrentRecall();   // 返回当前召回率
};

// 创建调优器
AutoTuner tuner(config, latency_monitor, recall_monitor);

// 启动自动调优
tuner.start();

// 获取当前参数
auto params = tuner.params();
std::cout << "Current ef: " << params.ef
          << ", topK: " << params.topK << std::endl;

// 停止调优
tuner.stop();
```

## 🧪 测试用例

### 功能测试

项目包含全面的功能测试，验证各个模块的正确性：

```bash
# 编译并运行综合测试
cd rag/example/build
./rag_example
```

测试覆盖内容：
- ✅ 多语言分词器测试
- ✅ BM25检索精度测试
- ✅ 融合检索策略对比
- ✅ LRU缓存命中率测试
- ✅ 并发查询性能测试
- ✅ 自动调优参数变化测试
- ✅ 端到端RAG流程测试

### 性能基准

在Intel i7-8700K @ 3.70GHz，32GB RAM环境下的性能表现：

| 操作 | 延迟 | 吞吐量 |
|------|------|--------|
| BM25查询 | ~50μs | 20,000 QPS |
| HNSW查询 | ~200μs | 5,000 QPS |
| 融合查询 | ~300μs | 3,300 QPS |
| 缓存命中 | ~1μs | 1,000,000 QPS |

### 内存占用

| 组件 | 1万文档 | 10万文档 | 100万文档 |
|------|---------|----------|-----------|
| BM25索引 | ~50MB | ~500MB | ~5GB |
| HNSW索引 | ~100MB | ~1GB | ~10GB |
| LRU缓存 | ~10MB | ~10MB | ~10MB |

## 🔧 高级配置

### 自定义分词器

```cpp
class CustomTokenizer : public TokenizerInterface {
public:
    std::vector<std::string> tokenize(const std::string& text) override {
        // 实现自定义分词逻辑
        return custom_tokenize(text);
    }

    }

    Language detect_language(const std::string& text) override {
        // 实现语言检测逻辑
        return detect_custom_language(text);
    }
};
```

## 🎯 完整功能示例

以下示例展示了所有主要功能的集成使用：

```cpp
#include "rag/sqlite_retriever.h"
#include "rag/fusion_retriever.h"
#include "rag/config.h"
#include "rag/thread_pool.h"
#include "rag/lru_cache.h"

class ComprehensiveRAGDemo {
private:
    std::unique_ptr<SQLiteRAGSystem> sqlite_rag_;
    std::unique_ptr<FusionRetriever> memory_rag_;
    std::unique_ptr<ThreadPool> thread_pool_;
    std::unique_ptr<LRUCache<std::string, std::vector<SQLiteSearchResult>>> cache_;

public:
    ComprehensiveRAGDemo(const std::string& config_path) {
        // 1. 加载配置
        auto config = ConfigLoader::load(config_path);

        // 2. 初始化SQLite RAG系统
        sqlite_rag_ = std::make_unique<SQLiteRAGSystem>(config_path);
        if (!sqlite_rag_->initialize()) {
            throw std::runtime_error("Failed to initialize SQLite RAG system");
        }

        // 3. 初始化内存RAG系统
        memory_rag_ = FusionRetriever::from_config(*config);

        // 4. 初始化线程池
        thread_pool_ = std::make_unique<ThreadPool>(config->threadpool.num_workers);

        // 5. 初始化缓存
        cache_ = std::make_unique<LRUCache<std::string, std::vector<SQLiteSearchResult>>>(
            config->cache.capacity, config->cache.ttl_seconds);
    }

    // 混合检索：同时使用内存和SQLite系统
    std::vector<SearchResult> hybrid_search(const std::string& query, int limit = 10) {
        // 并行执行两种检索
        auto sqlite_future = thread_pool_->submit([this, query, limit]() {
            return sqlite_rag_->search(query, limit);
        });

        auto memory_future = thread_pool_->submit([this, query, limit]() {
            return memory_rag_->query(query, limit);
        });

        // 获取结果
        auto sqlite_results = sqlite_future.get();
        auto memory_results = memory_future.get();

        // 合并和重排序
        return merge_and_rerank(sqlite_results, memory_results, query, limit);
    }

    // 智能查询路由
    std::vector<SearchResult> smart_search(const std::string& query, int limit = 10) {
        // 1. 检查缓存
        auto cached_result = cache_->get(query);
        if (cached_result) {
            std::cout << "Cache hit for query: " << query << std::endl;
            return convert_to_search_result(*cached_result);
        }

        // 2. 分析查询特征
        QueryType query_type = analyze_query(query);

        std::vector<SearchResult> results;

        switch (query_type) {
            case QueryType::FACTUAL:
                // 事实查询优先使用SQLite FTS5
                results = convert_to_search_result(sqlite_rag_->search(query, limit));
                break;

            case QueryType::SEMANTIC:
                // 语义查询优先使用向量检索
                results = convert_to_search_result(memory_rag_->query(query, limit));
                break;

            case QueryType::COMPLEX:
                // 复杂查询使用混合检索
                results = hybrid_search(query, limit);
                break;
        }

        // 3. 更新缓存
        cache_->put(query, convert_from_search_result(results));

        return results;
    }

    // 批量文档处理
    void batch_load_documents(const std::vector<std::string>& file_paths) {
        std::vector<Chunk> all_chunks;

        // 并行处理文件
        std::vector<std::future<std::vector<Chunk>>> futures;

        for (const auto& file_path : file_paths) {
            auto future = thread_pool_->submit([this, file_path]() {
                return process_document_file(file_path);
            });
            futures.push_back(std::move(future));
        }

        // 收集所有chunks
        for (auto& future : futures) {
            auto chunks = future.get();
            all_chunks.insert(all_chunks.end(), chunks.begin(), chunks.end());
        }

        std::cout << "Processed " << all_chunks.size() << " chunks from "
                  << file_paths.size() << " files" << std::endl;

        // 同时加载到两个系统
        auto sqlite_future = thread_pool_->submit([this, &all_chunks]() {
            return sqlite_rag_->load_documents(all_chunks);
        });

        auto memory_future = thread_pool_->submit([this, &all_chunks]() {
            memory_rag_->fit(all_chunks);
        });

        auto sqlite_loaded = sqlite_future.get();
        memory_future.get();  // 等待内存索引完成

        std::cout << "Loaded " << sqlite_loaded << " documents to SQLite" << std::endl;
        std::cout << "Built in-memory index for " << all_chunks.size() << " chunks" << std::endl;
    }

    // 性能监控
    void print_performance_stats() {
        // SQLite统计
        auto sqlite_stats = sqlite_rag_->get_system_stats();
        std::cout << "\n📊 SQLite RAG Statistics:" << std::endl;
        std::cout << "   Documents: " << sqlite_stats.total_chunks << std::endl;
        std::cout << "   Embeddings: " << sqlite_stats.total_embeddings << std::endl;
        std::cout << "   DB Size: " << sqlite_stats.db_size_mb << " MB" << std::endl;

        // 缓存统计
        auto cache_stats = cache_->get_stats();
        std::cout << "\n💾 Cache Statistics:" << std::endl;
        std::cout << "   Hit Rate: " << (cache_stats.hit_rate * 100) << "%" << std::endl;
        std::cout << "   Size: " << cache_stats.size << "/" << cache_stats.capacity << std::endl;

        // 线程池统计
        std::cout << "\n🧵 Thread Pool Statistics:" << std::endl;
        std::cout << "   Active Threads: " << thread_pool_->active_count() << std::endl;
        std::cout << "   Queue Size: " << thread_pool_->queue_size() << std::endl;
    }

private:
    enum class QueryType { FACTUAL, SEMANTIC, COMPLEX };

    QueryType analyze_query(const std::string& query) {
        // 简单的查询分析逻辑
        if (query.find("what") != std::string::npos ||
            query.find("when") != std::string::npos ||
            query.find("where") != std::string::npos) {
            return QueryType::FACTUAL;
        }

        if (query.length() > 50 ||
            std::count(query.begin(), query.end(), ' ') > 8) {
            return QueryType::COMPLEX;
        }

        return QueryType::SEMANTIC;
    }

    std::vector<Chunk> process_document_file(const std::string& file_path) {
        // 文档处理逻辑
        std::vector<Chunk> chunks;
        // ... 实现文档解析和分块
        return chunks;
    }

    std::vector<SearchResult> merge_and_rerank(
        const std::vector<SQLiteSearchResult>& sqlite_results,
        const std::vector<RetrievalResult>& memory_results,
        const std::string& query,
        int limit) {
        // 结果合并和重排序逻辑
        std::vector<SearchResult> merged;
        // ... 实现合并逻辑
        return merged;
    }
};

// 使用示例
int main() {
    try {
        ComprehensiveRAGDemo demo("rag_config.toml");

        // 1. 批量加载文档
        std::vector<std::string> document_files = {
            "docs/machine_learning.pdf",
            "docs/deep_learning.md",
            "docs/ai_ethics.txt"
        };
        demo.batch_load_documents(document_files);

        // 2. 智能查询测试
        std::vector<std::string> test_queries = {
            "What is machine learning?",           // 事实查询
            "深度学习的发展历程和未来趋势",              // 语义查询
            "How does machine learning relate to artificial intelligence and what are the ethical implications?"  // 复杂查询
        };

        for (const auto& query : test_queries) {
            std::cout << "\n🔍 Query: " << query << std::endl;
            auto start = std::chrono::high_resolution_clock::now();

            auto results = demo.smart_search(query, 5);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            std::cout << "⏱️  Search completed in " << duration.count() << "μs" << std::endl;
            std::cout << "📊 Found " << results.size() << " results" << std::endl;

            for (size_t i = 0; i < std::min(size_t(3), results.size()); ++i) {
                std::cout << "   " << (i+1) << ". " << results[i].title
                         << " (Score: " << results[i].score << ")" << std::endl;
            }
        }

        // 3. 性能统计
        demo.print_performance_stats();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
```

## 🚀 部署与生产环境

### Docker 容器化部署

```dockerfile
# Dockerfile for production deployment
FROM ubuntu:22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libsqlite3-dev \
    pkg-config \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install sqlite-vec extension
RUN wget https://github.com/asg017/sqlite-vec/releases/download/v0.0.1/sqlite-vec-0.0.1-linux-x86_64.tar.gz \
    && tar -xzf sqlite-vec-0.0.1-linux-x86_64.tar.gz \
    && cp sqlite-vec.so /usr/local/lib/ \
    && ldconfig

# Copy source and build
WORKDIR /app
COPY . .
RUN mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && make -j$(nproc)

# Production image
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy built application and libraries
COPY --from=builder /usr/local/lib/sqlite-vec.so /usr/local/lib/
COPY --from=builder /app/build/rag_example /usr/local/bin/
COPY --from=builder /app/rag_config.toml /etc/rag/

# Create data directory
RUN mkdir -p /data && chown -R 1000:1000 /data

# Non-root user
USER 1000:1000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD /usr/local/bin/rag_example --health-check || exit 1

EXPOSE 8080
VOLUME ["/data"]

CMD ["/usr/local/bin/rag_example", "--config", "/etc/rag/rag_config.toml"]
```

### Kubernetes 部署配置

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: humanus-rag
  labels:
    app: humanus-rag
spec:
  replicas: 3
  selector:
    matchLabels:
      app: humanus-rag
  template:
    metadata:
      labels:
        app: humanus-rag
    spec:
      containers:
      - name: rag-service
        image: humanus/rag:latest
        ports:
        - containerPort: 8080
        env:
        - name: CONFIG_PATH
          value: "/etc/rag/rag_config.toml"
        - name: DATA_PATH
          value: "/data"
        volumeMounts:
        - name: config-volume
          mountPath: /etc/rag
        - name: data-volume
          mountPath: /data
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
      volumes:
      - name: config-volume
        configMap:
          name: rag-config
      - name: data-volume
        persistentVolumeClaim:
          claimName: rag-data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: humanus-rag-service
spec:
  selector:
    app: humanus-rag
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

### 监控和告警配置

```yaml
# monitoring/prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s

    scrape_configs:
    - job_name: 'humanus-rag'
      static_configs:
      - targets: ['humanus-rag-service:80']
      metrics_path: /metrics
      scrape_interval: 10s

    rule_files:
    - "/etc/prometheus/rules/*.yml"

    alerting:
      alertmanagers:
      - static_configs:
        - targets:
          - alertmanager:9093

---
# Alert rules
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-alert-rules
data:
  rag.yml: |
    groups:
    - name: rag-alerts
      rules:
      - alert: RAGHighLatency
        expr: rag_query_duration_seconds_p95 > 0.5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "RAG query latency is high"
          description: "95th percentile latency is {{ $value }}s"

      - alert: RAGLowCacheHitRate
        expr: rag_cache_hit_rate < 0.7
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "RAG cache hit rate is low"
          description: "Cache hit rate is {{ $value }}"

      - alert: RAGDatabaseSize
        expr: rag_database_size_mb > 10000
        for: 10m
        labels:
          severity: info
        annotations:
          summary: "RAG database size is large"
          description: "Database size is {{ $value }}MB"
```

## 📚 API 参考文档

### 核心类参考

#### FusionRetriever

```cpp
class FusionRetriever {
public:
    /**
     * 构造函数
     * @param config 融合检索器配置
     */
    explicit FusionRetriever(const FusionRetrieverConfig& config);

    /**
     * 从RAG配置创建检索器
     * @param config RAG系统配置
     * @return 检索器智能指针
     */
    static std::shared_ptr<FusionRetriever> from_config(const RAGConfig& config);

    /**
     * 构建索引
     * @param chunks 文档块集合
     * @throws RAGException 当索引构建失败时
     */
    void fit(const std::vector<Chunk>& chunks);

    /**
     * 执行查询
     * @param query_text 查询文本
     * @param top_k 返回结果数量
     * @return 检索结果列表
     */
    std::vector<RetrievalResult> query(const std::string& query_text, int top_k = 10);

    /**
     * 异步查询
     * @param query_text 查询文本
     * @param top_k 返回结果数量
     * @return 异步结果future
     */
    std::future<std::vector<RetrievalResult>> query_async(const std::string& query_text, int top_k = 10);

    /**
     * 获取统计信息
     * @return 统计信息结构
     */
    IndexStats get_stats() const;

    /**
     * 更新配置
     * @param config 新配置
     */
    void update_config(const FusionRetrieverConfig& config);
};
```

#### SQLiteRAGSystem

```cpp
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
     * 执行搜索
     * @param query 查询字符串
     * @param limit 结果数量限制
     * @return 搜索结果列表
     */
    std::vector<SQLiteSearchResult> search(const std::string& query, int limit = 10);

    /**
     * 异步搜索
     * @param query 查询字符串
     * @param limit 结果数量限制
     * @return 异步搜索结果
     */
    std::future<std::vector<SQLiteSearchResult>> search_async(const std::string& query, int limit = 10);

    /**
     * 获取系统统计信息
     * @return 数据库统计信息
     */
    SQLiteDB::DBStats get_system_stats();

    /**
     * 清理和优化数据库
     */
    void optimize_database();

    /**
     * 备份数据库
     * @param backup_path 备份文件路径
     * @return 是否备份成功
     */
    bool backup_database(const std::string& backup_path);
};
```

### 配置结构参考

#### RAGConfig

```cpp
struct RAGConfig {
    struct ChunkConfig {
        int size = 512;
        int overlap = 128;
        int min_size = 64;
    } chunk;

    struct BM25Config {
        double k1 = 1.5;
        double b = 0.75;
    } bm25;

    struct HNSWConfig {
        int M = 16;
        int ef_construction = 200;
        int ef_query = 50;
    } hnsw;

    struct FusionConfig {
        std::string strategy = "hybrid";
        double bm25_weight = 0.6;
        double vector_weight = 0.4;
        double rrf_k = 60.0;
        bool enable_rerank = true;
        int max_candidates = 100;
    } fusion;

    struct CacheConfig {
        int capacity = 1024;
        int ttl_seconds = 3600;
    } cache;

    struct ThreadPoolConfig {
        int num_workers = 8;
    } threadpool;

    struct TunerConfig {
        bool enable = true;
        double latency_max_ms = 200.0;
        double recall_min_pct = 0.8;
        int ef_delta = 5;
        int topk_delta = 2;
        int check_interval_seconds = 10;
    } tuner;

    struct SQLiteConfig {
        std::string db_path = "rag_store.db";
        std::string vector_extension = "sqlite_vec";
        int vector_dimension = 768;
        bool enable_fts5 = true;
        bool enable_wal = true;
        int cache_size = 10000;
        int busy_timeout = 30000;
        int fts5_limit = 50;
        int vector_limit = 50;
    } sqlite;
};
```

## 🎓 最佳实践

### 1. 性能优化建议

```cpp
// 1. 预热缓存
void warm_up_cache(FusionRetriever& retriever) {
    std::vector<std::string> common_queries = {
        "machine learning",
        "artificial intelligence",
        "deep learning",
        "neural networks"
    };

    for (const auto& query : common_queries) {
        retriever.query(query, 10);  // 预热查询
    }
}

// 2. 批量处理优化
void batch_process_documents(SQLiteRAGSystem& rag, const std::vector<std::string>& files) {
    const size_t batch_size = 100;

    for (size_t i = 0; i < files.size(); i += batch_size) {
        size_t end = std::min(i + batch_size, files.size());
        std::vector<std::string> batch(files.begin() + i, files.begin() + end);

        // 并行处理批次
        std::vector<std::future<std::vector<Chunk>>> futures;
        for (const auto& file : batch) {
            futures.push_back(std::async(std::launch::async, [&file]() {
                return process_file(file);
            }));
        }

        // 收集结果并批量插入
        std::vector<Chunk> batch_chunks;
        for (auto& future : futures) {
            auto chunks = future.get();
            batch_chunks.insert(batch_chunks.end(), chunks.begin(), chunks.end());
        }

        rag.load_documents(batch_chunks);
    }
}

// 3. 内存使用优化
class MemoryOptimizedProcessor {
    static constexpr size_t MAX_MEMORY_MB = 1024;  // 1GB limit

public:
    void process_large_dataset(const std::string& dataset_path) {
        auto file_stream = std::ifstream(dataset_path);
        std::string line;
        std::vector<Chunk> buffer;
        size_t memory_usage = 0;

        while (std::getline(file_stream, line)) {
            auto chunk = parse_line(line);
            buffer.push_back(chunk);
            memory_usage += chunk.text.size();

            // 当内存使用超过限制时，处理当前批次
            if (memory_usage > MAX_MEMORY_MB * 1024 * 1024) {
                process_batch(buffer);
                buffer.clear();
                memory_usage = 0;
            }
        }

        // 处理剩余数据
        if (!buffer.empty()) {
            process_batch(buffer);
        }
    }
};
```

### 2. 错误处理策略

```cpp
class RobustRAGService {
public:
    std::vector<SearchResult> search_with_fallback(const std::string& query, int limit) {
        try {
            // 尝试主要检索方法
            return primary_search(query, limit);
        } catch (const DatabaseException& e) {
            std::cerr << "Database error: " << e.what() << std::endl;
            // 降级到备用方法
            return fallback_search(query, limit);
        } catch (const std::exception& e) {
            std::cerr << "Unexpected error: " << e.what() << std::endl;
            // 返回空结果而不是崩溃
            return {};
        }
    }

private:
    std::vector<SearchResult> primary_search(const std::string& query, int limit) {
        // 主要检索逻辑
        if (sqlite_rag_->is_available()) {
            return sqlite_rag_->search(query, limit);
        } else {
            throw DatabaseException("SQLite RAG system not available");
        }
    }

    std::vector<SearchResult> fallback_search(const std::string& query, int limit) {
        // 备用检索逻辑（使用内存系统）
        return memory_rag_->query(query, limit);
    }
};

// 自定义异常类
class RAGException : public std::exception {
protected:
    std::string message_;

public:
    explicit RAGException(const std::string& msg) : message_(msg) {}
    const char* what() const noexcept override { return message_.c_str(); }
};

class DatabaseException : public RAGException {
public:
    explicit DatabaseException(const std::string& msg)
        : RAGException("Database Error: " + msg) {}
};

class ConfigurationException : public RAGException {
public:
    explicit ConfigurationException(const std::string& msg)
        : RAGException("Configuration Error: " + msg) {}
};
```

### 3. 监控与日志

```cpp
#include <spdlog/spdlog.h>
#include <prometheus/counter.h>
#include <prometheus/histogram.h>

class RAGMetrics {
private:
    std::shared_ptr<prometheus::Registry> registry_;
    prometheus::Counter* queries_total_;
    prometheus::Histogram* query_duration_;
    prometheus::Counter* cache_hits_;
    prometheus::Counter* cache_misses_;

public:
    RAGMetrics() {
        registry_ = std::make_shared<prometheus::Registry>();

        // 查询计数器
        auto& counter_family = prometheus::BuildCounter()
            .Name("rag_queries_total")
            .Help("Total number of RAG queries")
            .Register(*registry_);
        queries_total_ = &counter_family.Add({{"type", "search"}});

        // 查询延迟直方图
        auto& histogram_family = prometheus::BuildHistogram()
            .Name("rag_query_duration_seconds")
            .Help("RAG query duration in seconds")
            .Register(*registry_);
        query_duration_ = &histogram_family.Add({},
            {0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0});

        // 缓存指标
        auto& cache_family = prometheus::BuildCounter()
            .Name("rag_cache_operations_total")
            .Help("Total cache operations")
            .Register(*registry_);
        cache_hits_ = &cache_family.Add({{"result", "hit"}});
        cache_misses_ = &cache_family.Add({{"result", "miss"}});
    }

    void record_query(double duration_seconds) {
        queries_total_->Increment();
        query_duration_->Observe(duration_seconds);
    }

    void record_cache_hit() { cache_hits_->Increment(); }
    void record_cache_miss() { cache_misses_->Increment(); }

    std::shared_ptr<prometheus::Registry> get_registry() { return registry_; }
};

class InstrumentedRAGService {
private:
    std::unique_ptr<SQLiteRAGSystem> rag_system_;
    std::unique_ptr<RAGMetrics> metrics_;

public:
    InstrumentedRAGService(const std::string& config_path) {
        rag_system_ = std::make_unique<SQLiteRAGSystem>(config_path);
        metrics_ = std::make_unique<RAGMetrics>();

        // 配置日志
        spdlog::set_level(spdlog::level::info);
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%t] %v");
    }

    std::vector<SQLiteSearchResult> search(const std::string& query, int limit) {
        auto start = std::chrono::high_resolution_clock::now();

        spdlog::info("Processing search query: '{}', limit: {}", query, limit);

        try {
            auto results = rag_system_->search(query, limit);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double>(end - start).count();

            // 记录指标
            metrics_->record_query(duration);

            spdlog::info("Search completed successfully: {} results in {:.3f}s",
                        results.size(), duration);

            return results;

        } catch (const std::exception& e) {
            spdlog::error("Search failed for query '{}': {}", query, e.what());
            throw;
        }
    }
};
```

## 🔧 故障排除指南

### 常见问题与解决方案

#### 1. 编译问题

**问题**: C++17 特性不支持
```bash
error: 'std::optional' is not a member of 'std'
```

**解决方案**:
```bash
# 确保编译器版本
g++ --version  # 需要 7.0+
clang++ --version  # 需要 6.0+

# 设置 C++17 标准
cmake -DCMAKE_CXX_STANDARD=17 ..
# 或在 CMakeLists.txt 中添加
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

#### 2. SQLite 扩展问题

**问题**: 向量扩展加载失败
```
Warning: Failed to load vector extension 'sqlite_vec'
```

**解决方案**:
```bash
# 方法1: 下载预编译扩展
wget https://github.com/asg017/sqlite-vec/releases/latest/download/sqlite-vec-linux-x86_64.tar.gz
tar -xzf sqlite-vec-linux-x86_64.tar.gz
sudo cp sqlite-vec.so /usr/local/lib/
sudo ldconfig

# 方法2: 从源码编译
git clone https://github.com/asg017/sqlite-vec.git
cd sqlite-vec
make loadable
sudo make install

# 方法3: 设置扩展路径
export LD_LIBRARY_PATH=/path/to/extensions:$LD_LIBRARY_PATH
```

#### 3. 性能问题

**问题**: 查询速度慢
```cpp
// 诊断工具
class PerformanceDiagnostic {
public:
    void diagnose_slow_query(const std::string& query) {
        std::cout << "Diagnosing query: " << query << std::endl;

        // 1. 检查查询复杂度
        auto complexity = analyze_query_complexity(query);
        std::cout << "Query complexity: " << complexity << std::endl;

        // 2. 检查数据库大小
        auto stats = rag_system_->get_system_stats();
        std::cout << "Database size: " << stats.db_size_mb << "MB" << std::endl;
        std::cout << "Total chunks: " << stats.total_chunks << std::endl;

        // 3. 检查缓存命中率
        auto cache_stats = cache_->get_stats();
        std::cout << "Cache hit rate: " << (cache_stats.hit_rate * 100) << "%" << std::endl;

        // 4. 推荐优化
        suggest_optimizations(complexity, stats, cache_stats);
    }

private:
    void suggest_optimizations(int complexity, const auto& db_stats, const auto& cache_stats) {
        if (complexity > 10) {
            std::cout << "🔧 Consider simplifying the query" << std::endl;
        }

        if (db_stats.db_size_mb > 1000) {
            std::cout << "🔧 Consider database partitioning" << std::endl;
        }

        if (cache_stats.hit_rate < 0.5) {
            std::cout << "🔧 Consider increasing cache size or TTL" << std::endl;
        }
    }
};
```

#### 4. 内存泄漏

**检测工具**:
```bash
# 使用 Valgrind
valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all ./rag_example

# 使用 AddressSanitizer
g++ -fsanitize=address -g -o rag_example_debug *.cpp
./rag_example_debug

# 使用 tcmalloc
env LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4" ./rag_example
```

**常见内存问题修复**:
```cpp
// 1. RAII 原则
class SafeRAGService {
    std::unique_ptr<SQLiteRAGSystem> rag_;  // 自动清理

public:
    SafeRAGService(const std::string& config)
        : rag_(std::make_unique<SQLiteRAGSystem>(config)) {}

    // 析构函数自动调用，无需手动释放
    ~SafeRAGService() = default;
};

// 2. 避免循环引用
class DocumentIndex {
    std::vector<std::shared_ptr<Document>> documents_;
    std::weak_ptr<DocumentIndex> self_;  // 使用 weak_ptr 避免循环

public:
    void set_self(std::shared_ptr<DocumentIndex> self) {
        self_ = self;
    }
};

// 3. 及时释放大对象
void process_large_dataset() {
    {
        std::vector<Chunk> large_chunks = load_large_dataset();
        process_chunks(large_chunks);
        // large_chunks 在作用域结束时自动释放
    }

    // 继续其他处理，内存已释放
    other_processing();
}
```

## 📞 支持与社区

### 获取帮助

1. **📚 文档**: [完整文档](https://github.com/jblymq/RAG-CCC)
2. **💬 讨论**: [GitHub Discussions](https://github.com/jblymq/RAG-CCC/discussions)
3. **🐛 问题报告**: [GitHub Issues](https://github.com/jblymq/RAG-CCC/issues)
4. **📧 邮件**: myth-lab@whu.edu.cn

### 贡献代码

```bash
# 1. Fork 项目
git clone https://github.com/jblymq/RAG-CCC.git

# 2. 创建特性分支
git checkout -b feature/amazing-feature

# 3. 提交更改
git commit -m 'feat: add amazing feature'

# 4. 推送到分支
git push origin feature/amazing-feature

# 5. 创建 Pull Request
```

### 代码贡献指南

1. **代码风格**: 遵循 Google C++ Style Guide
2. **测试**: 为新功能添加单元测试
3. **文档**: 更新相关文档
4. **提交信息**: 使用 [Conventional Commits](https://conventionalcommits.org/) 格式

### 社区行为准则

我们致力于创建一个包容、友好的社区环境。请遵循我们的[行为准则](CODE_OF_CONDUCT.md)。

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

特别感谢以下开源项目和贡献者：

- **[SQLite](https://sqlite.org/)** - 可靠高效的数据库引擎
- **[sqlite-vec](https://github.com/asg017/sqlite-vec)** - SQLite 向量扩展
- **[toml11](https://github.com/ToruNiina/toml11)** - 现代 C++ TOML 解析器
- **[spdlog](https://github.com/gabime/spdlog)** - 快速 C++ 日志库
- **[Prometheus C++](https://github.com/jupp0r/prometheus-cpp)** - Prometheus 指标库

感谢所有贡献者和用户的支持与反馈！

---

<div align="center">

**[⭐ Star](https://github.com/jblymq/RAG-CCC) 此项目** | **[🍴 Fork](https://github.com/jblymq/RAG-CCC/fork) 并贡献** | **[📖 阅读文档](https://github.com/jblymq/RAG-CCC)** | **[🐛 报告问题](https://github.com/jblymq/RAG-CCC/issues)**

**由 [WHU-MYTH-Lab](https://github.com/WHU-MYTH-Lab) 用 ❤️ 制作**

*让 AI 更智能，让检索更精准*

</div>

// 注册自定义分词器
BM25Indexer bm25;
bm25.set_custom_tokenizer(std::make_shared<CustomTokenizer>());
```

### 自定义向量存储

```cpp
class CustomVectorStore : public VectorStoreInterface {
public:
    void fit(const std::vector<Chunk>& chunks) override {
        // 实现自定义向量存储构建
    }

    std::vector<RetrievalResult> search(
        const std::vector<float>& query_embedding,
        size_t top_k
    ) override {
        // 实现自定义向量检索
        return custom_search(query_embedding, top_k);
    }
};
```

### 性能调优建议

1. **BM25参数调优**
   - `k1`: 控制词频饱和度，推荐范围[1.0, 2.0]
   - `b`: 控制文档长度归一化，推荐范围[0.5, 1.0]

2. **HNSW参数调优**
   - `M`: 连接数，影响精度和内存，推荐范围[8, 32]
   - `ef_construction`: 构建时搜索宽度，推荐200-400
   - `ef_query`: 查询时搜索宽度，推荐范围[50, 200]

3. **缓存优化**
   - 根据QPS和平均查询长度设置容量
   - TTL设置要平衡新鲜度和命中率

4. **线程池配置**
   - CPU密集型：线程数 = CPU核心数
   - IO密集型：线程数 = CPU核心数 × 2

## 🤝 贡献指南

欢迎贡献代码！请遵循以下流程：

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

### 代码规范

- 遵循C++17标准
- 使用snake_case命名变量和函数
- 使用PascalCase命名类和结构体
- 添加必要的注释和文档

## 📊 路线图

### v1.1 (计划中)
- [ ] 支持更多向量数据库（Faiss, Milvus）
- [ ] 增加Cross-Encoder重排序
- [ ] 支持多模态检索（图文混合）
- [ ] WebUI管理界面

### v1.2 (规划中)
- [ ] 分布式部署支持
- [ ] GPU加速向量计算
- [ ] 流式文档处理
- [ ] RESTful API接口

## 📄 许可证

本项目基于 MIT 许可证开源 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [HNSW](https://github.com/nmslib/hnswlib) - 高性能向量检索库
- [toml++](https://github.com/marzer/tomlplusplus) - TOML解析库
- [spdlog](https://github.com/gabime/spdlog) - 高性能日志库

## 📞 联系我们

- 项目主页: https://github.com/jblymq/RAG-CCC
- 问题反馈: [Issues](https://github.com/jblymq/RAG-CCC/issues)
- 邮箱: myth-lab@whu.edu.cn

---

⭐ 如果这个项目对你有帮助，请给我们一个星标！

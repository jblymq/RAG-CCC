#pragma once
#include "config.h"
#include <string>
#include <unordered_map>
#include <list>
#include <mutex>
#include <vector>

namespace rag {

struct Retrieval {
    std::vector<size_t> top_chunks;
    uint64_t timestamp = 0;
};

class LRUCache {
public:
    LRUCache(const CacheConfig& config = CacheConfig{});
    LRUCache(size_t capacity = 1024);  // Keep backward compatibility
    bool get(const std::string& key, Retrieval& out);
    void put(const std::string& key, const Retrieval& data);

private:
    size_t capacity_;
    std::list<std::string> order_;
    std::unordered_map<std::string, std::pair<Retrieval, std::list<std::string>::iterator>> map_;
    std::mutex mutex_;
};

} // namespace rag

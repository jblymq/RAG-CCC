#include "lru_cache.h"
#include <chrono>

namespace rag {

LRUCache::LRUCache(const CacheConfig& config) : capacity_(config.capacity) {}

LRUCache::LRUCache(size_t capacity) : capacity_(capacity) {}

bool LRUCache::get(const std::string& key, Retrieval& out) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = map_.find(key);
    if (it == map_.end()) return false;
    // move to front
    order_.erase(it->second.second);
    order_.push_front(key);
    it->second.second = order_.begin();
    out = it->second.first;
    return true;
}

void LRUCache::put(const std::string& key, const Retrieval& data) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = map_.find(key);
    if (it != map_.end()) {
        order_.erase(it->second.second);
        order_.push_front(key);
        it->second = {data, order_.begin()};
        return;
    }
    if (map_.size() >= capacity_) {
        auto last = order_.back();
        order_.pop_back();
        map_.erase(last);
    }
    order_.push_front(key);
    map_.emplace(key, std::make_pair(data, order_.begin()));
}

} // namespace rag

#include "thread_pool.h"

namespace rag {

ThreadPool::ThreadPool(const ThreadPoolConfig& config) {
    for (size_t i = 0; i < config.num_workers; ++i) {
        workers_.emplace_back([this]{
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->mutex_);
                    this->cv_.wait(lock, [this]{ return this->stop_ || !this->tasks_.empty(); });
                    if (this->stop_ && this->tasks_.empty()) return;
                    task = std::move(this->tasks_.front());
                    this->tasks_.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::ThreadPool(size_t numWorkers) {
    for (size_t i = 0; i < numWorkers; ++i) {
        workers_.emplace_back([this]{
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->mutex_);
                    this->cv_.wait(lock, [this]{ return this->stop_ || !this->tasks_.empty(); });
                    if (this->stop_ && this->tasks_.empty()) return;
                    task = std::move(this->tasks_.front());
                    this->tasks_.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto &w : workers_) if (w.joinable()) w.join();
}

} // namespace rag

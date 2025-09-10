#pragma once
#include "config.h"
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <future>
#include <condition_variable>
#include <functional>

namespace rag {

class ThreadPool {
public:
    explicit ThreadPool(const ThreadPoolConfig& config = ThreadPoolConfig{});
    explicit ThreadPool(size_t numWorkers);  // Keep backward compatibility
    ~ThreadPool();

    template<class F, class... Args>
    auto submit(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
        using RetT = decltype(f(args...));
        auto task = std::make_shared<std::packaged_task<RetT()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        std::future<RetT> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(mutex_);
            tasks_.emplace([task]{ (*task)(); });
        }
        cv_.notify_one();
        return res;
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
};

} // namespace rag

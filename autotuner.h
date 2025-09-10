#pragma once
#include "config.h"
#include <atomic>
#include <thread>
#include <functional>

namespace rag {

struct TunerParams {
    int ef = 50;
    int topK = 10;
};

class AutoTuner {
public:
    AutoTuner(const TunerConfig& config, std::function<double()> get_latency, std::function<double()> get_recall);
    AutoTuner(std::function<double()> get_latency, std::function<double()> get_recall);  // Keep backward compatibility
    ~AutoTuner();
    void start();
    void stop();
    TunerParams params();

private:
    std::atomic<bool> running_{false};
    std::thread worker_;
    std::function<double()> get_latency_;
    std::function<double()> get_recall_;
    TunerParams p_;
    TunerConfig config_;
};

} // namespace rag

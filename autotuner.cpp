#include "autotuner.h"
#include <chrono>
#include <iostream>

namespace rag {

AutoTuner::AutoTuner(const TunerConfig& config, std::function<double()> get_latency, std::function<double()> get_recall)
    : get_latency_(get_latency), get_recall_(get_recall), config_(config) {}

AutoTuner::AutoTuner(std::function<double()> get_latency, std::function<double()> get_recall)
    : get_latency_(get_latency), get_recall_(get_recall) {}

AutoTuner::~AutoTuner() { stop(); }

void AutoTuner::start() {
    if (running_) return;
    running_ = true;
    worker_ = std::thread([this]{
        while (running_) {
            double lat = get_latency_();
            double recall = get_recall_();
            if (lat > config_.latency_max_ms) {
                p_.ef = std::max(10, p_.ef - config_.ef_delta);
                p_.topK = std::max(1, p_.topK - config_.topk_delta);
            } else if (recall < config_.recall_min_pct) {
                p_.ef = std::min(500, p_.ef + config_.ef_delta);
                p_.topK = std::min(100, p_.topK + config_.topk_delta);
            }
            std::this_thread::sleep_for(std::chrono::seconds(config_.check_interval_seconds));
        }
    });
}

void AutoTuner::stop() {
    if (!running_) return;
    running_ = false;
    if (worker_.joinable()) worker_.join();
}

TunerParams AutoTuner::params() { return p_; }

} // namespace rag

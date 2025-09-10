#pragma once
#include <string>
#include <ctime>

namespace rag {

struct Chunk {
    std::string text;
    std::string doc_id;
    size_t seq_no = 0;
    std::string topic;
    std::string language;
    std::time_t created_at = std::time(nullptr);
};

} // namespace rag

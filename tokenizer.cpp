#include "tokenizer.h"
#include <sstream>
#include <iostream>

namespace rag {

Tokenizer::Tokenizer(const TokenizerConfig& config) : config_(config) {
    init_stopwords();

    // 初始化正则表达式 - 简化标点符号处理
    punctuation_regex_ = std::regex(R"([!"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~])");
    english_word_regex_ = std::regex(R"([a-zA-Z]+)");
}

void Tokenizer::init_stopwords() {
    // 英文停用词
    english_stopwords_ = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "were", "will", "with", "the", "this", "but", "they",
        "have", "had", "what", "said", "each", "which", "she", "do", "how",
        "their", "if", "up", "out", "many", "then", "them", "these", "so",
        "some", "her", "would", "make", "like", "into", "him", "time", "two",
        "more", "go", "no", "way", "could", "my", "than", "first", "been",
        "call", "who", "oil", "sit", "now", "find", "down", "day", "did",
        "get", "come", "made", "may", "part"
    };

    // 中文停用词
    chinese_stopwords_ = {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
        "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有",
        "看", "好", "自己", "这", "那", "它", "他", "她", "我们", "你们", "他们",
        "这个", "那个", "什么", "怎么", "为什么", "因为", "所以", "但是", "然后",
        "如果", "虽然", "可是", "而且", "或者", "比如", "关于", "对于", "根据",
        "按照", "除了", "包括", "特别", "尤其", "另外", "首先", "其次", "最后",
        "总之", "因此", "所以", "于是", "然而", "不过", "虽然", "尽管", "即使"
    };
}

Language Tokenizer::detect_language(const std::string& text) const {
    if (text.empty()) return Language::ENGLISH;

    int chinese_chars = 0;
    int english_chars = 0;
    int total_chars = 0;

    // 使用UTF-8字符遍历
    for (size_t i = 0; i < text.length(); ) {
        unsigned char c = text[i];

        if (c < 0x80) {
            // ASCII字符
            if (std::isalpha(c)) {
                english_chars++;
            }
            i++;
        } else if ((c & 0xE0) == 0xC0) {
            // 2字节UTF-8
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            // 3字节UTF-8，可能是中文
            if (i + 2 < text.length()) {
                // 简单的中文范围检测
                unsigned char c1 = text[i];
                unsigned char c2 = text[i + 1];
                unsigned char c3 = text[i + 2];

                // 中文字符的UTF-8编码范围
                if (c1 >= 0xE4 && c1 <= 0xE9) {
                    chinese_chars++;
                }
            }
            i += 3;
        } else if ((c & 0xF8) == 0xF0) {
            // 4字节UTF-8
            i += 4;
        } else {
            i++;
        }
        total_chars++;
    }

    if (total_chars == 0) return Language::ENGLISH;

    double chinese_ratio = static_cast<double>(chinese_chars) / total_chars;
    double english_ratio = static_cast<double>(english_chars) / total_chars;

    if (chinese_ratio > 0.3) {
        return english_ratio > 0.1 ? Language::MIXED : Language::CHINESE;
    } else if (english_ratio > 0.3) {
        return Language::ENGLISH;
    }

    return Language::MIXED;
}

std::vector<std::string> Tokenizer::tokenize(const std::string& text, Language lang) const {
    if (text.empty()) return {};

    // 自动检测语言
    Language detected_lang = (lang == Language::AUTO) ? detect_language(text) : lang;

    switch (detected_lang) {
        case Language::ENGLISH:
            return tokenize_english(text);
        case Language::CHINESE:
            return tokenize_chinese(text);
        case Language::MIXED:
            return tokenize_mixed(text);
        default:
            return tokenize_english(text);  // 默认使用英文处理
    }
}

std::vector<std::string> Tokenizer::tokenize_english(const std::string& text) const {
    std::string processed_text = clean_text(text);

    if (config_.lowercase) {
        processed_text = to_lowercase(processed_text);
    }

    if (config_.remove_punctuation) {
        processed_text = remove_punctuation(processed_text);
    }

    // 分词
    std::vector<std::string> tokens;
    std::istringstream iss(processed_text);
    std::string token;

    while (iss >> token) {
        if (token.length() >= static_cast<size_t>(config_.min_token_length) &&
            token.length() <= static_cast<size_t>(config_.max_token_length)) {
            tokens.push_back(token);
        }
    }

    return filter_tokens(tokens, Language::ENGLISH);
}

std::vector<std::string> Tokenizer::tokenize_chinese(const std::string& text) const {
    if (!config_.enable_chinese_segmentation) {
        // 简单按字符分割
        return simple_chinese_segmentation(text);
    }

    // 使用改进的中文分词
    return simple_chinese_segmentation(text);
}

std::vector<std::string> Tokenizer::simple_chinese_segmentation(const std::string& text) const {
    std::vector<std::string> tokens;
    std::string current_word;

    // 常用中文词汇词典（简化版）
    static const std::unordered_set<std::string> common_words = {
        "计算机", "人工智能", "机器学习", "深度学习", "神经网络", "算法", "数据",
        "分析", "处理", "系统", "技术", "方法", "模型", "训练", "预测", "优化",
        "自然语言", "图像识别", "语音识别", "推荐系统", "搜索引擎", "大数据",
        "云计算", "区块链", "物联网", "网络安全", "软件工程", "数据库",
        "编程语言", "开发", "应用", "平台", "框架", "工具", "服务", "产品",
        "用户", "客户", "市场", "商业", "企业", "公司", "团队", "项目",
        "管理", "运营", "策略", "规划", "设计", "创新", "研究", "开发"
    };

    for (size_t i = 0; i < text.length(); ) {
        unsigned char c = text[i];

        if (c < 0x80) {
            // ASCII字符
            if (std::isalnum(c)) {
                current_word += c;
            } else if (!current_word.empty()) {
                if (current_word.length() >= static_cast<size_t>(config_.min_token_length)) {
                    tokens.push_back(current_word);
                }
                current_word.clear();
            }
            i++;
        } else if ((c & 0xF0) == 0xE0) {
            // 3字节UTF-8字符（中文）
            if (i + 2 < text.length()) {
                std::string char_str = text.substr(i, 3);

                // 尝试最长匹配
                bool found_word = false;
                for (int len = 4; len >= 2 && !found_word; len--) {
                    if (i + len * 3 <= text.length()) {
                        std::string word = text.substr(i, len * 3);
                        if (common_words.find(word) != common_words.end()) {
                            if (!current_word.empty()) {
                                tokens.push_back(current_word);
                                current_word.clear();
                            }
                            tokens.push_back(word);
                            i += len * 3;
                            found_word = true;
                        }
                    }
                }

                if (!found_word) {
                    if (config_.keep_single_char) {
                        if (!current_word.empty()) {
                            tokens.push_back(current_word);
                            current_word.clear();
                        }
                        tokens.push_back(char_str);
                    } else {
                        current_word += char_str;
                    }
                    i += 3;
                }
            } else {
                i++;
            }
        } else {
            // 其他字符，跳过
            if (!current_word.empty()) {
                tokens.push_back(current_word);
                current_word.clear();
            }
            i++;
        }
    }

    if (!current_word.empty()) {
        tokens.push_back(current_word);
    }

    return filter_tokens(tokens, Language::CHINESE);
}

std::vector<std::string> Tokenizer::tokenize_mixed(const std::string& text) const {
    std::vector<std::string> all_tokens;
    std::string current_segment;
    Language current_lang = Language::ENGLISH;

    for (size_t i = 0; i < text.length(); ) {
        unsigned char c = text[i];

        if (c < 0x80) {
            // ASCII字符
            if (current_lang == Language::CHINESE && !current_segment.empty()) {
                // 切换到英文前，先处理中文片段
                auto chinese_tokens = tokenize_chinese(current_segment);
                all_tokens.insert(all_tokens.end(), chinese_tokens.begin(), chinese_tokens.end());
                current_segment.clear();
            }
            current_lang = Language::ENGLISH;
            current_segment += c;
            i++;
        } else if ((c & 0xF0) == 0xE0) {
            // 可能的中文字符
            if (current_lang == Language::ENGLISH && !current_segment.empty()) {
                // 切换到中文前，先处理英文片段
                auto english_tokens = tokenize_english(current_segment);
                all_tokens.insert(all_tokens.end(), english_tokens.begin(), english_tokens.end());
                current_segment.clear();
            }
            current_lang = Language::CHINESE;
            if (i + 2 < text.length()) {
                current_segment += text.substr(i, 3);
                i += 3;
            } else {
                i++;
            }
        } else {
            i++;
        }
    }

    // 处理最后的片段
    if (!current_segment.empty()) {
        if (current_lang == Language::ENGLISH) {
            auto english_tokens = tokenize_english(current_segment);
            all_tokens.insert(all_tokens.end(), english_tokens.begin(), english_tokens.end());
        } else {
            auto chinese_tokens = tokenize_chinese(current_segment);
            all_tokens.insert(all_tokens.end(), chinese_tokens.begin(), chinese_tokens.end());
        }
    }

    return all_tokens;
}

std::string Tokenizer::clean_text(const std::string& text) const {
    // 简单的文本清理：移除多余空格
    std::string cleaned = text;

    // 替换多个连续空格为单个空格
    std::regex multiple_spaces(R"(\s+)");
    cleaned = std::regex_replace(cleaned, multiple_spaces, " ");

    // 去除首尾空格
    size_t start = cleaned.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";

    size_t end = cleaned.find_last_not_of(" \t\r\n");
    return cleaned.substr(start, end - start + 1);
}

std::string Tokenizer::to_lowercase(const std::string& text) const {
    std::string result = text;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

std::string Tokenizer::remove_punctuation(const std::string& text) const {
    return std::regex_replace(text, punctuation_regex_, " ");
}

bool Tokenizer::is_stopword(const std::string& word, Language lang) const {
    if (!config_.filter_stopwords) return false;

    switch (lang) {
        case Language::ENGLISH:
            return english_stopwords_.find(word) != english_stopwords_.end();
        case Language::CHINESE:
            return chinese_stopwords_.find(word) != chinese_stopwords_.end();
        default:
            return english_stopwords_.find(word) != english_stopwords_.end() ||
                   chinese_stopwords_.find(word) != chinese_stopwords_.end();
    }
}

std::vector<std::string> Tokenizer::filter_tokens(const std::vector<std::string>& tokens, Language lang) const {
    std::vector<std::string> filtered;

    for (const auto& token : tokens) {
        if (!token.empty() && !is_stopword(token, lang)) {
            filtered.push_back(token);
        }
    }

    return filtered;
}

std::vector<std::vector<std::string>> Tokenizer::tokenize_batch(const std::vector<std::string>& texts, Language lang) const {
    std::vector<std::vector<std::string>> results;
    results.reserve(texts.size());

    for (const auto& text : texts) {
        results.push_back(tokenize(text, lang));
    }

    return results;
}

std::unordered_map<std::string, int> Tokenizer::get_token_counts(const std::string& text, Language lang) const {
    auto tokens = tokenize(text, lang);
    std::unordered_map<std::string, int> counts;

    for (const auto& token : tokens) {
        counts[token]++;
    }

    return counts;
}

void Tokenizer::set_config(const TokenizerConfig& config) {
    config_ = config;
    init_stopwords();  // 重新初始化
}

void Tokenizer::add_stopwords(const std::vector<std::string>& words, Language lang) {
    auto& stopwords = (lang == Language::CHINESE) ? chinese_stopwords_ : english_stopwords_;
    for (const auto& word : words) {
        stopwords.insert(word);
    }
}

void Tokenizer::remove_stopwords(const std::vector<std::string>& words, Language lang) {
    auto& stopwords = (lang == Language::CHINESE) ? chinese_stopwords_ : english_stopwords_;
    for (const auto& word : words) {
        stopwords.erase(word);
    }
}

std::vector<Language> Tokenizer::get_supported_languages() const {
    return {Language::AUTO, Language::ENGLISH, Language::CHINESE, Language::MIXED};
}

std::string Tokenizer::preprocess_text(const std::string& text, Language lang) const {
    std::string result = clean_text(text);

    if (config_.lowercase) {
        result = to_lowercase(result);
    }

    if (config_.remove_punctuation) {
        result = remove_punctuation(result);
    }

    return result;
}

// 便捷函数实现
namespace tokenizer_utils {

std::vector<std::string> quick_english_tokenize(const std::string& text) {
    TokenizerConfig config;
    config.language = Language::ENGLISH;
    Tokenizer tokenizer(config);
    return tokenizer.tokenize(text);
}

std::vector<std::string> quick_chinese_tokenize(const std::string& text) {
    TokenizerConfig config;
    config.language = Language::CHINESE;
    Tokenizer tokenizer(config);
    return tokenizer.tokenize(text);
}

Language detect_primary_language(const std::string& text) {
    Tokenizer tokenizer;
    return tokenizer.detect_language(text);
}

std::vector<std::string> get_default_english_stopwords() {
    return {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "were", "will", "with"
    };
}

std::vector<std::string> get_default_chinese_stopwords() {
    return {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一"
    };
}

} // namespace tokenizer_utils

} // namespace rag

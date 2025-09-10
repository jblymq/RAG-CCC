#ifndef RAG_TOKENIZER_H
#define RAG_TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <regex>
#include <algorithm>
#include <locale>
#include <codecvt>

namespace rag {

// 语言类型枚举
enum class Language {
    AUTO,       // 自动检测
    ENGLISH,    // 英文
    CHINESE,    // 中文
    MIXED       // 中英混合
};

// Tokenizer配置
struct TokenizerConfig {
    Language language = Language::AUTO;
    bool lowercase = true;                      // 是否转小写
    bool remove_punctuation = true;            // 是否移除标点符号
    bool filter_stopwords = true;              // 是否过滤停用词
    bool enable_stemming = false;              // 是否启用词干提取（暂未实现）
    int min_token_length = 1;                  // 最小token长度
    int max_token_length = 50;                 // 最大token长度

    // 中文分词配置
    bool enable_chinese_segmentation = true;   // 是否启用中文分词
    bool keep_single_char = false;             // 是否保留单字符（对中文）
};

// 改进的Tokenizer类
class Tokenizer {
private:
    TokenizerConfig config_;
    std::unordered_set<std::string> english_stopwords_;
    std::unordered_set<std::string> chinese_stopwords_;
    std::regex punctuation_regex_;
    std::regex english_word_regex_;
    std::regex chinese_char_regex_;

    // 初始化停用词
    void init_stopwords();

    // 英文预处理
    std::vector<std::string> tokenize_english(const std::string& text) const;

    // 中文分词
    std::vector<std::string> tokenize_chinese(const std::string& text) const;

    // 简单的中文分词算法（基于字符和常见词）
    std::vector<std::string> simple_chinese_segmentation(const std::string& text) const;

    // 混合语言处理
    std::vector<std::string> tokenize_mixed(const std::string& text) const;

    // 文本清理
    std::string clean_text(const std::string& text) const;

    // 是否为中文字符
    bool is_chinese_char(const std::string& ch) const;

    // 是否为英文字符
    bool is_english_char(char ch) const;

    // 转换为小写
    std::string to_lowercase(const std::string& text) const;

    // 移除标点符号
    std::string remove_punctuation(const std::string& text) const;

    // 是否为停用词
    bool is_stopword(const std::string& word, Language lang) const;

    // 过滤token
    std::vector<std::string> filter_tokens(const std::vector<std::string>& tokens, Language lang) const;

public:
    // 构造函数
    explicit Tokenizer(const TokenizerConfig& config = TokenizerConfig());

    // 语言检测
    Language detect_language(const std::string& text) const;

    // 主要的分词接口
    std::vector<std::string> tokenize(const std::string& text, Language lang = Language::AUTO) const;

    // 批量分词
    std::vector<std::vector<std::string>> tokenize_batch(const std::vector<std::string>& texts, Language lang = Language::AUTO) const;

    // 获取词汇统计
    std::unordered_map<std::string, int> get_token_counts(const std::string& text, Language lang = Language::AUTO) const;

    // 配置相关
    void set_config(const TokenizerConfig& config);
    const TokenizerConfig& get_config() const { return config_; }

    // 添加自定义停用词
    void add_stopwords(const std::vector<std::string>& words, Language lang = Language::ENGLISH);

    // 移除停用词
    void remove_stopwords(const std::vector<std::string>& words, Language lang = Language::ENGLISH);

    // 获取支持的语言
    std::vector<Language> get_supported_languages() const;

    // 预处理文本（不分词，只做清理）
    std::string preprocess_text(const std::string& text, Language lang = Language::AUTO) const;
};

// 便捷函数
namespace tokenizer_utils {
    // 快速英文分词
    std::vector<std::string> quick_english_tokenize(const std::string& text);

    // 快速中文分词
    std::vector<std::string> quick_chinese_tokenize(const std::string& text);

    // 检测文本主要语言
    Language detect_primary_language(const std::string& text);

    // 获取默认停用词列表
    std::vector<std::string> get_default_english_stopwords();
    std::vector<std::string> get_default_chinese_stopwords();
}

} // namespace rag

#endif // RAG_TOKENIZER_H

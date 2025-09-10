cmake_minimum_required(VERSION 3.10)
project(humanus_rag_example)

set(CMAKE_CXX_STANDARD 17)

# 查找 SQLite3
find_package(PkgConfig REQUIRED)
pkg_check_modules(SQLITE3 REQUIRED sqlite3)

# 综合 RAG 系统示例（包含内存和SQLite功能）
add_executable(rag_example main.cpp
    ../bm25.cpp
    ../fusion_retriever.cpp
    ../sqlite_db.cpp
    ../sqlite_retriever.cpp
    ../lru_cache.cpp
    ../thread_pool.cpp
    ../autotuner.cpp
    ../config.cpp
    ../tokenizer.cpp)

target_include_directories(rag_example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${SQLITE3_INCLUDE_DIRS})

target_link_libraries(rag_example ${SQLITE3_LIBRARIES} pthread)

# 混合 RAG 系统演示（内存+SQLite结合）
add_executable(hybrid_rag_demo hybrid_rag_demo.cpp
    ../bm25.cpp
    ../fusion_retriever.cpp
    ../sqlite_db.cpp
    ../sqlite_retriever.cpp
    ../lru_cache.cpp
    ../thread_pool.cpp
    ../autotuner.cpp
    ../config.cpp
    ../tokenizer.cpp)

target_include_directories(hybrid_rag_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${SQLITE3_INCLUDE_DIRS})

target_link_libraries(hybrid_rag_demo ${SQLITE3_LIBRARIES} pthread)

target_compile_options(rag_example PRIVATE ${SQLITE3_CFLAGS_OTHER})
target_compile_options(hybrid_rag_demo PRIVATE ${SQLITE3_CFLAGS_OTHER})

# 拷贝配置文件到生成文件同级目录
set(CONFIG_SOURCE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../config/rag_config.toml")
set(CONFIG_DEST_PATH "${CMAKE_CURRENT_BINARY_DIR}/rag_config.toml")

# 为 rag_example 添加配置文件拷贝
add_custom_command(TARGET rag_example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CONFIG_SOURCE_PATH}
    ${CONFIG_DEST_PATH}
    COMMENT "Copying rag_config.toml to build directory for rag_example"
)

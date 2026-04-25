#pragma once

/*
 * ooverlap/system/logging.h
 *
 * Compile-time logging.
 *
 * Usage:
 *   OOVERLAP_LOG_ERROR("bad thing: %d\n", x);
 *   OOVERLAP_LOG_DEBUG("ptr=%p\n", ptr);
 *
 * Build examples:
 *   quiet/default: no extra flag, defaults to ERROR level
 *   debug:         -DOOVERLAP_LOG_LEVEL=5
 *   trace:         -DOOVERLAP_LOG_LEVEL=6
 *   disable all:   -DOOVERLAP_LOG_LEVEL=0
 */

#include <cstdio>
#include <cstdlib>

#define OOVERLAP_LOG_LEVEL_NONE  0
#define OOVERLAP_LOG_LEVEL_FATAL 1
#define OOVERLAP_LOG_LEVEL_ERROR 2
#define OOVERLAP_LOG_LEVEL_WARN  3
#define OOVERLAP_LOG_LEVEL_INFO  4
#define OOVERLAP_LOG_LEVEL_DEBUG 5
#define OOVERLAP_LOG_LEVEL_TRACE 6

#ifndef OOVERLAP_LOG_LEVEL
#define OOVERLAP_LOG_LEVEL OOVERLAP_LOG_LEVEL_ERROR
#endif

#ifndef OOVERLAP_LOG_USE_COLOR
#define OOVERLAP_LOG_USE_COLOR 1
#endif

#if OOVERLAP_LOG_USE_COLOR
#define OOVERLAP_LOG_COLOR_FATAL "\033[0;31m"
#define OOVERLAP_LOG_COLOR_ERROR "\033[0;91m"
#define OOVERLAP_LOG_COLOR_WARN  "\033[0;93m"
#define OOVERLAP_LOG_COLOR_INFO  "\033[0;94m"
#define OOVERLAP_LOG_COLOR_DEBUG "\033[0;96m"
#define OOVERLAP_LOG_COLOR_TRACE "\033[0;90m"
#define OOVERLAP_LOG_COLOR_RESET "\033[0m"
#else
#define OOVERLAP_LOG_COLOR_FATAL ""
#define OOVERLAP_LOG_COLOR_ERROR ""
#define OOVERLAP_LOG_COLOR_WARN  ""
#define OOVERLAP_LOG_COLOR_INFO  ""
#define OOVERLAP_LOG_COLOR_DEBUG ""
#define OOVERLAP_LOG_COLOR_TRACE ""
#define OOVERLAP_LOG_COLOR_RESET ""
#endif

#define OOVERLAP_LOG_IMPL(stream, color, level_name, fmt, ...)                 \
    do {                                                                       \
        std::fprintf(                                                          \
            (stream),                                                          \
            "%s[ooverlap][%s][%s:%d:%s] " fmt "%s",                            \
            (color),                                                           \
            (level_name),                                                      \
            __FILE__,                                                          \
            __LINE__,                                                          \
            __func__,                                                          \
            ##__VA_ARGS__,                                                     \
            OOVERLAP_LOG_COLOR_RESET);                                         \
        std::fflush((stream));                                                 \
    } while (0)

#if OOVERLAP_LOG_LEVEL >= OOVERLAP_LOG_LEVEL_FATAL
#define OOVERLAP_LOG_FATAL(error_code, fmt, ...)                               \
    do {                                                                       \
        OOVERLAP_LOG_IMPL(                                                     \
            stderr,                                                            \
            OOVERLAP_LOG_COLOR_FATAL,                                          \
            "fatal",                                                           \
            fmt,                                                               \
            ##__VA_ARGS__);                                                    \
        std::exit((error_code));                                               \
    } while (0)
#else
#define OOVERLAP_LOG_FATAL(error_code, fmt, ...)                               \
    do {                                                                       \
        std::exit((error_code));                                               \
    } while (0)
#endif

#if OOVERLAP_LOG_LEVEL >= OOVERLAP_LOG_LEVEL_ERROR
#define OOVERLAP_LOG_ERROR(fmt, ...)                                           \
    OOVERLAP_LOG_IMPL(                                                         \
        stderr,                                                                \
        OOVERLAP_LOG_COLOR_ERROR,                                              \
        "error",                                                               \
        fmt,                                                                   \
        ##__VA_ARGS__)
#else
#define OOVERLAP_LOG_ERROR(fmt, ...) do {} while (0)
#endif

#if OOVERLAP_LOG_LEVEL >= OOVERLAP_LOG_LEVEL_WARN
#define OOVERLAP_LOG_WARN(fmt, ...)                                            \
    OOVERLAP_LOG_IMPL(                                                         \
        stderr,                                                                \
        OOVERLAP_LOG_COLOR_WARN,                                               \
        "warn",                                                                \
        fmt,                                                                   \
        ##__VA_ARGS__)
#else
#define OOVERLAP_LOG_WARN(fmt, ...) do {} while (0)
#endif

#if OOVERLAP_LOG_LEVEL >= OOVERLAP_LOG_LEVEL_INFO
#define OOVERLAP_LOG_INFO(fmt, ...)                                            \
    OOVERLAP_LOG_IMPL(                                                         \
        stdout,                                                                \
        OOVERLAP_LOG_COLOR_INFO,                                               \
        "info",                                                                \
        fmt,                                                                   \
        ##__VA_ARGS__)
#else
#define OOVERLAP_LOG_INFO(fmt, ...) do {} while (0)
#endif

#if OOVERLAP_LOG_LEVEL >= OOVERLAP_LOG_LEVEL_DEBUG
#define OOVERLAP_LOG_DEBUG(fmt, ...)                                           \
    OOVERLAP_LOG_IMPL(                                                         \
        stdout,                                                                \
        OOVERLAP_LOG_COLOR_DEBUG,                                              \
        "debug",                                                               \
        fmt,                                                                   \
        ##__VA_ARGS__)
#else
#define OOVERLAP_LOG_DEBUG(fmt, ...) do {} while (0)
#endif

#if OOVERLAP_LOG_LEVEL >= OOVERLAP_LOG_LEVEL_TRACE
#define OOVERLAP_LOG_TRACE(fmt, ...)                                           \
    OOVERLAP_LOG_IMPL(                                                         \
        stdout,                                                                \
        OOVERLAP_LOG_COLOR_TRACE,                                              \
        "trace",                                                               \
        fmt,                                                                   \
        ##__VA_ARGS__)
#else
#define OOVERLAP_LOG_TRACE(fmt, ...) do {} while (0)
#endif

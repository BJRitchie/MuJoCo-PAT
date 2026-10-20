#pragma once
/*! The malloc interposer behind alloc_counter.h. Include this in exactly ONE
 *  translation unit of a test executable: it defines the counters and replaces
 *  malloc, calloc and realloc for the whole process. */
#include "pat_rt/testing/alloc_counter.h"

extern "C" {
void* __libc_malloc(std::size_t);
void* __libc_calloc(std::size_t, std::size_t);
void* __libc_realloc(void*, std::size_t);
}

namespace pat_rt {
namespace testing {

volatile bool counting = false;
long          count    = 0;
volatile bool tracing  = false;
void*         sites[kMaxSites];
long          n_sites  = 0;

namespace {
inline void note(void* ret_addr) {
    ++count;
    if (tracing && n_sites < kMaxSites) sites[n_sites++] = ret_addr;
}
}  // namespace

}  // namespace testing
}  // namespace pat_rt

extern "C" void* malloc(std::size_t n) {
    if (pat_rt::testing::counting) pat_rt::testing::note(__builtin_return_address(0));
    return __libc_malloc(n);
}
extern "C" void* calloc(std::size_t n, std::size_t sz) {
    if (pat_rt::testing::counting) pat_rt::testing::note(__builtin_return_address(0));
    return __libc_calloc(n, sz);
}
extern "C" void* realloc(void* p, std::size_t n) {
    if (pat_rt::testing::counting) pat_rt::testing::note(__builtin_return_address(0));
    return __libc_realloc(p, n);
}

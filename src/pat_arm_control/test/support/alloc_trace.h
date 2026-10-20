#pragma once
/*! Allocation attribution for the studies: pat_rt's allocation counter plus the
 *  call site of every allocation, grouped by the shared object that made it. */
#include <dlfcn.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "pat_rt/testing/alloc_counter.h"

namespace pat_arm_control {
namespace test_support {

/*! AllocScope plus the call site of every allocation, grouped by the shared
 *  object that made it. Answers "whose allocation is this" rather than just how
 *  many, which is the difference between a site this package can remove and one
 *  inside a precompiled dependency.
 *
 *  One frame is enough for the question the studies ask, because acados, HPIPM
 *  and BLASFEO call malloc directly, so frame 0 names their shared object. A
 *  caller reaching malloc through libstdc++'s operator new resolves to libstdc++
 *  and is reported as such rather than guessed at. */
struct AllocTrace {
    AllocTrace() {
        namespace rt = pat_rt::testing;
        rt::n_sites  = 0;
        rt::count    = 0;
        rt::tracing  = true;
        rt::counting = true;
    }
    ~AllocTrace() { stop(); }

    //! End the measured region. Must be called before reading the results:
    //! byObject() allocates, and left counting it would tally itself.
    void stop() const {
        pat_rt::testing::counting = false;
        pat_rt::testing::tracing  = false;
    }
    long n() const { return pat_rt::testing::count; }

    //! Allocation count per originating object, most frequent first. Call stop()
    //! first; this allocates, and dladdr does too.
    std::vector<std::pair<std::string, long>> byObject() const {
        namespace rt = pat_rt::testing;
        std::map<std::string, long> tally;
        for (long i = 0; i < rt::n_sites; ++i) {
            Dl_info info{};
            const bool ok = dladdr(rt::sites[i], &info) != 0;
            const char* path = (ok && info.dli_fname) ? info.dli_fname : "<unresolved>";
            const char* base = std::strrchr(path, '/');
            ++tally[base ? base + 1 : path];
        }
        std::vector<std::pair<std::string, long>> rows(tally.begin(), tally.end());
        std::sort(rows.begin(), rows.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        return rows;
    }

    //! Per distinct call site: the object, the call's offset within it, and how
    //! many allocations came from there, most frequent first. The offset is what
    //! addr2line resolves to a file and line (the objects here are all built
    //! with debug info), which is what makes a site actionable rather than just
    //! attributed. Same precondition as byObject().
    struct Site { std::string object; std::uintptr_t offset; long count; };
    std::vector<Site> bySite() const {
        namespace rt = pat_rt::testing;
        std::map<std::pair<std::string, std::uintptr_t>, long> tally;
        for (long i = 0; i < rt::n_sites; ++i) {
            Dl_info info{};
            std::string object = "<unresolved>";
            std::uintptr_t offset = reinterpret_cast<std::uintptr_t>(rt::sites[i]);
            if (dladdr(rt::sites[i], &info) != 0 && info.dli_fname) {
                const char* base = std::strrchr(info.dli_fname, '/');
                object = base ? base + 1 : info.dli_fname;
                offset -= reinterpret_cast<std::uintptr_t>(info.dli_fbase);
            }
            ++tally[{object, offset}];
        }
        std::vector<Site> rows;
        rows.reserve(tally.size());
        for (const auto& [key, n] : tally) rows.push_back({key.first, key.second, n});
        std::sort(rows.begin(), rows.end(),
                  [](const Site& a, const Site& b) { return a.count > b.count; });
        return rows;
    }

    //! Whether the trace buffer overflowed, in which case byObject() is a
    //! truncated sample and the proportions are still meaningful but the counts
    //! are not.
    bool truncated() const { return pat_rt::testing::n_sites >= pat_rt::testing::kMaxSites; }
};

}  // namespace test_support
}  // namespace pat_arm_control

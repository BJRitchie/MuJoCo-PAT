#pragma once
/*! Heap-allocation counting for tests of realtime code.
 *
 *  Interposes glibc's malloc family so a test can count the heap operations of
 *  a region of code -- a control tick, a mailbox read. Defined in the test
 *  executable, so calls from libstdc++, Eigen, MuJoCo and acados/BLASFEO all
 *  land in it: each reaches malloc through the PLT, which resolves against the
 *  executable first. Counting malloc rather than operator new is deliberate --
 *  operator new is implemented on top of malloc, and acados/BLASFEO call malloc
 *  directly.
 *
 *  glibc-specific (__libc_* are its internal entry points) and deliberately not
 *  counting posix_memalign/aligned_alloc, which have no stable __libc_ pair. On
 *  x86-64 glibc, Eigen's aligned allocator is plain malloc (malloc already
 *  returns 16-byte-aligned blocks), so the hot path is covered.
 *
 *  This header declares the counters and AllocScope. The interposer itself is
 *  in alloc_counter_impl.h, which exactly ONE translation unit of each test
 *  executable includes. Test code only, never a library.
 */
#include <cstddef>

namespace pat_rt {
namespace testing {

// Zero-initialised statics with no dynamic initialiser: allocations happen
// during static init, before any constructor could have run.
extern volatile bool counting;
extern long count;

// Call-site attribution, off unless a test turns it on. Only the IMMEDIATE
// caller's return address is kept: frame 0 is the one address available
// without a frame pointer, and resolving further would need backtrace(), which
// takes the loader lock and can itself allocate -- inside malloc that
// deadlocks. Addresses only; symbolise them after counting stops, since dladdr
// allocates.
constexpr int kMaxSites = 4096;
extern volatile bool tracing;
extern void* sites[kMaxSites];
extern long n_sites;

/*! Counts allocations for as long as it is in scope. Read `n` after it closes:
 *  gtest's own assertion machinery allocates. */
struct AllocScope {
    AllocScope() { count = 0; counting = true; }
    ~AllocScope() { counting = false; }
    long n() const { return count; }
};

}  // namespace testing
}  // namespace pat_rt

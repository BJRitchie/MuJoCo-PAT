#pragma once

#include <atomic>
#include <cstdint>
#include <type_traits>

namespace pat_arm_control {

/*! Single-producer snapshot hand-off for a small trivially copyable value,
 *  where no consumer may block.
 *
 *  The control loop's problem is not throughput, it is that it cannot afford to
 *  wait: a thread at SCHED_FIFO blocking on a lock held by a normal-priority
 *  subscription callback is priority inversion, and the control thread then waits
 *  for the scheduler to get around to the callback. A mailbox removes the shared
 *  lock entirely -- the producer writes a whole new value, the consumer copies
 *  whichever value is current, and neither waits for the other.
 *
 *  A seqlock: `seq_` is odd while a write is in progress and even when the value
 *  is stable, so a reader that sees the same even sequence before and after its
 *  copy knows no write overlapped it. The reader retries a bounded number of
 *  times and then reports failure, so it cannot spin while the producer is
 *  descheduled mid-write -- the caller keeps the value it already had, which for
 *  a control loop means one tick of extra staleness rather than a missed
 *  deadline.
 *
 *  Reads and writes of `data_` are not themselves atomic, so a reader's copy can
 *  overlap a writer's and observe a torn value; the sequence check is what
 *  detects that and forces the retry, and the fences are what stop the compiler
 *  and the CPU reordering the copy outside the checked region. This is the
 *  standard seqlock construction (it is how the kernel's own `seqcount` works)
 *  and it relies on a torn read being discarded rather than never happening.
 *
 *  Both sides are allocation-free and syscall-free.
 *
 *  Only ONE thread may call write(): two producers would interleave their
 *  sequence bumps and could leave `seq_` permanently odd, at which point every
 *  reader fails forever. ANY NUMBER of threads may call read() concurrently --
 *  it is const, keeps no state between calls, and each reader's retry loop is
 *  independent, so readers neither see nor affect one another. That is relied
 *  on: the observer thread reads the same joint and base mailboxes the control
 *  thread reads, rather than being given duplicates of them.
 */
template <typename T>
class SeqlockMailbox {
    static_assert(std::is_trivially_copyable_v<T>,
                  "SeqlockMailbox copies T with a plain assignment while a reader "
                  "may be copying it: T must have no indirection, no allocation "
                  "and no non-trivial copy. Use a fixed-size array, not a vector.");

public:
    //! Publish a new value. Wait-free; never blocks a reader.
    void write(const T& value) {
        const std::uint32_t s = seq_.load(std::memory_order_relaxed);
        // Odd: a write is in progress, so any reader will retry or give up.
        seq_.store(s + 1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        data_ = value;
        // Publish the value before the sequence that advertises it as stable.
        std::atomic_thread_fence(std::memory_order_release);
        seq_.store(s + 2, std::memory_order_relaxed);
    }

    /*! Copy the current value into `out`. Returns false if nothing has been
     *  written yet, or if a write kept overlapping the read -- in which case
     *  `out` is untouched and the caller should keep using its previous value. */
    bool read(T& out) const {
        for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
            const std::uint32_t before = seq_.load(std::memory_order_relaxed);
            if (before == 0) return false;          // never written
            if (before & 1u) continue;              // writer is mid-update
            std::atomic_thread_fence(std::memory_order_acquire);
            out = data_;
            std::atomic_thread_fence(std::memory_order_acquire);
            if (seq_.load(std::memory_order_relaxed) == before) return true;
        }
        return false;
    }

    //! Whether anything has ever been written. Cheap enough to gate a warning on.
    bool written() const { return seq_.load(std::memory_order_acquire) != 0; }

private:
    //! Enough retries to ride out a write that started just before this read, few
    //! enough that a producer descheduled mid-write cannot stall the consumer.
    static constexpr int kMaxAttempts = 4;

    T                          data_{};
    std::atomic<std::uint32_t> seq_{0};
};

/*! Single-producer / single-consumer queue of fixed capacity, for handing values
 *  OUT of the control loop without losing any of them.
 *
 *  The mailbox above keeps only the newest value, which is right for a
 *  measurement -- a tick wants the current state, not a backlog. It is wrong for
 *  a command: a torque the consumer never publishes leaves the actuator holding
 *  the previous one for an extra period, so every tick's output has to survive
 *  until something publishes it. Hence a queue.
 *
 *  Producer and consumer each touch one index, published with release and read
 *  with acquire, so neither takes a lock and the producer never waits. A full
 *  queue drops the oldest-unread value rather than blocking the control loop:
 *  `push` reports the drop so the consumer can count it, which makes a drain that
 *  cannot keep up visible instead of silent.
 *
 *  Capacity must be a power of two, so the wrap is a mask rather than a modulo.
 */
template <typename T, std::size_t Capacity>
class SpscRing {
    static_assert(std::is_trivially_copyable_v<T>,
                  "SpscRing copies T by plain assignment: no indirection, no "
                  "allocation. Use a fixed-size array, not a vector.");
    static_assert(Capacity >= 2 && (Capacity & (Capacity - 1)) == 0,
                  "Capacity must be a power of two and at least 2");

public:
    //! Enqueue. Returns false if the queue was full and this value was dropped.
    bool push(const T& value) {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        if (head - tail >= Capacity) return false;         // full: caller counts it
        slots_[head & kMask] = value;
        head_.store(head + 1, std::memory_order_release);  // publish after the write
        return true;
    }

    //! Dequeue the oldest value. Returns false if the queue was empty.
    bool pop(T& out) {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) return false;   // empty
        out = slots_[tail & kMask];
        tail_.store(tail + 1, std::memory_order_release);  // free the slot after
        return true;
    }

    //! How many values are waiting. A consumer's sanity check, not a lock.
    std::size_t size() const {
        return head_.load(std::memory_order_acquire) -
               tail_.load(std::memory_order_acquire);
    }

private:
    static constexpr std::size_t kMask = Capacity - 1;

    T                         slots_[Capacity]{};
    std::atomic<std::size_t>  head_{0};   //!< producer's cursor
    std::atomic<std::size_t>  tail_{0};   //!< consumer's cursor
};

}  // namespace pat_arm_control

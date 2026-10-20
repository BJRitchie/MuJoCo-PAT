/*! Tests for the realtime hand-off primitives: SeqlockMailbox (a control loop's
 *  inputs) and SpscRing (its outputs). Every periodic-loop thread in the
 *  workspace relies on them, so the properties pinned here -- no torn read, no
 *  starved reader, no lost or reordered value, no allocation -- are contracts,
 *  not implementation details.
 */
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>

#include "pat_rt/rt_mailbox.h"
#include "pat_rt/testing/alloc_counter.h"

using pat_rt::SeqlockMailbox;
using pat_rt::SpscRing;
using pat_rt::testing::AllocScope;

namespace {

// --- The control loop's input hand-off ------------------------------------ //

//! A payload big enough that a copy cannot be one instruction, so a torn read is
//! reachable if the sequence check is wrong. The invariant is that every element
//! holds the same value: any mixture of two writes is a tear.
struct Payload {
    static constexpr int kN = 64;
    std::uint64_t v[kN];

    void fill(std::uint64_t x) { for (int i = 0; i < kN; ++i) v[i] = x; }
    bool uniform() const {
        for (int i = 1; i < kN; ++i) if (v[i] != v[0]) return false;
        return true;
    }
};

TEST(RtMailbox, ReadsNothingBeforeAnyWrite)
{
    SeqlockMailbox<Payload> mail;
    EXPECT_FALSE(mail.written());
    Payload out;
    out.fill(7);
    EXPECT_FALSE(mail.read(out));
    // A failed read must leave the caller's value alone -- the control loop
    // relies on that to keep running on its previous measurement.
    EXPECT_EQ(out.v[0], 7u);
}

TEST(RtMailbox, ReadsTheMostRecentWrite)
{
    SeqlockMailbox<Payload> mail;
    Payload in, out;
    for (std::uint64_t x : {1u, 2u, 99u}) {
        in.fill(x);
        mail.write(in);
        ASSERT_TRUE(mail.read(out));
        EXPECT_EQ(out.v[0], x);
        EXPECT_TRUE(out.uniform());
    }
    EXPECT_TRUE(mail.written());
    // Reading twice without an intervening write yields the same value, rather
    // than consuming it: the control loop reads every tick, not every message.
    ASSERT_TRUE(mail.read(out));
    EXPECT_EQ(out.v[0], 99u);
}

/*! The property that matters under contention: a reader never observes a mixture
 *  of two writes. A torn value reaching the control loop would be a state vector
 *  whose base and joints came from different instants, which no amount of
 *  downstream checking would catch. */
TEST(RtMailbox, ConcurrentReadsAreNeverTorn)
{
    SeqlockMailbox<Payload> mail;
    std::atomic<bool> stop{false};
    Payload seed;
    seed.fill(0);
    mail.write(seed);

    std::thread writer([&] {
        for (std::uint64_t x = 1; !stop.load(std::memory_order_relaxed); ++x) {
            Payload in;
            in.fill(x);
            mail.write(in);
        }
    });

    long reads = 0, torn = 0, missed = 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(400)) {
        Payload out;
        if (mail.read(out)) {
            ++reads;
            if (!out.uniform()) ++torn;
        } else {
            ++missed;
        }
    }
    stop.store(true, std::memory_order_relaxed);
    writer.join();

    std::printf("[mailbox] back-to-back writer: %ld reads, %ld torn, "
                "%ld gave way (%.1f%% of attempts succeeded)\n",
                reads, torn, missed,
                100.0 * static_cast<double>(reads) / std::max(reads + missed, 1L));
    EXPECT_EQ(torn, 0) << "a reader observed a mixture of two writes";
    // Deliberately no assertion on the success RATE here. A writer in a tight
    // loop leaves almost no stable window, so the reader gives way nearly every
    // attempt -- correct behaviour for this input, and nothing like production.
    // ReaderWinsAtProductionRates covers the rate that matters.
    EXPECT_GT(reads, 0) << "the reader never once completed a read";
}

/*! Two independent readers on one mailbox.
 *
 *  The observer thread reads the same joint and base mailboxes the control
 *  thread reads, rather than being handed duplicates, so this is a contract the
 *  node depends on rather than a theoretical property. A seqlock supports it --
 *  read() is const, holds no state between calls, and each reader's retry loop
 *  is independent -- but the header used to forbid it, so it is worth a test
 *  rather than a corrected comment alone.
 *
 *  What would break if readers did interfere: one reader's retry would consume
 *  or reset state the other needed, showing up as tearing or as one reader
 *  systematically starving. Both are checked.
 */
TEST(RtMailbox, TwoReadersBothSucceedAndNeitherTears)
{
    SeqlockMailbox<Payload> mail;
    std::atomic<bool> stop{false};
    Payload seed;
    seed.fill(0);
    mail.write(seed);

    // Paced like the real producer rather than a tight loop, so each reader
    // should land a fresh value nearly every attempt -- the question here is
    // whether having TWO of them changes that.
    std::thread writer([&] {
        for (std::uint64_t x = 1; !stop.load(std::memory_order_relaxed); ++x) {
            Payload in;
            in.fill(x);
            mail.write(in);
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    struct Result { long reads = 0, torn = 0, missed = 0; };
    Result r[2];

    auto reader = [&](Result& out) {
        const auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(400)) {
            Payload got;
            if (mail.read(got)) {
                ++out.reads;
                if (!got.uniform()) ++out.torn;
            } else {
                ++out.missed;
            }
        }
    };

    std::thread a([&] { reader(r[0]); });
    std::thread b([&] { reader(r[1]); });
    a.join();
    b.join();
    stop.store(true, std::memory_order_relaxed);
    writer.join();

    for (int i = 0; i < 2; ++i) {
        const double rate = 100.0 * static_cast<double>(r[i].reads) /
                            std::max(r[i].reads + r[i].missed, 1L);
        std::printf("[mailbox] reader %d: %ld reads, %ld torn, %ld gave way "
                    "(%.2f%% succeeded)\n",
                    i, r[i].reads, r[i].torn, r[i].missed, rate);
        EXPECT_EQ(r[i].torn, 0) << "reader " << i << " observed a torn value";
        EXPECT_GT(r[i].reads, 0) << "reader " << i << " never completed a read";
        // Neither reader may be starved by the other's presence. Against a
        // producer idle most of the period, a reader that interfered would show
        // up here as a collapsed success rate.
        EXPECT_GT(rate, 95.0) << "reader " << i << " was starved";
    }
}

/*! Progress at the rate the node actually runs at. The hand-off is only useful
 *  if the control loop gets a fresh measurement essentially every tick: the
 *  no-tearing guarantee would be worthless if the loop paid for it by reading
 *  stale state. Paced an order of magnitude faster than the real 100 Hz, so the
 *  margin is real rather than borrowed from a slow writer. */
TEST(RtMailbox, ReaderWinsAtProductionRates)
{
    SeqlockMailbox<Payload> mail;
    std::atomic<bool> stop{false};
    Payload seed;
    seed.fill(0);
    mail.write(seed);

    std::thread writer([&] {
        for (std::uint64_t x = 1; !stop.load(std::memory_order_relaxed); ++x) {
            Payload in;
            in.fill(x);
            mail.write(in);
            std::this_thread::sleep_for(std::chrono::microseconds(1000));   // 1 kHz
        }
    });

    long reads = 0, torn = 0, missed = 0;
    for (int i = 0; i < 2000; ++i) {                 // 2000 ticks at ~1 kHz
        Payload out;
        if (mail.read(out)) {
            ++reads;
            if (!out.uniform()) ++torn;
        } else {
            ++missed;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }
    stop.store(true, std::memory_order_relaxed);
    writer.join();

    const double rate = 100.0 * static_cast<double>(reads) / std::max(reads + missed, 1L);
    std::printf("[mailbox] 1 kHz writer: %ld reads, %ld torn, %ld gave way "
                "(%.2f%% succeeded)\n", reads, torn, missed, rate);
    EXPECT_EQ(torn, 0);
    EXPECT_GT(rate, 99.0) << "the control loop would be reading stale state";
}

/*! The consumer must not allocate: it runs on the control thread, which the
 *  allocation work exists to keep off the heap. */
TEST(RtMailbox, ReadingDoesNotAllocate)
{
    SeqlockMailbox<Payload> mail;
    Payload in, out;
    in.fill(3);
    mail.write(in);
    mail.read(out);              // warm anything lazy before counting

    long allocs = 0;
    {
        AllocScope scope;
        for (int i = 0; i < 100; ++i) {
            in.fill(static_cast<std::uint64_t>(i));
            mail.write(in);
            mail.read(out);
        }
        allocs = scope.n();
    }
    EXPECT_EQ(allocs, 0);
}

// --- The control loop's output queue -------------------------------------- //

TEST(RtRing, PreservesOrderAndReportsEmpty)
{
    SpscRing<int, 4> ring;
    int out = -1;
    EXPECT_FALSE(ring.pop(out));
    EXPECT_EQ(ring.size(), 0u);

    for (int i = 0; i < 3; ++i) EXPECT_TRUE(ring.push(i));
    EXPECT_EQ(ring.size(), 3u);
    for (int i = 0; i < 3; ++i) {
        ASSERT_TRUE(ring.pop(out));
        EXPECT_EQ(out, i) << "the queue reordered its values";
    }
    EXPECT_FALSE(ring.pop(out));
}

TEST(RtRing, ReportsTheDropWhenFullRatherThanBlocking)
{
    SpscRing<int, 4> ring;
    for (int i = 0; i < 4; ++i) EXPECT_TRUE(ring.push(i));
    // Full. The control loop must get an answer, not wait for room.
    EXPECT_FALSE(ring.push(99));
    EXPECT_EQ(ring.size(), 4u);
    // And the values already queued are intact -- a refused push must not
    // overwrite the oldest, or the consumer would see a gap it cannot detect.
    int out = -1;
    for (int i = 0; i < 4; ++i) {
        ASSERT_TRUE(ring.pop(out));
        EXPECT_EQ(out, i);
    }
}

TEST(RtRing, WrapsRepeatedlyWithoutLoss)
{
    // Many more pushes than the capacity, to catch an index that wraps wrongly.
    SpscRing<int, 4> ring;
    int out = -1;
    for (int i = 0; i < 1000; ++i) {
        ASSERT_TRUE(ring.push(i));
        ASSERT_TRUE(ring.pop(out));
        ASSERT_EQ(out, i);
    }
}

/*! The property the outbound path depends on: with a consumer that keeps up,
 *  every value the control loop produces is published exactly once, in order.
 *  A dropped torque leaves an actuator on a stale command. */
TEST(RtRing, LosesNothingAcrossThreadsWhenDrained)
{
    SpscRing<long, 16> ring;
    constexpr long kN = 200000;
    std::atomic<bool> producer_done{false};

    std::thread producer([&] {
        for (long i = 0; i < kN; ++i) {
            while (!ring.push(i)) std::this_thread::yield();   // consumer catches up
        }
        producer_done.store(true, std::memory_order_release);
    });

    long expected = 0, got = 0, misordered = 0;
    long v = 0;
    while (expected < kN) {
        if (ring.pop(v)) {
            if (v != expected) ++misordered;
            ++expected;
            ++got;
        } else if (producer_done.load(std::memory_order_acquire) &&
                   ring.size() == 0) {
            break;
        }
    }
    producer.join();

    std::printf("[ring] %ld of %ld values through a 16-slot queue, %ld misordered\n",
                got, kN, misordered);
    EXPECT_EQ(got, kN) << "values were lost in transit";
    EXPECT_EQ(misordered, 0);
}

TEST(RtRing, PushingDoesNotAllocate)
{
    SpscRing<long, 16> ring;
    long v = 0;
    ring.push(1);
    ring.pop(v);

    long allocs = 0;
    {
        AllocScope scope;
        for (long i = 0; i < 1000; ++i) {
            ring.push(i);
            ring.pop(v);
        }
        allocs = scope.n();
    }
    EXPECT_EQ(allocs, 0);
}

}  // namespace

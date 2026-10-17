/*! nav_node -- the base state estimator, on its own realtime thread.
 *
 *  Runs a navigator (navigator_factory.hpp; "imu_ekf" is the one this node
 *  exists for) and publishes its estimate as nav_msgs/Odometry on
 *  /chaser/nav/odom, covariance filled. Unlike gnc_node it commands nothing,
 *  so it can run alongside pat_arm_control, which reads the estimate in place
 *  of /chaser/odom when given odom_topic:=/chaser/nav/odom.
 *
 *  Threading mirrors pat_arm_control's arm_control_node:
 *
 *    /chaser/imu  --(executor callback)--> imu_ring_   --+
 *    /chaser/odom --(executor callback)--> pose_ring_  --+
 *                                                        v
 *              nav thread (SCHED_FIFO rt_priority_nav, pinned to rt_cpus_nav)
 *              wakes at nav_hz, drains both rings in timestamp order, runs
 *              the filter, writes est_mail_ and stats_mail_
 *                                                        |
 *              executor timers: publish /chaser/nav/odom at publish_hz,
 *                               report once a second
 *
 *  Callbacks only copy into a ring -- no filter work, no lock -- and a full
 *  ring is counted, never waited on. The nav thread never calls into ROS:
 *  publishing allocates inside DDS, so the executor publishes from the
 *  mailbox. Nothing on the nav thread allocates once start() has run.
 */
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include "pat_rt/rt_mailbox.h"
#include "pat_rt/rt_thread.h"

#include "navigator_factory.hpp"

namespace {

//! One IMU sample, as handed from the callback to the nav thread.
struct ImuSample {
    double t = 0.0;
    double ax = 0.0, ay = 0.0, gz = 0.0;
};

//! One odometry message: pose and twist, stamp.
struct PoseSample {
    double t = 0.0;
    double x = 0.0, y = 0.0, yaw = 0.0;
    double vx = 0.0, vy = 0.0, w = 0.0;
};

//! The estimate, handed from the nav thread to the publisher.
struct NavEstimate {
    double t = 0.0;
    bool ready = false;
    double x[6] = {};       //!< [px, py, th, vx, vy, omega]
    double cov[36] = {};    //!< row-major 6x6, same order
};

//! What the report prints, written by the nav thread once per wake.
struct NavStats {
    std::uint64_t wakes = 0;
    std::uint64_t overruns = 0;         //!< wakes whose work outlasted the period
    double work_us_max = 0.0;           //!< since the last report
    double work_us_sum = 0.0;
    std::uint64_t work_n = 0;
    std::uint64_t imu_in = 0, pose_in = 0;
    bool have_filter = false;
    pat_gnc::ImuEkf::Diagnostics filter{};
};

// Ring capacities. The IMU ring holds what arrives between two wakes with a
// wide margin (a 500 Hz IMU against a 200 Hz thread is 3 per wake); a full
// ring is a dropped sample, counted. Powers of two, as SpscRing requires.
constexpr std::size_t kImuRing = 256;
constexpr std::size_t kPoseRing = 64;

}  // namespace

class NavNode : public rclcpp::Node {
public:
    NavNode() : Node("pat_nav") {
        nav_hz_     = declare_parameter<double>("nav_hz", 200.0);
        pub_hz_     = declare_parameter<double>("publish_hz", 100.0);
        imu_hz_     = declare_parameter<double>("imu_hz", 100.0);
        rt_prio_    = declare_parameter<int>("rt_priority_nav", 0);
        rt_cpus_    = declare_parameter<std::vector<long>>("rt_cpus_nav", std::vector<long>{});
        lock_mem_   = declare_parameter<bool>("lock_memory", false);
        out_topic_  = declare_parameter<std::string>("output_topic", "/chaser/nav/odom");
        if (!(nav_hz_ > 0.0) || !(pub_hz_ > 0.0) || !(imu_hz_ > 0.0)) {
            throw std::runtime_error("nav_node: nav_hz, publish_hz and imu_hz must be positive");
        }

        // The navigator is built here and from then on touched ONLY by the nav
        // thread. imu_hz is the plain "ekf" navigator's fixed predict rate,
        // since it predicts once per IMU sample; the others read the stamps.
        navigator_ = pat_gnc::makeNavigator(*this, imu_hz_, nav_type_);
        imu_nav_ = dynamic_cast<pat_gnc::ImuEkfNavigator*>(navigator_.get());
        if (imu_nav_) checkRealtimeBudget();

        imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
            "/chaser/imu", rclcpp::SensorDataQoS().keep_last(50),
            [this](sensor_msgs::msg::Imu::SharedPtr m) {
                ImuSample s;
                s.t = rclcpp::Time(m->header.stamp).seconds();
                s.ax = m->linear_acceleration.x;
                s.ay = m->linear_acceleration.y;
                s.gz = m->angular_velocity.z;
                if (!imu_ring_.push(s)) imu_drops_.fetch_add(1, std::memory_order_relaxed);
            });
        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/chaser/odom", 10, [this](nav_msgs::msg::Odometry::SharedPtr m) {
                if (frame_id_.empty()) frame_id_ = m->header.frame_id;
                const auto& q = m->pose.pose.orientation;
                PoseSample s;
                s.t = rclcpp::Time(m->header.stamp).seconds();
                s.x = m->pose.pose.position.x;
                s.y = m->pose.pose.position.y;
                s.yaw = std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                                   1.0 - 2.0 * (q.y * q.y + q.z * q.z));
                s.vx = m->twist.twist.linear.x;
                s.vy = m->twist.twist.linear.y;
                s.w = m->twist.twist.angular.z;
                if (!pose_ring_.push(s)) pose_drops_.fetch_add(1, std::memory_order_relaxed);
            });

        pub_ = create_publisher<nav_msgs::msg::Odometry>(out_topic_, 10);
        pub_tmr_ = create_wall_timer(std::chrono::duration<double>(1.0 / pub_hz_),
                                     [this] { publishEstimate(); });
        report_tmr_ = create_wall_timer(std::chrono::seconds(1), [this] { report(); });

        RCLCPP_INFO(get_logger(), "nav_node: navigator %s, thread %.0f Hz, publishing %s "
                    "at %.0f Hz", nav_type_.c_str(), nav_hz_, out_topic_.c_str(), pub_hz_);
    }

    ~NavNode() override {
        running_.store(false, std::memory_order_release);
        if (thread_.joinable()) thread_.join();
    }

    /*! Start the nav thread. Called after construction, not from it: a thread
     *  inherits the scheduling policy of its creator, so it must start after
     *  DDS has made its own threads, which then stay SCHED_OTHER. */
    void start() {
        if (lock_mem_) mlock_rc_.store(pat_rt::lockMemory(), std::memory_order_relaxed);
        running_.store(true, std::memory_order_release);
        thread_ = std::thread(&NavNode::navThread, this);
    }

private:
    /*! Refuse a configuration the thread cannot meet, rather than discover it
     *  as overruns.
     *
     *  Two checks. The history has to reach back further than the oldest
     *  latency, or every late measurement is dropped as too old. And the
     *  worst wake -- one odometry message whose every part is late enough to
     *  replay the whole history -- has to fit in half the wake period. That
     *  cost is MEASURED here, on this machine, by running exactly that replay
     *  on a scratch filter, so the check means the same thing on the dev box
     *  and on the Jetson. */
    void checkRealtimeBudget() {
        const auto& cfg = imu_nav_->config();
        const auto& p = cfg.ekf;
        const double max_lat = std::max({cfg.latency_yaw, cfg.latency_pos,
                                         cfg.use_twist ? cfg.latency_vel : 0.0});
        // +2: the measurement lands one step before its stamp at worst, and the
        // newest step is not history.
        const double need = max_lat * imu_hz_ + 2.0;
        if (static_cast<double>(p.history_len) < need) {
            throw std::runtime_error(
                "nav_node: imu_ekf.history_len " + std::to_string(p.history_len) +
                " cannot hold the longest latency (" + std::to_string(max_lat) +
                " s at imu_hz " + std::to_string(imu_hz_) + " needs " +
                std::to_string(static_cast<int>(std::ceil(need))) + ")");
        }

        pat_gnc::ImuEkf scratch(p);
        scratch.initialise(0.0, 0.0, 0.0, 0.0);
        const double dt = 1.0 / imu_hz_;
        for (std::size_t k = 1; k <= p.history_len + 1; ++k) {
            scratch.propagate(dt * static_cast<double>(k), {0.01, 0.0}, 0.001);
        }
        const double oldest = dt * 2.0;
        double worst_us = 0.0;
        for (int rep = 0; rep < 5; ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            scratch.updateYaw(oldest, 0.0);
            const double us = std::chrono::duration<double, std::micro>(
                                  std::chrono::steady_clock::now() - t0).count();
            worst_us = std::max(worst_us, us);
        }
        const int replays_per_msg = 2 + (cfg.use_twist ? 1 : 0);
        const double wake_us = worst_us * replays_per_msg;
        const double budget_us = 0.5e6 / nav_hz_;
        RCLCPP_INFO(get_logger(),
                    "nav_node: worst replay of %zu steps measured at %.0f us; worst wake "
                    "(%d replays) %.0f us against a budget of %.0f us (half of 1/nav_hz)",
                    p.history_len, worst_us, replays_per_msg, wake_us, budget_us);
        if (wake_us > budget_us) {
            throw std::runtime_error(
                "nav_node: the worst-case wake does not fit -- lower imu_ekf.history_len, "
                "lower nav_hz, or move the thread to a faster core");
        }
    }

    void navThread() {
        affinity_rc_.store(pat_rt::applyAffinity(rt_cpus_), std::memory_order_relaxed);
        rt_rc_.store(pat_rt::applyRealtimePolicy(rt_prio_), std::memory_order_relaxed);

        const long period_ns = static_cast<long>(1e9 / nav_hz_);
        pat_rt::PeriodicWaker waker(period_ns);
        waker.start();
        while (running_.load(std::memory_order_acquire)) {
            waker.waitNext();
            if (!running_.load(std::memory_order_acquire)) break;
            const auto t0 = std::chrono::steady_clock::now();
            step();
            const double us = std::chrono::duration<double, std::micro>(
                                  std::chrono::steady_clock::now() - t0).count();
            ++stats_.wakes;
            if (us * 1e3 > static_cast<double>(period_ns)) ++stats_.overruns;
            stats_.work_us_max = std::max(stats_.work_us_max, us);
            stats_.work_us_sum += us;
            ++stats_.work_n;
            if (imu_nav_) {
                stats_.have_filter = true;
                stats_.filter = imu_nav_->filter().diagnostics();
            }
            stats_mail_.write(stats_);
        }
    }

    //! One wake: drain both rings and feed the navigator in timestamp order.
    void step() {
        std::size_t n_imu = 0, n_pose = 0;
        while (n_imu < kImuRing && imu_ring_.pop(imu_buf_[n_imu])) ++n_imu;
        while (n_pose < kPoseRing && pose_ring_.pop(pose_buf_[n_pose])) ++n_pose;
        stats_.imu_in += n_imu;
        stats_.pose_in += n_pose;
        if (n_imu == 0 && n_pose == 0) return;

        // Merge on stamp. On a tie the IMU goes first, so the filter has been
        // propagated to a measurement's time before the measurement is applied
        // -- the simulator stamps the IMU and odometry of one tick identically.
        std::size_t i = 0, j = 0;
        while (i < n_imu || j < n_pose) {
            const bool take_imu =
                j >= n_pose || (i < n_imu && imu_buf_[i].t <= pose_buf_[j].t);
            if (take_imu) {
                const ImuSample& s = imu_buf_[i++];
                navigator_->onImu(s.t, {s.ax, s.ay}, s.gz);
            } else {
                const PoseSample& s = pose_buf_[j++];
                navigator_->onOdometry(s.t, {s.x, s.y, s.yaw}, {s.vx, s.vy, s.w});
            }
        }

        NavEstimate e;
        e.ready = navigator_->ready();
        if (e.ready) {
            e.t = imu_nav_ ? imu_nav_->filter().time()
                           : (n_pose ? pose_buf_[n_pose - 1].t : imu_buf_[n_imu - 1].t);
            const auto x = navigator_->state();
            const auto C = navigator_->covariance();
            for (int r = 0; r < 6; ++r) {
                e.x[r] = x(r);
                for (int c = 0; c < 6; ++c) e.cov[6 * r + c] = C(r, c);
            }
        }
        est_mail_.write(e);
    }

    void publishEstimate() {
        NavEstimate e;
        if (!est_mail_.read(e) || !e.ready) return;
        if (e.t == last_pub_t_) return;          // nothing new since last publish
        last_pub_t_ = e.t;

        nav_msgs::msg::Odometry m;
        m.header.stamp = rclcpp::Time(static_cast<std::int64_t>(std::llround(e.t * 1e9)),
                                      RCL_ROS_TIME);
        m.header.frame_id = frame_id_.empty() ? "map" : frame_id_;
        m.child_frame_id = "chaser";
        m.pose.pose.position.x = e.x[0];
        m.pose.pose.position.y = e.x[1];
        m.pose.pose.orientation.w = std::cos(0.5 * e.x[2]);
        m.pose.pose.orientation.z = std::sin(0.5 * e.x[2]);
        m.twist.twist.linear.x = e.x[3];
        m.twist.twist.linear.y = e.x[4];
        m.twist.twist.angular.z = e.x[5];

        // Odometry carries pose and twist covariances as separate 6x6 blocks
        // over [x, y, z, roll, pitch, yaw], so the planar states land on
        // indices 0, 1 and 5 of each. The pose-twist cross-covariance has no
        // slot in the message and is not published.
        const int pose_idx[3] = {0, 1, 5};   // px, py, th
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                m.pose.covariance[static_cast<std::size_t>(6 * pose_idx[r] + pose_idx[c])] =
                    e.cov[6 * r + c];
                m.twist.covariance[static_cast<std::size_t>(6 * pose_idx[r] + pose_idx[c])] =
                    e.cov[6 * (r + 3) + (c + 3)];
            }
        }
        pub_->publish(m);
    }

    void report() {
        if (!rt_reported_) {
            rt_reported_ = true;
            const int rt = rt_rc_.load(), aff = affinity_rc_.load(), ml = mlock_rc_.load();
            if (rt == 0) {
                RCLCPP_INFO(get_logger(), "nav_node: nav thread at SCHED_FIFO %d", rt_prio_);
            } else if (rt > 0) {
                RCLCPP_WARN(get_logger(), "nav_node: SCHED_FIFO %d refused (%s), running at "
                            "normal priority", rt_prio_, std::strerror(rt));
            }
            if (aff > 0) {
                RCLCPP_WARN(get_logger(), "nav_node: rt_cpus_nav refused (%s)", std::strerror(aff));
            }
            if (ml > 0) {
                RCLCPP_WARN(get_logger(), "nav_node: mlockall failed (%s)", std::strerror(ml));
            }
        }

        NavStats s;
        if (!stats_mail_.read(s)) return;
        // Per-window figures come from differencing the running totals, since
        // the nav thread never resets anything the report reads.
        const std::uint64_t n = s.work_n - prev_.work_n;
        const double mean = n ? (s.work_us_sum - prev_.work_us_sum) / static_cast<double>(n) : 0.0;
        const double imu_drops = static_cast<double>(imu_drops_.load());
        const double pose_drops = static_cast<double>(pose_drops_.load());
        if (s.have_filter) {
            const auto& f = s.filter;
            RCLCPP_INFO(get_logger(),
                        "nav[%s] %lu wakes, work mean %.0f max %.0f us, overruns %lu | in imu %lu "
                        "pose %lu, ring drops %.0f/%.0f | yaw %lu/%lu rej, pos %lu/%lu rej, too_old "
                        "%lu, gaps %lu, replays %lu (max %zu steps) | NIS yaw %.2f pos %.2f",
                        nav_type_.c_str(), static_cast<unsigned long>(s.wakes - prev_.wakes), mean,
                        s.work_us_max, static_cast<unsigned long>(s.overruns),
                        static_cast<unsigned long>(s.imu_in - prev_.imu_in),
                        static_cast<unsigned long>(s.pose_in - prev_.pose_in), imu_drops, pose_drops,
                        static_cast<unsigned long>(f.yaw_updates),
                        static_cast<unsigned long>(f.yaw_rejected),
                        static_cast<unsigned long>(f.pos_updates),
                        static_cast<unsigned long>(f.pos_rejected),
                        static_cast<unsigned long>(f.too_old), static_cast<unsigned long>(f.imu_gaps),
                        static_cast<unsigned long>(f.replays), f.max_replay_steps,
                        f.last_nis_yaw, f.last_nis_pos);
        } else {
            RCLCPP_INFO(get_logger(), "nav[%s] %lu wakes, work mean %.0f max %.0f us, overruns %lu",
                        nav_type_.c_str(), static_cast<unsigned long>(s.wakes - prev_.wakes), mean,
                        s.work_us_max, static_cast<unsigned long>(s.overruns));
        }
        prev_ = s;
    }

    // --- configuration -----------------------------------------------------
    double nav_hz_{200.0}, pub_hz_{100.0}, imu_hz_{100.0};
    int rt_prio_{0};
    std::vector<long> rt_cpus_;
    bool lock_mem_{false};
    std::string out_topic_;
    std::string nav_type_;

    // --- nav thread only ---------------------------------------------------
    std::unique_ptr<pat_gnc::INavigator> navigator_;
    pat_gnc::ImuEkfNavigator* imu_nav_{nullptr};   //!< navigator_, if it is one
    std::array<ImuSample, kImuRing> imu_buf_{};
    std::array<PoseSample, kPoseRing> pose_buf_{};
    NavStats stats_;

    // --- hand-off ----------------------------------------------------------
    pat_rt::SpscRing<ImuSample, kImuRing> imu_ring_;
    pat_rt::SpscRing<PoseSample, kPoseRing> pose_ring_;
    pat_rt::SeqlockMailbox<NavEstimate> est_mail_;
    pat_rt::SeqlockMailbox<NavStats> stats_mail_;
    std::atomic<std::uint64_t> imu_drops_{0}, pose_drops_{0};

    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<int> affinity_rc_{-1}, rt_rc_{-1}, mlock_rc_{-1};

    // --- executor only -----------------------------------------------------
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr pub_tmr_, report_tmr_;
    std::string frame_id_;
    double last_pub_t_{-1.0};
    bool rt_reported_{false};
    NavStats prev_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<NavNode>();
    node->start();
    rclcpp::spin(node);
    rclcpp::shutdown();
}

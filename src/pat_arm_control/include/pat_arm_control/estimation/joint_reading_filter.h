#pragma once
/*! JointKalmanFilter as a node runs it: on timestamped encoder readings that
 *  arrive at their own rate, not one per control tick.
 *
 *  The control thread and the joint_states topic run at the same nominal rate
 *  but are not synchronised, so a tick can read the same message twice, or a
 *  tick can miss one. Fusing a repeat as a new reading would count the same
 *  noise twice and make the filter overconfident; stepping a fixed period when
 *  two periods have passed would mis-predict the velocity. So a reading is
 *  fused only when its stamp is newer than the last one fused, and the filter
 *  steps by the time between the two stamps.
 *
 *  Fed one reading per period, it computes what contact_bench's
 *  `--joint-filter kalman` computes (JointKalmanFilter::update with the control
 *  period): the live and offline `ekf-jkf` sources are the same filter.
 *
 *  A stamp that goes BACKWARDS (a simulator restarted under a running node, a
 *  clock reset) restarts the filter at that reading rather than freezing it.
 *
 *  update() allocates nothing and does not throw, so it can run on the
 *  realtime control thread. Zero ROS. */
#include <cstdint>

#include <Eigen/Dense>

#include "pat_arm_control/estimation/joint_kalman_filter.h"

namespace pat_arm_control {

class JointReadingFilter {
public:
    /*! Throws std::invalid_argument as JointKalmanFilter does. */
    JointReadingFilter(const JointKalmanParams& p, int n_joints);

    /*! One reading of every joint, `q` and `qd` each n_joints long, taken at
     *  `stamp` [s]. Returns true when it was fused, false for a repeat. */
    bool update(double stamp, const double* q, const double* qd) noexcept;

    bool ready() const noexcept { return kf_.ready(); }
    int size() const noexcept { return kf_.size(); }
    double position(int i) const { return kf_.position(i); }
    double velocity(int i) const { return kf_.velocity(i); }
    const JointKalmanParams& params() const noexcept { return kf_.params(); }

    std::uint64_t fused() const noexcept { return n_fused_; }
    std::uint64_t repeats() const noexcept { return n_repeats_; }
    std::uint64_t restarts() const noexcept { return n_restarts_; }

private:
    JointKalmanFilter kf_;
    Eigen::VectorXd zq_, zv_;      //!< the reading, sized once
    double last_stamp_ = 0.0;
    std::uint64_t n_fused_ = 0, n_repeats_ = 0, n_restarts_ = 0;
};

}  // namespace pat_arm_control

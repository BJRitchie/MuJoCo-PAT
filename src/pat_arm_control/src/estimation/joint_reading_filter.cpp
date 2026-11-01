#include "pat_arm_control/estimation/joint_reading_filter.h"

#include <cmath>
#include <exception>

namespace pat_arm_control {

JointReadingFilter::JointReadingFilter(const JointKalmanParams& p, int n_joints)
    : kf_(p, n_joints), zq_(Eigen::VectorXd::Zero(n_joints)),
      zv_(Eigen::VectorXd::Zero(n_joints)) {}

bool JointReadingFilter::update(double stamp, const double* q, const double* qd) noexcept {
    if (!std::isfinite(stamp)) return false;
    if (kf_.ready()) {
        if (stamp == last_stamp_) {
            ++n_repeats_;
            return false;
        }
        if (stamp < last_stamp_) {
            kf_.reset();               // the next update initialises at this reading
            ++n_restarts_;
        }
    }
    const int n = kf_.size();
    for (int i = 0; i < n; ++i) {
        zq_(i) = q[i];
        zv_(i) = qd[i];
    }
    try {
        // h is ignored on the first update (or after a restart).
        kf_.update(kf_.ready() ? stamp - last_stamp_ : 0.0, zq_, zv_);
    } catch (const std::exception&) {
        // Unreachable: sizes are fixed and h > 0 by the checks above. Restart
        // rather than let an exception reach the control thread.
        kf_.reset();
        ++n_restarts_;
        return false;
    }
    last_stamp_ = stamp;
    ++n_fused_;
    return true;
}

}  // namespace pat_arm_control

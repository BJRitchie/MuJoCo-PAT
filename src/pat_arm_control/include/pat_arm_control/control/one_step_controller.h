#pragma once
/*! A task-space law that is a closed-form function of the current state: no
 *  horizon, no solver, nothing carried from one tick to the next. The sliding
 *  mode and the one-step MPC are both this; the NMPC is not.
 *
 *  A tick is: reduce the model at the live state, form the planar task-space
 *  error, ask the law for a task-space force, map it to joint torques. This
 *  class does everything except the law:
 *
 *    - the reduced model (generalised Jacobian rows [x, y, thetaz], H_g, Cv,
 *      Jdot*qdot) and the task-space error against the reference, with
 *      e = p_d - p throughout;
 *    - L^-1 = J H_g^-1 J^T, from one Cholesky of H_g (the damped
 *      pseudo-inverse if H_g is not positive definite);
 *    - safe mode, the torque mapping and clamp, the profile and the trace.
 *
 *  A law implements taskForce(): given the state, the task-space force
 *  tau_task such that the joints are commanded tau = J^T tau_task + Cv. With
 *  the reduced dynamics H_g qdd + Cv = tau this gives the task acceleration
 *
 *      xdd = L^-1 tau_task + Jdot*qdot .
 *
 *  Allocation-free apart from the returned vector, provided the law is.
 */
#include <chrono>
#include <vector>

#include <Eigen/Dense>

#include "pat_arm_control/common/profiling.h"
#include "pat_arm_control/control/controller_params.h"
#include "pat_arm_control/control/task_space_controller.h"

namespace pat_arm_control {

class OneStepController : public TaskSpaceController {
public:
    void setTickBudgetUs(double us) override { tick.budget_us = us; }
    void snapshotProfile(ProfileSnapshot& out) const override;
    void resetProfile() override;

    //! Record every tick's duration and mode into a buffer allocated here, up
    //! to `capacity` rows (then stop, never grow). The phase columns stay 0 and
    //! the QP columns report "no solver", so the node's trace schema is shared.
    void enableTrace(std::size_t capacity) override;
    void drainTrace(std::vector<TickTrace>& sink) override;

protected:
    OneStepController(const std::string& mjcf_path, const ControllerParams& params);

    //! The planar task-space state a law works from, [x, y, thetaz] per axis.
    struct TaskState {
        Eigen::Vector3d e;        //!< p_d - p
        Eigen::Vector3d edot;     //!< v_d - J qdot
        Eigen::Vector3d a_d;      //!< reference acceleration
        Eigen::Vector3d v_ee;     //!< J qdot, the measured task velocity
        Eigen::Vector3d hp;       //!< Jdot*qdot, the bias acceleration
        Eigen::Matrix3d Lam_inv;  //!< J H_g^-1 J^T, undamped
    };

    /*! The law: the task-space force for this state. May throw (checkNaN): the
     *  guard turns that into safe mode. */
    virtual Eigen::Vector3d taskForce(const TaskState& s) = 0;

    // What a law can read while it runs, valid inside taskForce().
    const Eigen::MatrixXd& hgMatrix() const noexcept { return H_g_; }
    const Eigen::VectorXd& coriolis() const noexcept { return Cv_; }
    /*! H_g^-1 into `out` (n_joints_ x n_joints_, sized by the caller): from
     *  the Cholesky already computed, or the damped pseudo-inverse if H_g was
     *  not positive definite. */
    void hgInverse(Eigen::MatrixXd& out) const;

    Eigen::VectorXd controlLawImpl(const Eigen::VectorXd& q,
                                   const Eigen::VectorXd& v) final;

    PhaseStat tick;       //!< whole controlLawImpl
    PhaseStat dynamics;   //!< mass-matrix reduction, Jacobian, H_g and Cv
    PhaseStat jdotqdot;   //!< jdotQdot6
    PhaseStat pinv;       //!< the Cholesky of H_g and the solve for H_g^-1 J^T
    PhaseStat lambda;     //!< L^-1
    PhaseStat law;        //!< taskForce()
    PhaseStat finalize;   //!< torque mapping and clamp

private:
    // Per-tick workspace, sized at construction so a tick never allocates.
    BaseElim be_;
    Eigen::MatrixXd J6_, J3_, Jt_, HinvJt_, H_g_, I_;
    Eigen::VectorXd Cv_, tau_task_, out_;
    Eigen::LLT<Eigen::MatrixXd> llt_;
    bool hg_pd_ = true;

    std::vector<TickTrace> trace_;
    std::size_t trace_cap_ = 0;
    bool trace_t0_set_ = false;
    std::chrono::steady_clock::time_point trace_t0_;
    void recordTick(std::chrono::steady_clock::time_point start, double tick_us);
};

}  // namespace pat_arm_control

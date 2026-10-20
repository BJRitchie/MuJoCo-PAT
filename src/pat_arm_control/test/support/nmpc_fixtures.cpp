#include "nmpc_fixtures.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace pat_arm_control {
namespace test_support {

NMPCParams leftArmParams(double Ts)
{
    NMPCParams p;
    p.Ts             = Ts;
    p.N              = 5;
    p.ee_site_name   = "ee_site_L";
    p.fullNonlinear  = true;
    // Hg_damping deliberately left at the NMPCParams default so the
    // over-damping test tracks that default rather than a local choice.
    p.owned_joints   = {"joint2_L", "joint3_L", "joint5_L"};
    for (const auto& name : p.owned_joints) {
        p.joint_lims.push_back({name, -M_PI, M_PI, /*qd_max=*/3.0});
    }
    // Every model joint, not just the owned ones — the controller rejects an
    // incomplete list, since the torque constraint covers all of them.
    for (const auto& name : {"joint2_L", "joint3_L", "joint5_L",
                              "joint2_R", "joint3_R", "joint5_R"}) {
        p.model_torque_lims.emplace_back(name, 0.25);
    }
    return p;
}

NMPCParams liveParams()
{
    // The control period matters beyond Tier 1: the warm-start shift advances
    // by it, so leaving it unset makes the harness shift a whole stage per tick
    // where the real one shifts 0.3 of one.
    NMPCParams p = leftArmParams(0.033);
    p.control_period = 0.010;
    p.N = 30;
    p.sqp_iters = 1;
    p.fullNonlinear = true;
    p.terminal_velocity_constraint = true;
    p.terminal_cost_multiplier = 100.0;
    p.Q_pos = {10.0, 10.0, 1.0};
    p.Q_vel = {1.0, 1.0, 3.0};
    p.R     = {1e-1, 1e-1, 1e-1};
    p.qp_max_iter = 60;
    p.qp_tol = 1e-5;
    // Soft box, HARD torque, stated here rather than left to the struct defaults
    // because the pair is the measured decision and a harness that silently
    // drifted off it would be measuring a configuration nobody deploys.
    p.soft_joint_limits  = true;
    p.soft_torque_limits = false;
    for (auto& jl : p.joint_lims) { jl.qd_max = 2.0; }
    p.joint_lims[0].q_min =  0.0;   p.joint_lims[0].q_max = 3.14;
    p.joint_lims[1].q_min = -2.967; p.joint_lims[1].q_max = 0.0;
    p.joint_lims[2].q_min = -1.22;  p.joint_lims[2].q_max = 1.22;
    return p;
}

State referenceState(int n_joints)
{
    const double ja[6] = { 0.90, -1.20,  0.35,  1.10, -0.90, -0.40};
    const double jv[6] = { 0.12, -0.08,  0.05, -0.10,  0.06, -0.04};

    State s;
    s.q = Eigen::VectorXd::Zero(7 + n_joints);
    s.q(3) = 1.0;                     // base quaternion w
    s.v = Eigen::VectorXd::Zero(6 + n_joints);
    for (int i = 0; i < n_joints; ++i) {
        s.q(7 + i) = ja[i % 6];
        s.v(6 + i) = jv[i % 6];
    }
    return s;
}

RunMetrics runClosedLoop(NMPCParams params, const Scenario& sc, int ticks)
{
    const double dt = 1.0 / 200.0;      // control_hz
    if (sc.q_min_override > 0.0) params.joint_lims[0].q_min = sc.q_min_override;

    ModelProbe c(kMjcf, params);
    State s = referenceState(c.n_joints_);

    const ArmNMPC::EePose start = c.currentEePose(s.q, s.v);
    double tx = start.pos.x() + sc.setpoint_offset;
    double ty = start.pos.y() + sc.setpoint_offset;
    c.setDesiredPos(tx, ty, start.pos.z());
    c.setDesiredOrient(start.quat(0), start.quat(1), start.quat(2), start.quat(3));

    RunMetrics m;
    long damping_ticks = 0, relax_ticks = 0;
    int  run_damping = 0;
    double err_sum = 0.0, t_sum = 0.0;

    for (int t = 0; t < ticks; ++t) {
        if (sc.step_at_tick > 0 && t == sc.step_at_tick) {
            tx += sc.step_offset;
            ty += sc.step_offset;
            c.setDesiredPos(tx, ty, start.pos.z());
        }
        const auto t0 = std::chrono::high_resolution_clock::now();
        const Eigen::VectorXd tau_all = c.computeControl(s.q, s.v);
        const double tick_us =
            std::chrono::duration<double, std::micro>(
                std::chrono::high_resolution_clock::now() - t0).count();

        t_sum += tick_us;
        m.max_tick_us = std::max(m.max_tick_us, tick_us);
        if (tick_us > dt * 1e6) m.frac_over_budget += 1.0;
        ++m.status_hist[c.lastQpStatus()];

        if (c.mode() == ArmNMPC::ControlMode::Damping) {
            ++damping_ticks;
            m.max_consecutive_damping = std::max(m.max_consecutive_damping, ++run_damping);
        } else {
            run_damping = 0;
        }

        if (c.qp_->lastRelaxCount() > 0) {
            ++relax_ticks;
            m.guard_relax_max = std::max(m.guard_relax_max, c.qp_->lastRelaxMax());
        }

        const int n_box = c.qp_->nsbxAt(1);
        if (n_box > 0) {
            Eigen::VectorXd sl(c.qp_->nsAt(1)), su(c.qp_->nsAt(1));
            c.qp_->getSlack(1, sl.data(), su.data());
            m.max_box_slack = std::max({m.max_box_slack,
                                         sl.head(n_box).maxCoeff(),
                                         su.head(n_box).maxCoeff()});
        }

        Eigen::VectorXd tau = Eigen::VectorXd::Zero(c.n_joints_);
        for (int idx : c.ownedJointInds) tau[idx] = tau_all[idx];
        Eigen::VectorXd q1, v1;
        c.integrateStep(s.q, s.v, tau, dt, q1, v1);
        s.q = q1;
        s.v = v1;

        // Realised violation of the trajectory actually flown, against the
        // configured limits -- the safety payoff a hard constraint is meant to buy.
        double worst_pos = 0.0;
        for (int idx : c.ownedJointInds) {
            const double q = s.q[7 + idx];
            worst_pos = std::max({worst_pos,
                                   c.qlimsByIndex[idx].first - q,
                                   q - c.qlimsByIndex[idx].second});
            m.max_vel_violation = std::max(
                m.max_vel_violation, std::abs(s.v[6 + idx]) - c.vlimsByIndex[idx]);
        }
        worst_pos = std::max(0.0, worst_pos);
        m.max_pos_violation = std::max(m.max_pos_violation, worst_pos);
        m.pos_violation_integral += worst_pos * dt;
        if (t == 0) m.initial_pos_violation = worst_pos;
        m.final_pos_violation = worst_pos;

        const ArmNMPC::EePose now = c.currentEePose(s.q, s.v);
        err_sum += std::hypot(now.pos.x() - tx, now.pos.y() - ty);
    }

    m.qp_iter_mean      = c.qp_iters.mean();
    m.qp_iter_max       = c.qp_iters.max;
    m.qp_retries        = c.qp_retries;
    m.recovered         = m.initial_pos_violation - m.final_pos_violation;
    m.availability      = 1.0 - static_cast<double>(damping_ticks) / ticks;
    m.guard_relax_frac  = static_cast<double>(relax_ticks) / ticks;
    m.mean_ee_err       = err_sum / ticks;
    m.mean_tick_us      = t_sum / ticks;
    m.frac_over_budget /= ticks;
    m.max_vel_violation = std::max(0.0, m.max_vel_violation);
    return m;
}

Scenario warmStartScenario()
{
    Scenario sc;
    sc.setpoint_offset = 0.06;
    sc.step_at_tick    = 150;
    sc.step_offset     = 0.12;
    return sc;
}

}  // namespace test_support
}  // namespace pat_arm_control

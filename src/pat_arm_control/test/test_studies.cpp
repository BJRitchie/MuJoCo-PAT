/*! Measurement studies, not tests: each prints a table and asserts nothing.
 *  Disabled so they never run in CI; run one with
 *
 *    ./build/pat_arm_control/test_pat_arm_control \
 *        --gtest_also_run_disabled_tests --gtest_filter='DISABLED_ConstraintStudy.*'
 *
 *  Several of the README's settled decisions rest on these (the constraint
 *  configuration, the warm-start mode, the allocation-free control path), so
 *  re-run the relevant one before changing what it measured.
 */
#include <gtest/gtest.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "support/alloc_trace.h"
#include "support/nmpc_fixtures.h"

namespace pat_arm_control {
namespace {
using namespace test_support;

/*! Where the tick actually goes, as the per-phase breakdown. The instrument for
 *  judging a cost change: run it before and after, one change at a time. Harness
 *  ticks are faster than live ones (one arm, no ROS), so compare only against
 *  another run of this same test. */
TEST(DISABLED_PhaseStudy, Breakdown)
{
    ModelProbe c(kMjcf, liveParams());
    c.setTickBudgetUs(10000.0);          // the live 100 Hz period
    const State s = referenceState(c.n_joints_);
    const ArmNMPC::EePose start = c.currentEePose(s.q, s.v);
    c.setDesiredPos(start.pos.x() + 0.06, start.pos.y() + 0.06, start.pos.z());
    c.setDesiredOrient(start.quat(0), start.quat(1), start.quat(2), start.quat(3));

    for (int i = 0; i < 5; ++i) c.computeControl(s.q, s.v);   // warm up
    c.resetProfile();

    State st = s;
    for (int t = 0; t < 400; ++t) {
        const Eigen::VectorXd tau_all = c.computeControl(st.q, st.v);
        Eigen::VectorXd tau = Eigen::VectorXd::Zero(c.n_joints_);
        for (int idx : c.ownedJointInds) tau[idx] = tau_all[idx];
        Eigen::VectorXd q1, v1;
        c.integrateStep(st.q, st.v, tau, 1.0 / 100.0, q1, v1);
        st.q = q1;
        st.v = v1;
    }
    std::printf("%s\n", c.profileReport().c_str());
}

/*! How much of each arm's plan depends on joints it does not command.
 *
 *  Each node runs its own ArmNMPC over the WHOLE model, so J_task is 3 x 6: its
 *  columns for the other arm's joints are non-zero, because moving that arm pushes
 *  the shared free-floating base and so moves this arm's end effector. The QP is
 *  free to exploit those columns, but the node publishes effort only for its own
 *  joints, and the other controller drives the rest to its own objective. The
 *  column norms say whether that is a rounding error or a real model mismatch. */
TEST(DISABLED_ArchitectureStudy, CrossArmCoupling)
{
    ModelProbe c(kMjcf, liveParams());        // left arm owns joints 0,1,2
    const State s = referenceState(c.n_joints_);
    const NodeDynamics nd = c.dynamicsAt(s.q, s.v, c.ee_site_id_);
    const ModelProbe::NodeTaskSpace ts =
        c.evalTaskSpace(s.q, s.v, nd.ee_pos, nd.ee_quat, nd.J_g6, nd.jdot_qdot6);

    std::vector<bool> owned(c.n_joints_, false);
    for (int i : c.ownedJointInds) owned[i] = true;

    double owned_sq = 0.0, other_sq = 0.0;
    std::printf("J_task column norms (rows = this arm's x, y, yaw task wrench)\n");
    for (int j = 0; j < c.n_joints_; ++j) {
        const double n = ts.J_task.col(j).norm();
        (owned[j] ? owned_sq : other_sq) += n * n;
        std::printf("  joint %d %-9s |col| = %.4f\n", j,
                    owned[j] ? "(owned)" : "(OTHER arm)", n);
    }
    const double frac = std::sqrt(other_sq) / std::sqrt(owned_sq + other_sq);
    std::printf("other-arm share of ||J_task|| = %.1f%%\n", 100.0 * frac);

    // Also the authority the QP thinks it has, with and without those columns.
    const Eigen::VectorXd bound_all = c.reachableWrenchBound(ts.J_task);
    Eigen::MatrixXd J_owned = ts.J_task;
    for (int j = 0; j < c.n_joints_; ++j)
        if (!owned[j]) J_owned.col(j).setZero();
    const Eigen::VectorXd bound_owned = c.reachableWrenchBound(J_owned);
    std::printf("reachable wrench bound  all joints: %s\n",
                (std::ostringstream{} << bound_all.transpose()).str().c_str());
    std::printf("reachable wrench bound owned only: %s\n",
                (std::ostringstream{} << bound_owned.transpose()).str().c_str());
}

/*! Where the per-tick heap allocations come from, so the hot-path refactor is
 *  ordered by payoff rather than by guesswork.
 *
 *  Counts each primitive in isolation and multiplies by the calls per tick the
 *  profile reports, then compares the total against what the whole-tick counter
 *  measures. A large residual means something allocates outside these primitives
 *  and a refactor targeting only them would miss it. */
TEST(DISABLED_AllocStudy, PerPrimitive)
{
    NMPCParams p = liveParams();
    ModelProbe c(kMjcf, p);
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    c.computeControl(s.q, s.v);          // warm up: one-off sizing out of the way

    const int N   = p.N;
    const int n_j = c.n_joints_;

    // Inputs for the primitives, built outside the measured scopes.
    const NodeDynamics nd = c.dynamicsAt(s.q, s.v, c.ee_site_id_);
    const ModelProbe::NodeTaskSpace ts =
        c.evalTaskSpace(s.q, s.v, nd.ee_pos, nd.ee_quat, nd.J_g6, nd.jdot_qdot6);
    const Eigen::VectorXd q0 = s.q.tail(n_j);
    Eigen::VectorXd tau_j = Eigen::VectorXd::Zero(n_j);

    struct Row { const char* name; long per_call; int calls; };
    std::vector<Row> rows;
    const auto measure = [&](const char* name, int calls, auto&& fn) {
        constexpr int kReps = 20;        // average out any lazy one-off inside
        long n = 0;
        {
            AllocScope scope;
            for (int i = 0; i < kReps; ++i) fn();
            n = scope.n();
        }
        rows.push_back({name, n / kReps, calls});
    };

    // Per rollout node. N+1 dynamicsAt (the terminal node too), N of the rest.
    // These call the OUT-PARAMETER overloads with a persistent destination, which
    // is what the rollout does; measuring the by-value forms instead would charge
    // the study for returns production does not make.
    NodeDynamics nd_out;
    ModelProbe::NodeTaskSpace ts_out;
    c.dynamicsAt(s.q, s.v, c.ee_site_id_, nd_out);        // warm its buffers first
    c.evalTaskSpace(s.q, s.v, nd.ee_pos, nd.ee_quat, nd.J_g6, nd.jdot_qdot6, ts_out);
    measure("dynamicsAt", N + 1, [&] {
        c.dynamicsAt(s.q, s.v, c.ee_site_id_, nd_out); });
    // The standalone overloads, for reference only. These are NOT a decomposition
    // of dynamicsAt: each rebuilds its own BaseElim, where dynamicsAt builds one
    // and shares it, which is why they sum to far more than the 16 it costs. They
    // are off the hot path; listed to show what the sharing already saves.
    measure("  [standalone] computeBaseElim", N + 1, [&] {
        ModelProbe::BaseElim be;
        c.computeBaseElim(nullptr, be); });
    measure("  [standalone] getDynamics", N + 1, [&] {
        Eigen::MatrixXd H_g; Eigen::VectorXd Cv;
        c.getDynamics(H_g, Cv, nullptr); });
    measure("  [standalone] gjm6", N + 1, [&] {
        volatile auto r = c.gjm6(c.ee_site_id_, nullptr); (void)r; });
    measure("  [standalone] jdotQdot6", N + 1, [&] {
        volatile auto r = c.jdotQdot6(c.ee_site_id_, nullptr); (void)r; });
    measure("evalTaskSpace", N, [&] {
        c.evalTaskSpace(s.q, s.v, nd.ee_pos, nd.ee_quat,
                        nd.J_g6, nd.jdot_qdot6, ts_out); });
    // Persistent destinations, again matching the rollout: these all write into
    // members there, so measuring them into fresh locals would charge the study
    // for allocations production has already hoisted out.
    Eigen::MatrixXd Hjt, Li, A_out, B_out;
    Eigen::VectorXd q1, v1;
    c.computeInertiaProducts(nd.H_g, ts.J_task, Hjt, Li);
    c.buildAugmentedModel(Li, ts.bias, Hjt, A_out, B_out);
    c.integrateStep(s.q, s.v, tau_j, p.Ts, q1, v1);
    measure("computeInertiaProducts", N, [&] {
        c.computeInertiaProducts(nd.H_g, ts.J_task, Hjt, Li); });
    measure("buildAugmentedModel", N, [&] {
        c.buildAugmentedModel(Li, ts.bias, Hjt, A_out, B_out); });
    measure("integrateStep", N, [&] {
        c.integrateStep(s.q, s.v, tau_j, p.Ts, q1, v1); });
    // Once per tick, also into persistent destinations as production does.
    Eigen::VectorXd lb, ub, lg, ug;
    Eigen::MatrixXd D;
    c.buildJointBoxBounds(q0, lb, ub);
    c.buildTorqueGeneralConstraint(ts.J_task, nd.Cv_joints, D, lg, ug);
    measure("buildJointBoxBounds", 1, [&] {
        c.buildJointBoxBounds(q0, lb, ub); });
    measure("buildTorqueGeneralConstraint", 1, [&] {
        c.buildTorqueGeneralConstraint(ts.J_task, nd.Cv_joints, D, lg, ug); });
    measure("finalizeJointTorques", 1, [&] {
        volatile auto r = c.finalizeJointTorques(
            ts.J_task, Eigen::Vector3d(0.05, -0.03, 0.01), nd.Cv_joints); (void)r; });

    std::printf("\nN=%d, %d model joints, %d owned\n", N, n_j, c.n_owned);
    std::printf("%-30s %9s %7s %9s\n", "primitive", "per call", "calls", "per tick");
    long total = 0;
    for (const auto& r : rows) {
        // The indented rows are components of dynamicsAt, already counted in it.
        const bool reference_only = r.name[0] == ' ';
        const long sub = r.per_call * r.calls;
        if (!reference_only) total += sub;
        std::printf("%-30s %9ld %7d %9ld%s\n", r.name, r.per_call, r.calls, sub,
                    reference_only ? "  (not counted)" : "");
    }

    // What the whole tick actually costs, for the residual.
    long whole = 0;
    {
        AllocScope scope;
        for (int i = 0; i < 10; ++i) c.computeControl(s.q, s.v);
        whole = scope.n();
    }
    whole /= 10;
    std::printf("%-30s %9s %7s %9ld\n", "--- primitives subtotal", "", "", total);
    std::printf("%-30s %9s %7s %9ld\n", "--- whole tick (measured)", "", "", whole);
    std::printf("%-30s %9s %7s %9ld  (%.0f%% unattributed)\n", "--- residual", "", "",
                whole - total, 100.0 * (whole - total) / std::max(whole, 1L));
}

/*! Which binary object the steady-state allocations belong to.
 *
 *  PerPrimitive localises them to a phase of the tick; this attributes them to
 *  the code that issues them, which is the question that decides whether a count
 *  is reachable. An allocation from this executable is this package's own (the
 *  lib is static, so its Eigen and STL call sites resolve here) and can be
 *  hoisted; one from libacados/libhpipm/libblasfeo is inside a precompiled
 *  dependency's per-solve path and cannot be, short of changing solver. */
TEST(DISABLED_AllocStudy, Attribution)
{
    NMPCParams p = liveParams();
    ModelProbe c(kMjcf, p);
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    // Several ticks first: the first sizes every buffer, and a warm-started
    // steady state is what the deployed loop actually runs.
    for (int i = 0; i < 5; ++i) c.computeControl(s.q, s.v);

    AllocTrace trace;
    c.computeControl(s.q, s.v);
    trace.stop();
    const long total     = trace.n();
    const bool truncated = trace.truncated();
    const auto rows      = trace.byObject();

    std::printf("\none steady-state tick: %ld allocations%s\n", total,
                truncated ? "  (TRACE BUFFER FULL -- counts truncated)" : "");
    std::printf("%-40s %9s %7s\n", "originating object", "allocs", "share");
    for (const auto& r : rows) {
        std::printf("%-40s %9ld %6.0f%%\n", r.first.c_str(), r.second,
                    100.0 * static_cast<double>(r.second) / std::max(total, 1L));
    }

    // Distinct call sites, as addr2line input: resolve with
    //   addr2line -f -C -i -e <object> <offset>
    std::printf("\n%-40s %18s %9s\n", "object", "offset", "allocs");
    for (const auto& st : trace.bySite()) {
        std::printf("%-40s %#18" PRIxPTR " %9ld\n",
                    st.object.c_str(), st.offset, st.count);
    }
}

TEST(DISABLED_WarmStartStudy, Ladder)
{
    // Diagnostic: the whole warm-start ladder on one scenario, for choosing a
    // mode and an iteration cap from measured numbers.
    std::printf("%-5s %9s %9s %9s %8s %8s %7s %8s %9s\n",
                "mode", "tick_mean", "tick_max", "over_bud", "it_mean",
                "it_max", "retries", "avail", "ee_err");
    for (int ws : {2, 1, 0}) {
        NMPCParams p = liveParams();
        p.qp_warm_start = ws;
        const RunMetrics m = runClosedLoop(p, warmStartScenario(), 400);
        std::printf("%-5d %9.0f %9.0f %9.3f %8.2f %8.0f %7ld %8.3f %9.5f\n",
                    ws, m.mean_tick_us, m.max_tick_us, m.frac_over_budget,
                    m.qp_iter_mean, m.qp_iter_max, m.qp_retries,
                    m.availability, m.mean_ee_err);
    }
}

TEST(DISABLED_ConstraintStudy, BoxIsActuallyEnforced)
{
    // Diagnostic: with joint2_L at 0.90 and q_min raised to 1.60, the Δq box for
    // that joint is [0.70, 2.24] while Δq is pinned to 0 at stage 0. Print what
    // the solver actually does with that, for both soft and hard.
    for (bool soft : {true, false}) {
        NMPCParams p = liveParams();
        p.joint_lims[0].q_min = 1.60;
        p.soft_joint_limits   = soft;
        ModelProbe c(kMjcf, p);
        State s = referenceState(c.n_joints_);
        const ArmNMPC::EePose st = c.currentEePose(s.q, s.v);
        c.setDesiredPos(st.pos.x() + 0.06, st.pos.y() + 0.06, st.pos.z());
        c.setDesiredOrient(st.quat(0), st.quat(1), st.quat(2), st.quat(3));

        c.computeControl(s.q, s.v);

        const int nx = 7 + 2 * c.n_joints_;
        Eigen::VectorXd x0(nx), x1(nx);
        c.qp_->getX(0, x0.data());
        c.qp_->getX(1, x1.data());
        const int row = 7 + c.ownedJointInds[0];   // Δq of the offending joint
        printf("DIAG soft=%d  qlim=[%.3f,%.3f] q0=%.3f -> box=[%.3f,%.3f]  "
               "x0[dq]=%.4f x1[dq]=%.4f  status=%d mode=%s nsbx(1)=%d\n",
               soft, c.qlimsByIndex[c.ownedJointInds[0]].first,
               c.qlimsByIndex[c.ownedJointInds[0]].second, s.q[7 + c.ownedJointInds[0]],
               c.qlimsByIndex[c.ownedJointInds[0]].first - s.q[7 + c.ownedJointInds[0]],
               c.qlimsByIndex[c.ownedJointInds[0]].second - s.q[7 + c.ownedJointInds[0]],
               x0[row], x1[row], c.lastQpStatus(),
               c.mode() == ArmNMPC::ControlMode::Damping ? "DAMP" : "nom",
               c.qp_->nsbxAt(1));
    }
}

/*! The experiment behind the constraint configuration: every hard/soft
 *  combination against three scenarios, with and without the terminal velocity
 *  constraint. */
TEST(DISABLED_ConstraintStudy, Ladder)
{
    struct Cfg { const char* name; bool soft_box; bool soft_tau; bool guard; };
    const Cfg cfgs[] = {
        {"soft/soft   ", true,  true,  false},
        {"soft/hardtau", true,  false, false},   // soft box needs no guard
        {"hard-raw    ", false, false, false},
        {"hard-guarded", false, false, true },
        {"hardbox-only", false, true,  true },
    };
    struct Sc { const char* name; Scenario sc; };
    const Sc scs[] = {
        {"nominal (constraints idle)", Scenario{0.06, 0.0}},
        {"aggressive setpoint",        Scenario{0.25, 0.0}},
        {"starts outside a limit",     Scenario{0.06, 1.60}},
    };

    printf("\nHard vs soft constraints. availability = fraction of ticks with an\n"
           "NMPC command (rest is safe-mode damping); maxdamp = longest unbroken\n"
           "damping run; pos_viol = worst realised excursion past the configured\n"
           "joint limits on the flown trajectory; relax = fraction of ticks the\n"
           "guard had to widen a box.\n"
           "NOTE a hard box provides no RESTORING force: soft slack penalises\n"
           "being outside and pulls back, hard+guard only forbids getting worse.\n");

    for (bool term_vel : {true, false}) {
        printf("\n=== terminal_velocity_constraint = %s ===\n", term_vel ? "true" : "false");
        for (const auto& s : scs) {
            printf("\n  scenario: %s\n", s.name);
            printf("    %-13s  avail  maxdamp  viol:init->final  recov   integral  "
                   "ee_err   box_slack  relax%%   tick_us(mean/max)  over%%  status\n", "config");
            for (const auto& cf : cfgs) {
                NMPCParams p = liveParams();
                p.terminal_velocity_constraint = term_vel;
                p.soft_joint_limits            = cf.soft_box;
                p.soft_torque_limits           = cf.soft_tau;
                p.relax_box_to_contain_x0      = cf.guard;
                const RunMetrics m = runClosedLoop(p, s.sc, 300);

                std::string hist;
                for (const auto& [st, n] : m.status_hist)
                    hist += std::to_string(st) + ":" + std::to_string(n) + " ";
                printf("    %-13s %5.1f%%  %7d  %6.4f->%-6.4f  %+6.4f  %7.4f  "
                       "%7.4f  %9.2e  %5.1f%%  %8.0f/%8.0f  %4.1f%%  %s\n",
                       cf.name, 100.0 * m.availability, m.max_consecutive_damping,
                       m.initial_pos_violation, m.final_pos_violation, m.recovered,
                       m.pos_violation_integral, m.mean_ee_err,
                       m.max_box_slack, 100.0 * m.guard_relax_frac,
                       m.mean_tick_us, m.max_tick_us, 100.0 * m.frac_over_budget,
                       hist.c_str());
            }
        }
    }
}

/*! Why the soft/soft configuration falls apart on aarch64, and what fixes it.
 *
 *  `HardConstraints.SoftBoxCarriesRealSlackWhenTheArmIsOutsideItsLimit` passes on
 *  x86-64 and fails on aarch64 under soft torque rows, with everything else in
 *  the suite green and the same commit on both. Measured, same scenario, soft
 *  box + soft torque + terminal_velocity_constraint:
 *
 *      x86-64   availability 96.7%, recovers 0.6994 -> 0.0000, status 0:231 2:9 3:60
 *      aarch64  availability 21.7%, recovers 0.6994 -> 0.0749, status 0:60  3:240
 *
 *  It needs BOTH soft torque rows and the terminal velocity constraint: harden the
 *  torque rows and aarch64 matches x86 exactly (96.7%, full recovery), and with
 *  terminal_velocity_constraint off the two agree to the printed digit on every
 *  column. So this is not a porting bug, it is an interior-point method sitting
 *  near infeasibility where the iterate path depends on the BLAS kernels --
 *  BLASFEO's ARMV8A_ARM_CORTEX_A57 set against X64_AUTOMATIC. x86 is not healthy
 *  here either, just luckier: 60 NaN solves in 300.
 *
 *  This sweeps the knobs that could buy the solver margin, on the configuration
 *  that fails. RUN IT ON THE FAILING ARCHITECTURE: x86 passes at 96.7%, so it
 *  cannot tell a setting that fixes aarch64 from one that does nothing. Use an
 *  x86 run only afterwards, to check the chosen value does not regress it.
 *
 *  Reads: availability and the recovery column are the outcome; the status
 *  histogram is the mechanism, where 3:N is N NaN solves. */
TEST(DISABLED_RegularisationStudy, SoftSoftNearInfeasibility)
{
    // The failing case exactly: arm parked ~0.7 rad outside joint 0's limit.
    const Scenario sc{0.06, 1.60, 0, 0.0};

    std::printf("\nsoft box + soft torque, terminal_velocity_constraint on, "
                "arm 0.7 rad outside its limit\n"
                "reg_prim is HPIPM's primal regularisation; tau_slack scales the "
                "soft TORQUE row penalty\n"
                "(linear and quadratic together), which is the fragile group here."
                "\n\n");
    std::printf("%-10s %-10s %7s %8s %-18s %8s  %s\n",
                "reg_prim", "tau_slack", "avail", "maxdamp", "viol:init->final",
                "tick_us", "status");

    for (const double reg : {1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2}) {
        for (const double sc_mult : {1.0, 10.0, 100.0}) {
            NMPCParams p = liveParams();
            p.soft_joint_limits  = true;
            p.soft_torque_limits = true;
            p.terminal_velocity_constraint = true;
            p.qp_reg_prim = reg;
            p.torque_slack_linear    *= sc_mult;
            p.torque_slack_quadratic *= sc_mult;

            const RunMetrics m = runClosedLoop(p, sc, 300);
            std::string hist;
            for (const auto& [st, n] : m.status_hist)
                hist += std::to_string(st) + ":" + std::to_string(n) + " ";
            std::printf("%-10.0e %-10.0f %6.1f%% %8d  %6.4f->%-6.4f  %8.0f  %s\n",
                        reg, sc_mult, 100.0 * m.availability,
                        m.max_consecutive_damping, m.initial_pos_violation,
                        m.final_pos_violation, m.mean_tick_us, hist.c_str());
        }
    }
    std::printf("\nThe current defaults are reg_prim 1e-6, tau_slack 1. A row with "
                "availability near\n100%% and a 'viol' column reaching 0.0000 is a "
                "candidate; verify it on x86 before\nchanging nmpc.yaml, and re-run "
                "DISABLED_ConstraintStudy.Ladder, since these knobs\nalso move the "
                "nominal and aggressive-setpoint cases.\n");
}

}  // namespace
}  // namespace pat_arm_control

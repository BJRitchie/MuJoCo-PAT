/*! Model-fidelity tests for the NMPC formulation.
 *
 *  These drive the prediction-model construction directly, through a subclass
 *  that re-exports the protected seams, and compare it against MuJoCo's own
 *  forward dynamics evaluated on the same state. Nothing here touches ROS or
 *  the QP solver -- the question being asked is only "does the linearised
 *  model predict the plant it is supposed to predict".
 *
 *  PAT_NMPC_TEST_MJCF is the build-tree pat_platform_planar.xml (see
 *  CMakeLists.txt); the model has a free-floating base plus two planar 3R
 *  arms, so nq = 13, nv = 12, n_joints_ = 6.
 */
#include <gtest/gtest.h>

#include <dlfcn.h>

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "pat_arm_control/control/arm_nmpc.h"
#include "pat_arm_control/observers/momentum_observer.h"
#include "pat_arm_control/rt/rt_mailbox.h"

// --- Heap-allocation counter ---------------------------------------------- //
// Interposes glibc's malloc family so a test can count the heap operations of a
// single control tick. Defined in the test executable, so calls from libstdc++,
// Eigen, MuJoCo and acados/BLASFEO all land here: each reaches malloc through
// the PLT, which resolves against the executable first. Counting malloc rather
// than operator new is deliberate -- operator new is implemented on top of
// malloc, and acados/BLASFEO call malloc directly.
//
// glibc-specific (__libc_* are its internal entry points) and deliberately not
// counting posix_memalign/aligned_alloc, which have no stable __libc_ pair. On
// x86-64 glibc, Eigen's aligned allocator is plain malloc (malloc already
// returns 16-byte-aligned blocks), so the hot path is covered.
extern "C" {
void* __libc_malloc(std::size_t);
void* __libc_calloc(std::size_t, std::size_t);
void* __libc_realloc(void*, std::size_t);
}

namespace pat_rt_alloc {
// Zero-initialised statics with no dynamic initialiser: allocations happen
// during static init, before any constructor here could have run.
volatile bool counting = false;
long          count    = 0;

// Call-site attribution, off unless a study turns it on. Only the IMMEDIATE
// caller's return address is kept: frame 0 is the one address available without
// a frame pointer, and resolving further would need backtrace(), which takes
// the loader lock and can itself allocate -- inside malloc that deadlocks. One
// frame is enough for the question these studies ask, because acados, HPIPM and
// BLASFEO call malloc directly, so frame 0 names their shared object. A caller
// reaching malloc through libstdc++'s operator new resolves to libstdc++ and is
// reported as such rather than guessed at.
//
// Addresses only, symbolised after the scope closes: dladdr allocates.
constexpr int kMaxSites = 4096;
volatile bool tracing   = false;
void*         sites[kMaxSites];
long          n_sites = 0;

inline void note(void* ret_addr) {
    ++count;
    if (tracing && n_sites < kMaxSites) sites[n_sites++] = ret_addr;
}
}  // namespace pat_rt_alloc

extern "C" void* malloc(std::size_t n) {
    if (pat_rt_alloc::counting) pat_rt_alloc::note(__builtin_return_address(0));
    return __libc_malloc(n);
}
extern "C" void* calloc(std::size_t n, std::size_t sz) {
    if (pat_rt_alloc::counting) pat_rt_alloc::note(__builtin_return_address(0));
    return __libc_calloc(n, sz);
}
extern "C" void* realloc(void* p, std::size_t n) {
    if (pat_rt_alloc::counting) pat_rt_alloc::note(__builtin_return_address(0));
    return __libc_realloc(p, n);
}

namespace pat_arm_control {
namespace {

/*! Counts allocations for as long as it is in scope. Read `n` after it closes:
 *  gtest's own assertion machinery allocates. */
struct AllocScope {
    AllocScope() { pat_rt_alloc::count = 0; pat_rt_alloc::counting = true; }
    ~AllocScope() { pat_rt_alloc::counting = false; }
    long n() const { return pat_rt_alloc::count; }
};

/*! AllocScope plus the call site of every allocation, grouped by the shared
 *  object that made it. Answers "whose allocation is this" rather than just how
 *  many, which is the difference between a site this package can remove and one
 *  inside a precompiled dependency. */
struct AllocTrace {
    AllocTrace() {
        pat_rt_alloc::n_sites = 0;
        pat_rt_alloc::count   = 0;
        pat_rt_alloc::tracing = true;
        pat_rt_alloc::counting = true;
    }
    ~AllocTrace() { stop(); }

    //! End the measured region. Must be called before reading the results:
    //! byObject() allocates, and left counting it would tally itself.
    void stop() const {
        pat_rt_alloc::counting = false;
        pat_rt_alloc::tracing  = false;
    }
    long n() const { return pat_rt_alloc::count; }

    //! Allocation count per originating object, most frequent first. Call stop()
    //! first; this allocates, and dladdr does too.
    std::vector<std::pair<std::string, long>> byObject() const {
        std::map<std::string, long> tally;
        for (long i = 0; i < pat_rt_alloc::n_sites; ++i) {
            Dl_info info{};
            const bool ok = dladdr(pat_rt_alloc::sites[i], &info) != 0;
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
        std::map<std::pair<std::string, std::uintptr_t>, long> tally;
        for (long i = 0; i < pat_rt_alloc::n_sites; ++i) {
            Dl_info info{};
            std::string object = "<unresolved>";
            std::uintptr_t offset = reinterpret_cast<std::uintptr_t>(pat_rt_alloc::sites[i]);
            if (dladdr(pat_rt_alloc::sites[i], &info) != 0 && info.dli_fname) {
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
    bool truncated() const { return pat_rt_alloc::n_sites >= pat_rt_alloc::kMaxSites; }
};

// Re-exports the protected model-construction seams so a test can build A/B at
// a known state rather than inferring them from closed-loop behaviour.
class ModelProbe : public ArmNMPC {
public:
    ModelProbe(const std::string& mjcf, const NMPCParams& p) : ArmNMPC(mjcf, p) {}

    using ArmNMPC::NodeTaskSpace;
    using ArmNMPC::buildAugmentedModel;
    using ArmNMPC::ee_site_id_;
    using ArmNMPC::enterDamping;
    using ArmNMPC::kd_;
    using ArmNMPC::torqueLimsByIndex;
    using ArmNMPC::vlimsByIndex;
    using ArmNMPC::evalTaskSpace;
    using ArmNMPC::n_owned;
    using ArmNMPC::ownedJointInds;
    using ArmNMPC::qpsolve;
    using ArmNMPC::lastQpStatus;
    using ArmNMPC::mode;
    using ArmNMPC::params_;
    using ArmNMPC::planStageForAge;
    using ArmNMPC::qlimsByIndex;
    using ArmNMPC::qp_;
    using ArmNMPC::reachableWrenchBound;
    using ArmNMPC::resetProfile;
    // Timing members, so a test can check the profile covers the tick.
    using ArmNMPC::build_model;
    using ArmNMPC::defect;
    using ArmNMPC::finalize;
    using ArmNMPC::pinv;
    using ArmNMPC::qp_iters;
    using ArmNMPC::qp_retries;
    using ArmNMPC::qp_time_ms;
    using ArmNMPC::rollout_dynamics;
    using ArmNMPC::rollout_integrate;
    using ArmNMPC::tick;
    using ArmNMPC::trust_region;
    using ArmNMPC::buildJointBoxBounds;
    using ArmNMPC::buildTorqueGeneralConstraint;
    // The live-state path, so FullDynamics.AgreesWithTheLiveStatePath can
    // compare the public Workspace query against the one a control tick uses.
    using ArmNMPC::getDynamics;
    using ArmNMPC::loadLiveState;
    using ArmNMPC::computeInertiaProducts;
    using ArmNMPC::finalizeJointTorques;
    using FloatingBaseDynamics::BaseElim;
    using FloatingBaseDynamics::computeBaseElim;
    using FloatingBaseDynamics::computeSiteJacobian;
    using FloatingBaseDynamics::dynamicsAt;
    using FloatingBaseDynamics::getDynamics;
    using FloatingBaseDynamics::gjm6;
    using FloatingBaseDynamics::integrateStep;
    using FloatingBaseDynamics::jdotQdot6;
    using FloatingBaseDynamics::modelJointRanges;
    using FloatingBaseDynamics::n_joints_;
    using FloatingBaseDynamics::lastMjError;
    using FloatingBaseDynamics::lastMjWarning;
    using FloatingBaseDynamics::mjErrorCount;
    using FloatingBaseDynamics::mjWarningCount;
    using FloatingBaseDynamics::pinvDLS;
    using FloatingBaseDynamics::solveSpd;
};

constexpr const char* kMjcf = PAT_NMPC_TEST_MJCF;

// Left arm, matching nmpc.yaml's per-instance shape: this controller owns one
// arm's three joints out of the model's six.
NMPCParams leftArmParams(double Ts = 0.01)
{
    NMPCParams p;
    p.Ts             = Ts;
    p.N              = 5;
    p.ee_site_name   = "ee_site_L";
    p.fullNonlinear  = true;
    // Hg_damping deliberately left at the NMPCParams default so the
    // over-damping test below tracks that default rather than a local choice.
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

struct State {
    Eigen::VectorXd q, v;
};

/*! A deliberately generic state: base at the origin with identity orientation,
 *  both arms away from any singular (straight or folded) configuration, and
 *  every joint moving so the Coriolis terms are actually exercised.
 *
 *  Every angle sits strictly inside the MJCF joint ranges (joint2 [0, 3.14],
 *  joint3 [-2.967, 0], joint5 [-1.22, 1.22]). This matters: MuJoCo enforces
 *  those ranges with constraint forces, and a state outside them accelerates
 *  violently under forces the prediction model has no term for, which would
 *  make every comparison below meaningless. */
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

// --- Dynamics reduction ---------------------------------------------------

TEST(NmpcDynamics, GeneralizedInertiaIsSymmetricPositiveDefinite)
{
    ModelProbe c(kMjcf, leftArmParams());
    const State s = referenceState(c.n_joints_);
    const NodeDynamics nd = c.dynamicsAt(s.q, s.v, c.ee_site_id_);

    ASSERT_EQ(nd.H_g.rows(), c.n_joints_);
    ASSERT_EQ(nd.H_g.cols(), c.n_joints_);
    EXPECT_LT((nd.H_g - nd.H_g.transpose()).cwiseAbs().maxCoeff(), 1e-9)
        << "H_g = H_mm - H_bm^T H_bb^-1 H_bm is symmetric by construction";

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(nd.H_g);
    EXPECT_GT(es.eigenvalues().minCoeff(), 0.0)
        << "eigenvalues: " << es.eigenvalues().transpose();
}

TEST(NmpcDynamics, GeneralizedInertiaInverseIsNotOverDamped)
{
    // pinvDLS damps with s/(s^2 + lambda^2), so the damping only leaves the
    // inverse intact while H_g's singular values stay well above Hg_damping.
    // A planar arm's generalised inertia is small in absolute terms, so this
    // is a real risk rather than a theoretical one: over-damping here scales
    // Lambda_inv down by the same factor and the QP then models an arm far
    // more sluggish than the plant.
    ModelProbe c(kMjcf, leftArmParams());
    const State s = referenceState(c.n_joints_);
    const NodeDynamics nd = c.dynamicsAt(s.q, s.v, c.ee_site_id_);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(nd.H_g);
    const double min_sv = svd.singularValues()(svd.singularValues().size() - 1);

    const Eigen::MatrixXd damped = c.pinvDLS(nd.H_g, c.params_.Hg_damping);
    const Eigen::MatrixXd exact =
        nd.H_g.ldlt().solve(Eigen::MatrixXd::Identity(c.n_joints_, c.n_joints_));

    const double rel = (damped - exact).norm() / exact.norm();
    EXPECT_LT(rel, 0.01)
        << "damped H_g^-1 differs from the exact solve by " << (100.0 * rel)
        << "%.  Hg_damping=" << c.params_.Hg_damping
        << ", H_g singular values=" << svd.singularValues().transpose()
        << " (smallest " << min_sv << ")";
}

TEST(NmpcDynamics, GeneralizedBiasVanishesAtRest)
{
    // Gravity is disabled in the MJCF, so at zero velocity there is no
    // Coriolis/centrifugal term for the base elimination to carry through.
    ModelProbe c(kMjcf, leftArmParams());
    State s = referenceState(c.n_joints_);
    s.v.setZero();

    const NodeDynamics nd = c.dynamicsAt(s.q, s.v, c.ee_site_id_);
    EXPECT_LT(nd.Cv_joints.cwiseAbs().maxCoeff(), 1e-9)
        << "Cv_joints = " << nd.Cv_joints.transpose();
}

TEST(NmpcDynamics, GeneralizedBiasIsNonZeroWhenMoving)
{
    // Guards the test above from passing for the wrong reason (a bias that is
    // always zero would satisfy it trivially).
    ModelProbe c(kMjcf, leftArmParams());
    const State s = referenceState(c.n_joints_);
    const NodeDynamics nd = c.dynamicsAt(s.q, s.v, c.ee_site_id_);
    EXPECT_GT(nd.Cv_joints.cwiseAbs().maxCoeff(), 1e-12);
}

// --- Joint limits ---------------------------------------------------------

/*! The model build gets H_g's damped inverse from a Cholesky of H_g^2 + d^2 I
 *  rather than an SVD. That is an identity for symmetric H_g, not an
 *  approximation, and this pins it: if the two ever diverge, the prediction model
 *  has silently changed. */
TEST(NmpcDynamics, CholeskyDampedInverseMatchesTheSvdOne)
{
    ModelProbe c(kMjcf, leftArmParams());
    const State s = referenceState(c.n_joints_);
    const NodeDynamics nd = c.dynamicsAt(s.q, s.v, c.ee_site_id_);
    const double lambda = c.params_.Hg_damping;

    const Eigen::MatrixXd via_svd = c.pinvDLS(nd.H_g, lambda);

    Eigen::MatrixXd G = nd.H_g * nd.H_g;
    G.diagonal().array() += lambda * lambda;
    Eigen::MatrixXd via_chol;
    ASSERT_TRUE(c.solveSpd(G, nd.H_g, via_chol));

    EXPECT_LT((via_chol - via_svd).norm() / via_svd.norm(), 1e-9)
        << "chol:\n" << via_chol << "\nsvd:\n" << via_svd;
}

TEST(NmpcLimits, EveryModelJointGetsItsMjcfRange)
{
    // The controller owns only the left arm, but the rollout advances all six
    // joints and MuJoCo enforces all six ranges. The right arm's bounds must
    // therefore be the model's, not a placeholder -- a rollout that walks a
    // non-owned joint past its range meets constraint forces the linearised
    // model cannot represent.
    ModelProbe c(kMjcf, leftArmParams());
    const auto ranges = c.modelJointRanges();

    ASSERT_EQ(static_cast<int>(ranges.size()), c.n_joints_);
    for (int i = 0; i < c.n_joints_; ++i) {
        EXPECT_LT(ranges[i].first, ranges[i].second) << "joint " << i;
        EXPECT_GT(ranges[i].first, -M_PI - 1e-9) << "joint " << i;
        EXPECT_LT(ranges[i].second, M_PI + 1e-9) << "joint " << i;
    }

    // joint2 [0, 3.14] / joint3 [-2.967, 0] / joint5 [-1.22, 1.22], per arm,
    // in MuJoCo body-tree order (left arm then right).
    const double expect_lo[6] = {0.0, -2.967, -1.22, 0.0, -2.967, -1.22};
    const double expect_hi[6] = {3.14,  0.0,   1.22, 3.14,  0.0,   1.22};
    for (int i = 0; i < c.n_joints_; ++i) {
        EXPECT_NEAR(ranges[i].first,  expect_lo[i], 1e-9) << "joint " << i << " lower";
        EXPECT_NEAR(ranges[i].second, expect_hi[i], 1e-9) << "joint " << i << " upper";
    }
}

TEST(NmpcLimits, TrustRegionTracksTheReachableWrench)
{
    // The trust region is a fraction of this bound, so it has to sit in the
    // same range as the wrench the arm can actually produce -- the defect the
    // old fixed 50 N / 5 N*m pair had was being orders of magnitude above it,
    // which made it unable to bind at all.
    ModelProbe c(kMjcf, leftArmParams());
    const State s = referenceState(c.n_joints_);
    const NodeDynamics nd = c.dynamicsAt(s.q, s.v, c.ee_site_id_);
    const ModelProbe::NodeTaskSpace ts =
        c.evalTaskSpace(s.q, s.v, nd.ee_pos, nd.ee_quat, nd.J_g6, nd.jdot_qdot6);

    const Eigen::VectorXd bound = c.reachableWrenchBound(ts.J_task);
    ASSERT_EQ(bound.size(), 3);

    // A wrench at the bound must be deliverable: applied on its own channel it
    // may not ask any joint for more than its configured limit.
    for (int j = 0; j < 3; ++j) {
        EXPECT_GT(bound(j), 0.0) << "channel " << j;
        Eigen::Vector3d u = Eigen::Vector3d::Zero();
        u(j) = bound(j);
        const Eigen::VectorXd tau = ts.J_task.transpose() * u;
        for (int i = 0; i < c.n_joints_; ++i) {
            EXPECT_LE(std::abs(tau[i]), 0.25 + 1e-9)
                << "channel " << j << " at its bound overloads joint " << i
                << " (tau " << tau[i] << ")";
        }
    }

    // And it must be tight rather than merely safe: scaling past the bound has
    // to overload something, or the region is loose enough never to bind.
    for (int j = 0; j < 3; ++j) {
        Eigen::Vector3d u = Eigen::Vector3d::Zero();
        u(j) = 1.01 * bound(j);
        const Eigen::VectorXd tau = ts.J_task.transpose() * u;
        EXPECT_GT(tau.cwiseAbs().maxCoeff(), 0.25)
            << "channel " << j << " bound " << bound(j) << " is not tight";
    }
}

// --- Rollout integrator ---------------------------------------------------

TEST(NmpcIntegrator, RestStaysAtRestWithNoAppliedTorque)
{
    // Gravity is off and the rollout applies torque only to the joint rows, so
    // a state at rest under zero torque must not move at all. Anything else
    // means a force the prediction model knows nothing about -- contact being
    // the obvious candidate, since the reference configuration is chosen by
    // hand and the internal model still carries the bus and arm geoms.
    const double Ts = 0.01;
    ModelProbe c(kMjcf, leftArmParams(Ts));
    State s = referenceState(c.n_joints_);
    s.v.setZero();

    Eigen::VectorXd q1, v1;
    c.integrateStep(s.q, s.v, Eigen::VectorXd::Zero(c.n_joints_), Ts, q1, v1);

    EXPECT_LT((v1 - s.v).cwiseAbs().maxCoeff(), 1e-9)
        << "velocity moved to " << v1.transpose();
    EXPECT_LT((q1 - s.q).cwiseAbs().maxCoeff(), 1e-9)
        << "position moved by " << (q1 - s.q).transpose();
}

TEST(NmpcIntegrator, JointPositionUpdateUsesMidIntervalVelocity)
{
    // Under a torque held constant across the step, the joint position must
    // advance by Ts*v + Ts^2/2*a, matching the Ts^2/2 coefficient the
    // augmented model's own task-space rows assume. Integrating with the
    // end-of-step velocity instead would double that term, so asserting the
    // residual is a small fraction of it discriminates between the two.
    const double Ts = 0.01;
    ModelProbe c(kMjcf, leftArmParams(Ts));
    const State s = referenceState(c.n_joints_);
    const int n_j = c.n_joints_;

    const NodeDynamics nd = c.dynamicsAt(s.q, s.v, c.ee_site_id_);
    const ModelProbe::NodeTaskSpace ts =
        c.evalTaskSpace(s.q, s.v, nd.ee_pos, nd.ee_quat, nd.J_g6, nd.jdot_qdot6);

    const Eigen::MatrixXd H_g_inv = c.pinvDLS(nd.H_g, c.params_.Hg_damping);
    const Eigen::Vector3d u(0.08, -0.05, 0.01);          // task wrench [N, N, N*m]
    const Eigen::VectorXd tau_j = ts.J_task.transpose() * u + nd.Cv_joints;

    // The Cv feedforward cancels the generalised bias exactly, leaving
    // qddot = H_g^-1 J^T u.
    const Eigen::VectorXd a = H_g_inv * ts.J_task.transpose() * u;

    Eigen::VectorXd q1, v1;
    c.integrateStep(s.q, s.v, tau_j, Ts, q1, v1);

    const Eigen::VectorXd dq_actual = q1.tail(n_j) - s.q.tail(n_j);
    const Eigen::VectorXd dq_expect = Ts * s.v.tail(n_j) + 0.5 * Ts * Ts * a;
    const double accel_term = (0.5 * Ts * Ts * a).norm();

    ASSERT_GT(accel_term, 0.0) << "test wrench produced no joint acceleration";
    EXPECT_LT((dq_actual - dq_expect).norm(), 0.1 * accel_term)
        << "residual " << (dq_actual - dq_expect).norm()
        << " vs Ts^2/2*a term " << accel_term;

    EXPECT_LT((v1.tail(n_j) - (s.v.tail(n_j) + Ts * a)).norm(), 0.05 * (Ts * a).norm())
        << "velocity update should be Ts*a to first order";
}

// --- Augmented prediction model ------------------------------------------

/*! Builds A/B at (q,v) and returns the model's one-step prediction alongside
 *  the plant's true one-step response, in the same augmented coordinates. */
struct OneStepComparison {
    Eigen::VectorXd x0, x_pred, x_true;
    //! n_q is the number of joint rows the STATE carries -- the owned joints only,
    //! not every model joint. The non-owned ones were dropped from the QP because
    //! they appear in neither the cost nor the constraints.
    int d = 3, n_q = 0;

    Eigen::VectorXd taskBlock(const Eigen::VectorXd& x) const { return x.head(2 * d); }
    Eigen::VectorXd jointBlock(const Eigen::VectorXd& x) const
    {
        return x.segment(2 * d + 1, 2 * n_q);
    }
};

OneStepComparison compareOneStep(ModelProbe& c, const Eigen::Vector3d& u)
{
    const int n_j   = c.n_joints_;
    const int d     = 3;
    const double Ts = c.params_.Ts;
    const State s   = referenceState(n_j);

    const NodeDynamics nd0 = c.dynamicsAt(s.q, s.v, c.ee_site_id_);
    const ModelProbe::NodeTaskSpace ts0 =
        c.evalTaskSpace(s.q, s.v, nd0.ee_pos, nd0.ee_quat, nd0.J_g6, nd0.jdot_qdot6);

    const Eigen::MatrixXd H_g_inv = c.pinvDLS(nd0.H_g, c.params_.Hg_damping);
    const Eigen::MatrixXd H_g_inv_Jt = H_g_inv * ts0.J_task.transpose();
    const Eigen::MatrixXd Lambda_inv = ts0.J_task * H_g_inv_Jt;

    Eigen::MatrixXd A, B;
    c.buildAugmentedModel(Lambda_inv, ts0.bias, H_g_inv_Jt, A, B);

    OneStepComparison out;
    out.d   = d;
    out.n_q = c.n_owned;

    const int n_q = c.n_owned;
    const int nx = 2 * d + 1 + 2 * n_q;
    out.x0 = Eigen::VectorXd::Zero(nx);
    out.x0.segment(0, d)             = ts0.e;
    out.x0.segment(d, d)             = ts0.edot;
    out.x0(2 * d)                    = 1.0;
    // Owned joints, in ownedJointInds order: the state's layout. Δq starts at
    // zero because it is measured relative to this tick's q0.
    for (int i = 0; i < n_q; ++i)
        out.x0(2 * d + 1 + n_q + i) = s.v(6 + c.ownedJointInds[i]);

    out.x_pred = A * out.x0 + B * u;

    // True plant response to the same wrench, deliberately unclamped so the
    // comparison stays inside the linear regime the model describes.
    const Eigen::VectorXd tau_j = ts0.J_task.transpose() * u + nd0.Cv_joints;
    Eigen::VectorXd q1, v1;
    c.integrateStep(s.q, s.v, tau_j, Ts, q1, v1);

    const NodeDynamics nd1 = c.dynamicsAt(q1, v1, c.ee_site_id_);
    const ModelProbe::NodeTaskSpace ts1 =
        c.evalTaskSpace(q1, v1, nd1.ee_pos, nd1.ee_quat, nd1.J_g6, nd1.jdot_qdot6);

    out.x_true = Eigen::VectorXd::Zero(nx);
    out.x_true.segment(0, d)             = ts1.e;
    out.x_true.segment(d, d)             = ts1.edot;
    out.x_true(2 * d)                    = 1.0;
    for (int i = 0; i < n_q; ++i) {
        const int j = c.ownedJointInds[i];
        out.x_true(2 * d + 1 + i)       = q1(7 + j) - s.q(7 + j);
        out.x_true(2 * d + 1 + n_q + i) = v1(6 + j);
    }

    return out;
}

TEST(NmpcModel, TaskRowsPredictPlantOneStep)
{
    ModelProbe c(kMjcf, leftArmParams());
    const OneStepComparison r = compareOneStep(c, Eigen::Vector3d(0.08, -0.05, 0.01));

    const double err    = (r.taskBlock(r.x_pred) - r.taskBlock(r.x_true)).norm();
    const double change = (r.taskBlock(r.x_true) - r.taskBlock(r.x0)).norm();

    ASSERT_GT(change, 1e-9) << "the step produced no task-space motion to compare";
    EXPECT_LT(err, 0.05 * change)
        << "task rows captured " << (100.0 * (1.0 - err / change))
        << "% of the true one-step change (err " << err << ", change " << change << ")";
}

TEST(NmpcModel, JointRowsPredictPlantOneStep)
{
    ModelProbe c(kMjcf, leftArmParams());
    const OneStepComparison r = compareOneStep(c, Eigen::Vector3d(0.08, -0.05, 0.01));

    const double err    = (r.jointBlock(r.x_pred) - r.jointBlock(r.x_true)).norm();
    const double change = (r.jointBlock(r.x_true) - r.jointBlock(r.x0)).norm();

    ASSERT_GT(change, 1e-9) << "the step produced no joint motion to compare";
    EXPECT_LT(err, 0.05 * change)
        << "joint rows captured " << (100.0 * (1.0 - err / change))
        << "% of the true one-step change (err " << err << ", change " << change << ")";
}

TEST(NmpcModel, JointRowsRespondToControl)
{
    // The joint Δq/qdot rows carry the box constraints, so they have to move
    // when the control moves -- a model whose B block is zero on those rows
    // predicts the same joint trajectory whatever torque the QP chooses, and
    // the joint limits stop constraining anything the solver can act on.
    ModelProbe c(kMjcf, leftArmParams());
    const OneStepComparison a = compareOneStep(c, Eigen::Vector3d( 0.08, -0.05,  0.01));
    const OneStepComparison b = compareOneStep(c, Eigen::Vector3d(-0.08,  0.05, -0.01));

    const double pred_spread = (a.jointBlock(a.x_pred) - b.jointBlock(b.x_pred)).norm();
    const double true_spread = (a.jointBlock(a.x_true) - b.jointBlock(b.x_true)).norm();

    ASSERT_GT(true_spread, 1e-9) << "opposing wrenches produced no joint-motion difference";
    EXPECT_GT(pred_spread, 0.5 * true_spread)
        << "model predicts a " << pred_spread << " joint-row spread between opposing "
        << "wrenches where the plant gives " << true_spread;
}

// --- QP constraint contract ----------------------------------------------
//
// A double integrator, small enough that the answer is checkable by hand:
// drive position from 1 to 0 with the terminal velocity pinned. The point is
// not the trajectory but the two HARD constraints -- the stage-0 state pin and
// the terminal velocity row -- both of which reach HPIPM as box bounds whose
// lower and upper halves are equal. Anything that changes how those are
// declared to the solver has to keep both exactly satisfied.

struct DoubleIntegratorQP {
    quad_prob_solver::QuadProbSolverParams params;
    std::vector<Eigen::MatrixXd> A_k, B_k, D_k;
    std::vector<Eigen::VectorXd> b_k, lg_k, ug_k;
    Eigen::MatrixXd Q, R, Q_N;
    Eigen::VectorXd x0, lbx, ubx;
};

/*! State is [position, velocity, dummy]. The dummy is decoupled and exists only
 *  to push the index arrays out of alignment: with idxbx_k = {0, 2}, stage N's
 *  box list is [0, 2, 1], so the terminal velocity row sits at box POSITION 2
 *  while naming STATE 1. HPIPM's idxbxe is documented nowhere in the installed
 *  headers, so that misalignment is what tells the two candidate conventions
 *  apart: declare the equality by box position and the terminal velocity is
 *  pinned; declare it by state index and row 2 (the dummy's soft box) gets
 *  eliminated instead, leaving velocity free. */
DoubleIntegratorQP makeDoubleIntegratorQP(bool pin_terminal_velocity)
{
    const double Ts = 0.1;
    const int N = 6;

    DoubleIntegratorQP p;
    p.params.nx_task = 3;
    p.params.nx      = 3;
    p.params.nu      = 1;
    p.params.N       = N;
    // One general row bounding the control, standing in for the real
    // controller's torque constraint. Without it `u` is unbounded and ANY box is
    // reachable in a single step, which would make every hard-box test vacuous.
    p.params.ng      = 1;
    p.params.qp_max_iter = 200;
    p.params.qp_tol      = 1e-9;
    p.params.idxbx_k = {0, 2};
    if (pin_terminal_velocity) p.params.idxbx_terminal = {1};

    p.A_k.assign(N, Eigen::MatrixXd::Identity(3, 3));
    p.B_k.assign(N, Eigen::MatrixXd::Zero(3, 1));
    p.b_k.assign(N, Eigen::VectorXd::Zero(3));
    for (int k = 0; k < N; ++k) {
        p.A_k[k](0, 1) = Ts;
        p.B_k[k](0, 0) = 0.5 * Ts * Ts;
        p.B_k[k](1, 0) = Ts;
    }
    // |u| <= 10: enough authority to track, far too little to jump a box in one
    // step, which is what makes an out-of-box x0 genuinely infeasible.
    p.D_k.assign(N, Eigen::MatrixXd::Constant(1, 1, 1.0));
    p.lg_k.assign(N, Eigen::VectorXd::Constant(1, -10.0));
    p.ug_k.assign(N, Eigen::VectorXd::Constant(1,  10.0));

    p.Q = Eigen::MatrixXd::Zero(3, 3);
    p.Q(0, 0) = 1.0;
    p.R = Eigen::MatrixXd::Constant(1, 1, 1e-4);   // cheap control: drives hard, still moving at N
    p.Q_N = p.Q * 10.0;

    p.x0 = Eigen::VectorXd(3);
    p.x0 << 1.0, 0.0, 0.0;
    p.lbx = Eigen::VectorXd(2);   // one per idxbx_k row, wide enough not to bind
    p.ubx = Eigen::VectorXd(2);
    p.lbx << -10.0, -10.0;
    p.ubx <<  10.0,  10.0;
    return p;
}

bool solveQP(quad_prob_solver::QuadProbSolver& qp, DoubleIntegratorQP& p)
{
    return qp.solveMultiStage(
        p.A_k, p.B_k, p.b_k, p.Q.data(), p.R.data(),
        p.lbx.data(), p.ubx.data(),
        p.D_k, p.lg_k, p.ug_k, p.x0.data(), p.Q_N.data());
}

TEST(NmpcQP, TerminalVelocityIsFreeWithoutTheConstraint)
{
    // Establishes that the pin test below is meaningful: without the terminal
    // row, the optimum really does arrive at stage N still moving, so a passing
    // pin cannot be an accident of the cost.
    DoubleIntegratorQP p = makeDoubleIntegratorQP(/*pin_terminal_velocity=*/false);
    quad_prob_solver::QuadProbSolver qp(p.params);
    ASSERT_TRUE(solveQP(qp, p));

    Eigen::VectorXd x_at_N(3);
    qp.getX(p.params.N, x_at_N.data());
    EXPECT_GT(std::abs(x_at_N(1)), 1e-2)
        << "unpinned terminal velocity is " << x_at_N(1)
        << ", too close to zero for the pin test to prove anything";
}

TEST(NmpcQP, HardBoundsAreSatisfiedExactly)
{
    DoubleIntegratorQP p = makeDoubleIntegratorQP(/*pin_terminal_velocity=*/true);
    quad_prob_solver::QuadProbSolver qp(p.params);
    ASSERT_TRUE(solveQP(qp, p)) << "double-integrator QP failed to solve";

    // Stage-0 pin: the solver's own stage-0 state must be the state handed in.
    Eigen::VectorXd x_at_0(3);
    qp.getX(0, x_at_0.data());
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(x_at_0(i), p.x0(i), 1e-8) << "stage-0 pin, row " << i;
    }

    // Terminal velocity row: hard, so it holds to solver tolerance. This is the
    // assertion that distinguishes the two idxbxe index conventions.
    Eigen::VectorXd x_at_N(3);
    qp.getX(p.params.N, x_at_N.data());
    EXPECT_NEAR(x_at_N(1), 0.0, 1e-6) << "terminal velocity should be pinned to zero";

    // And the solve is actually doing something: position is driven toward 0
    // from 1, so the first control must push back (negative).
    Eigen::VectorXd u0(1);
    qp.getU(0, u0.data());
    EXPECT_LT(u0(0), 0.0) << "u0 = " << u0(0);

    // The dummy row is decoupled (A = I, B = 0) and starts at zero, so it must
    // stay at zero at every stage. This is what actually catches a mis-declared
    // equality: declaring the wrong box row as an equality does NOT unpin the
    // terminal velocity — lb == ub still enforces that on its own — it instead
    // tells HPIPM that some OTHER row's genuine inequality is an equality, and
    // that row then gets collapsed onto a bound.
    for (int k = 0; k <= p.params.N; ++k) {
        Eigen::VectorXd xk(3);
        qp.getX(k, xk.data());
        EXPECT_NEAR(xk(2), 0.0, 1e-6)
            << "decoupled row moved at stage " << k << " (to " << xk(2)
            << ") — a box row that is not an equality has been eliminated as one";
    }
}

TEST(NmpcQP, SoftLayoutMatchesWhatTheSolverWasBuiltWith)
{
    // nsAt/nsbxAt/nsgAt must describe the wiring for every flag combination --
    // this is what lets a slack reader slice by asking instead of assuming, and
    // in the real model the box and general counts are equal, so a wrong split
    // would go unnoticed.
    for (bool soft_box : {true, false}) {
        for (bool soft_tau : {true, false}) {
            DoubleIntegratorQP p = makeDoubleIntegratorQP(true);
            p.params.soft_joint_limits  = soft_box;
            p.params.soft_torque_limits = soft_tau;
            quad_prob_solver::QuadProbSolver qp(p.params);

            const int nbx_k = static_cast<int>(p.params.idxbx_k.size());
            const int N     = p.params.N;
            for (int k = 0; k <= N; ++k) {
                const int want_box = (soft_box && k > 0) ? nbx_k : 0;
                const int want_gen = (soft_tau && k < N) ? p.params.ng : 0;
                EXPECT_EQ(qp.nsbxAt(k), want_box)
                    << "stage " << k << " box, soft_box=" << soft_box;
                EXPECT_EQ(qp.nsgAt(k), want_gen)
                    << "stage " << k << " general, soft_tau=" << soft_tau;
                EXPECT_EQ(qp.nsAt(k), want_box + want_gen) << "stage " << k;
            }
        }
    }
}

TEST(NmpcQP, HardBoxGoesInfeasibleWhenX0IsOutsideIt)
{
    // The mechanism behind "hard constraints failed at every loop": the box
    // constrains stages 1..N, the stage-0 pin fixes where the trajectory starts,
    // and bounded control cannot travel far in one step. Put x0 outside the box
    // and there is no feasible trajectory at all.
    DoubleIntegratorQP p = makeDoubleIntegratorQP(false);
    p.params.soft_joint_limits  = false;
    p.params.soft_torque_limits = false;   // else the plan buys its way out, below
    p.x0 << 50.0, 0.0, 0.0;                // position box is [-10, 10]

    quad_prob_solver::QuadProbSolver qp(p.params);
    EXPECT_FALSE(solveQP(qp, p))
        << "hard box with x0 outside it should not solve; status "
        << qp.lastStatus();
}

TEST(NmpcQP, SoftBoxToleratesX0OutsideIt)
{
    // The same problem stays solvable when the box is soft -- the contrast that
    // makes the hard-vs-soft comparison meaningful at all.
    DoubleIntegratorQP p = makeDoubleIntegratorQP(false);
    p.params.soft_joint_limits = true;
    p.x0 << 50.0, 0.0, 0.0;

    quad_prob_solver::QuadProbSolver qp(p.params);
    EXPECT_TRUE(solveQP(qp, p)) << "status " << qp.lastStatus();
}

TEST(NmpcQP, HardBoxWithSoftTorqueFailsWithANaNSolution)
{
    // The mixed configuration is the worst of the three, and worth pinning. A
    // hard box the state cannot reach, combined with a soft control bound the
    // solver can buy its way through, does not produce a clean infeasibility --
    // it produces status 3, a NaN solution. Consistent with the HPIPM behaviour
    // recorded on QuadProbSolverParams::ng, where soft general rows are the
    // fragile ones. The caller's allFinite() checks turn this into a failed
    // solve, so it reaches safe mode rather than the actuators, but anyone
    // choosing to harden only the box should know this is the failure they get.
    DoubleIntegratorQP p = makeDoubleIntegratorQP(false);
    p.params.soft_joint_limits  = false;
    p.params.soft_torque_limits = true;
    p.x0 << 50.0, 0.0, 0.0;

    quad_prob_solver::QuadProbSolver qp(p.params);
    EXPECT_FALSE(solveQP(qp, p));
    EXPECT_EQ(qp.lastStatus(), 3) << "expected NAN_SOL (3), got " << qp.lastStatus();
}

TEST(NmpcQP, GuardMakesAHardBoxSolvableAndReportsTheRelaxation)
{
    DoubleIntegratorQP p = makeDoubleIntegratorQP(false);
    p.params.soft_joint_limits         = false;
    p.params.relax_box_to_contain_x0   = true;
    p.params.relax_box_margin          = 1e-2;
    p.x0 << 50.0, 0.0, 0.0;

    quad_prob_solver::QuadProbSolver qp(p.params);
    ASSERT_TRUE(solveQP(qp, p)) << "status " << qp.lastStatus();

    // The pin is still honoured exactly — the guard widens the box, it does not
    // move the trajectory's start.
    Eigen::VectorXd x_at_0(3);
    qp.getX(0, x_at_0.data());
    EXPECT_NEAR(x_at_0(0), 50.0, 1e-8);

    // And it reports what it had to give, so a study can tell a guarded hard box
    // from no box at all.
    EXPECT_GT(qp.lastRelaxCount(), 0);
    EXPECT_GT(qp.lastRelaxMax(), 0.0);
}

TEST(NmpcQP, GuardIsInertWhenX0IsAlreadyInsideTheBox)
{
    // A ratchet, not a blanket relaxation: with the state inside its bounds the
    // guard must change nothing, or "hard + guard" would quietly mean "no box".
    DoubleIntegratorQP p = makeDoubleIntegratorQP(false);
    p.params.soft_joint_limits       = false;
    p.params.relax_box_to_contain_x0 = true;
    p.params.relax_box_margin        = 1e-3;
    p.x0 << 1.0, 0.0, 0.0;           // well inside [-10, 10]

    quad_prob_solver::QuadProbSolver qp(p.params);
    ASSERT_TRUE(solveQP(qp, p)) << "status " << qp.lastStatus();
    EXPECT_EQ(qp.lastRelaxCount(), 0);
    EXPECT_DOUBLE_EQ(qp.lastRelaxMax(), 0.0);
}

TEST(NmpcQP, DynamicsAreConsistentAcrossStages)
{
    // Guards the pin test above: a solver that satisfied the bounds but
    // ignored the dynamics would still pass it.
    DoubleIntegratorQP p = makeDoubleIntegratorQP(/*pin_terminal_velocity=*/true);
    quad_prob_solver::QuadProbSolver qp(p.params);
    ASSERT_TRUE(solveQP(qp, p));

    for (int k = 0; k < p.params.N; ++k) {
        Eigen::VectorXd xk(3), xk1(3), uk(1);
        qp.getX(k, xk.data());
        qp.getX(k + 1, xk1.data());
        qp.getU(k, uk.data());
        const Eigen::VectorXd pred = p.A_k[k] * xk + p.B_k[k] * uk + p.b_k[k];
        EXPECT_LT((xk1 - pred).cwiseAbs().maxCoeff(), 1e-7)
            << "stage " << k << " violates its own dynamics";
    }
}

// --- Soft-constraint slack -----------------------------------------------

NMPCParams liveParams()
{
    // The shape nmpc.yaml actually runs: SQP path, N=30, Ts=0.033, 100 Hz,
    // qp.max_iter 60. Keep these in step with config/nmpc.yaml -- every harness
    // measurement is read as representative of the deployed controller, so a drift
    // here silently invalidates the comparison. The control period matters beyond
    // Tier 1: the warm-start shift advances by it, so leaving it unset makes the
    // harness shift a whole stage per tick where the real one shifts 0.3 of one.
    NMPCParams p = leftArmParams(0.033);
    p.control_period = 0.010;
    p.N = 30;
    p.sqp_iters = 1;
    p.fullNonlinear = true;
    p.terminal_velocity_constraint = true;
    p.terminal_cost_multiplier = 100.0;
    p.Qx = 10.0;   p.Qy = 10.0;   p.Qwz = 1.0;
    p.Qdotx = 1.0; p.Qdoty = 1.0; p.Qdotwz = 3.0;
    p.Rx = 1e-1;   p.Ry = 1e-1;   p.Rwz = 1e-1;
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

/*! Runs the real controller closed-loop against MuJoCo and reports the worst
 *  slack seen, so a test can distinguish "the soft constraints are idle" from
 *  "they are absorbing a violation". */
/*! Everything a soft-vs-hard comparison needs from one closed-loop run. The
 *  headline pair is availability and max_consecutive_damping: "failed at every
 *  loop" is a max-consecutive claim, and a mean alone cannot tell 2% scattered
 *  dropouts from one long outage. */
struct RunMetrics {
    double availability = 0.0;          // fraction of ticks that produced NMPC torque
    int    max_consecutive_damping = 0;
    // CAUTION: max_pos_violation is a running max over a run that BEGINS at the
    // violation, so it is pinned to the initial condition and reads ~the starting
    // offset for any controller, including a perfect one. Use initial/final/
    // recovered to ask whether the joint actually came back.
    double max_pos_violation = 0.0;     // [rad]   worst excursion past the config limits
    double pos_violation_integral = 0.0;// [rad s] distinguishes a transient from parking outside
    double initial_pos_violation = 0.0; // [rad]   at the first tick
    double final_pos_violation = 0.0;   // [rad]   at the last tick
    double recovered = 0.0;             // [rad]   initial - final; >0 means it came back
    double max_vel_violation = 0.0;     // [rad/s]
    double mean_ee_err = 0.0;           // [m]
    double max_box_slack = 0.0;
    double guard_relax_frac = 0.0;      // fraction of ticks the guard widened anything
    double guard_relax_max = 0.0;
    double mean_tick_us = 0.0;
    double max_tick_us = 0.0;
    double frac_over_budget = 0.0;      // ticks exceeding one control period
    double qp_iter_mean = 0.0;
    double qp_iter_max = 0.0;
    long   qp_retries = 0;              // solves that needed the cold-start retry
    std::map<int, int> status_hist;     // HPIPM status -> count
};

struct Scenario {
    double setpoint_offset = 0.06;      // [m] per axis, from the measured start pose
    double q_min_override  = 0.0;       // >0 raises joint_lims[0].q_min, parking it outside
    // Optional mid-run setpoint jump, reproducing what ee_target_publisher does
    // live: the arm settles, then the target moves. That transition is where the
    // warm-start iterate stops being a good guess, so a study of warm-start
    // behaviour has to contain one -- a single fixed setpoint never sees it.
    int    step_at_tick  = 0;           // 0 disables
    double step_offset   = 0.0;         // [m] per axis, added to the original target
};

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

// Thin wrapper keeping the two soft-contract tests below expressed as they were.
double maxBoxSlackOverRun(const NMPCParams& params, int ticks)
{
    return runClosedLoop(params, Scenario{}, ticks).max_box_slack;
}

TEST(NmpcSlack, IdleWhenInsideTheLimits)
{
    // Nominal tracking must not lean on the soft constraints at all: any real
    // slack here would mean the penalty, not the cost, is shaping the solution.
    EXPECT_LT(maxBoxSlackOverRun(liveParams(), 200), 1e-6);
}

TEST(NmpcSlack, AbsorbsAViolationInsteadOfFailing)
{
    // Arm starts outside its own SOFTWARE limit (joint2_L is at 0.90; q_min is
    // raised to 1.60) while staying inside the MJCF range. That is the case the
    // soft box exists for: the QP must stay feasible and take slack, rather than
    // failing and dropping through to the fallback chain.
    NMPCParams p = liveParams();
    p.joint_lims[0].q_min = 1.60;
    EXPECT_GT(maxBoxSlackOverRun(p, 200), 1e-9);
}

// --- Safe mode ------------------------------------------------------------

/*! liveParams() with the QP crippled so it cannot converge. iter_max = 1 is the
 *  load-bearing part: the forced cold-start retry inside QuadProbSolver runs with
 *  the same cap, so it fails too and solveMultiStage genuinely returns false.
 *  This is an ordinary config path, so the fallback is reachable in a test
 *  without any production seam. */
NMPCParams failingParams()
{
    NMPCParams p = liveParams();
    p.qp_max_iter = 1;
    p.qp_tol      = 1e-14;
    return p;
}

TEST(SafeMode, DampingTorqueIsDissipative)
{
    ModelProbe c(kMjcf, liveParams());
    const State s = referenceState(c.n_joints_);

    // A spread of velocities, including sign flips, so this is not passing off
    // one lucky direction.
    for (double scale : {0.1, 1.0, -1.0, 3.7}) {
        Eigen::VectorXd v = s.v * scale;
        const Eigen::VectorXd tau = c.dampingTorque(v);
        EXPECT_LE(tau.dot(v.tail(c.n_joints_)), 0.0) << "scale " << scale;
    }

    // Owned joints only: the node publishes no effort for the rest, and writing
    // them would be a silent claim on the other arm.
    Eigen::VectorXd tau = c.dampingTorque(s.v);
    for (int i = 0; i < c.n_joints_; ++i) {
        const bool owned = std::find(c.ownedJointInds.begin(), c.ownedJointInds.end(), i)
                           != c.ownedJointInds.end();
        if (!owned) EXPECT_DOUBLE_EQ(tau[i], 0.0) << "non-owned joint " << i;
    }
    EXPECT_GT(tau.cwiseAbs().maxCoeff(), 0.0) << "owned joints produced no damping";
}

TEST(SafeMode, DampingTorqueIsFiniteWithNonFiniteVelocity)
{
    // Safe mode is most often reached BECAUSE the state went non-finite, so
    // -kd*v must not hand NaN to an actuator that will latch it forever.
    ModelProbe c(kMjcf, liveParams());
    State s = referenceState(c.n_joints_);
    const int owned0 = c.ownedJointInds.front();

    for (double bad : {std::numeric_limits<double>::quiet_NaN(),
                        std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()}) {
        Eigen::VectorXd v = s.v;
        v[6 + owned0] = bad;
        const Eigen::VectorXd tau = c.dampingTorque(v);
        EXPECT_TRUE(tau.allFinite()) << "tau = " << tau.transpose();
        EXPECT_DOUBLE_EQ(tau[owned0], 0.0) << "poisoned joint should get zero";
    }
}

TEST(SafeMode, DampingRespectsTorqueLimits)
{
    ModelProbe c(kMjcf, liveParams());
    State s = referenceState(c.n_joints_);
    s.v.tail(c.n_joints_).setConstant(50.0);   // far past any rated speed

    const Eigen::VectorXd tau = c.dampingTorque(s.v);
    for (int idx : c.ownedJointInds) {
        EXPECT_LE(std::abs(tau[idx]), c.torqueLimsByIndex[idx] + 1e-12)
            << "joint " << idx;
    }
}

TEST(SafeMode, DampingIsNonSaturatingWithinRatedSpeed)
{
    // kd = frac * tau_max / qd_max, so at exactly qd_max the command should sit
    // at frac*tau_max — i.e. safe mode never needs the output clamp to stay
    // physical. That is the property that makes the gain derivable at all.
    NMPCParams p = liveParams();
    ModelProbe c(kMjcf, p);
    State s = referenceState(c.n_joints_);
    for (int idx : c.ownedJointInds) s.v[6 + idx] = c.vlimsByIndex[idx];

    const Eigen::VectorXd tau = c.dampingTorque(s.v);
    for (int idx : c.ownedJointInds) {
        EXPECT_NEAR(std::abs(tau[idx]),
                    p.safe_mode_damping_frac * c.torqueLimsByIndex[idx], 1e-9)
            << "joint " << idx;
    }
}

TEST(SafeMode, DampingIsStableInDiscreteTime)
{
    // Explicit velocity damping diverges if kd*dt exceeds the joint's inertia.
    // Nothing at runtime checks this, so pin it: it is the one way a raised
    // damping_frac could make safe mode actively unsafe.
    const double dt = 1.0 / 200.0;          // control_hz in nmpc.yaml
    ModelProbe c(kMjcf, liveParams());
    const State s = referenceState(c.n_joints_);
    const NodeDynamics nd = c.dynamicsAt(s.q, s.v, c.ee_site_id_);

    for (int idx : c.ownedJointInds) {
        EXPECT_LT(c.kd_[idx] * dt, nd.H_g(idx, idx))
            << "joint " << idx << ": kd=" << c.kd_[idx]
            << " dt=" << dt << " H_g=" << nd.H_g(idx, idx)
            << " — raise control_hz or lower safe_mode_damping_frac";
    }
}

TEST(SafeMode, QpFailureEntersDampingAndCommandsIt)
{
    ModelProbe c(kMjcf, failingParams());
    const State s = referenceState(c.n_joints_);

    const Eigen::VectorXd tau = c.computeControl(s.q, s.v);
    EXPECT_EQ(c.mode(), ArmNMPC::ControlMode::Damping);
    EXPECT_TRUE(tau.isApprox(c.dampingTorque(s.v)))
        << "tau = " << tau.transpose();
    EXPECT_GT(c.safeModeTicks(), 0);
}

TEST(SafeMode, StaysDampingUntilRecoveryTicks)
{
    // The mode must not clear on the first good solve, or a marginal solver
    // would chatter the output between damping and full tracking.
    NMPCParams p = liveParams();
    p.safe_mode_recovery_ticks = 5;
    ModelProbe c(kMjcf, p);
    const State s = referenceState(c.n_joints_);

    c.enterDamping();
    ASSERT_EQ(c.mode(), ArmNMPC::ControlMode::Damping);

    for (int t = 1; t < p.safe_mode_recovery_ticks; ++t) {
        c.computeControl(s.q, s.v);
        EXPECT_EQ(c.mode(), ArmNMPC::ControlMode::Damping)
            << "left safe mode after only " << t << " good solves";
    }
    c.computeControl(s.q, s.v);
    EXPECT_EQ(c.mode(), ArmNMPC::ControlMode::Nominal)
        << "never recovered after " << p.safe_mode_recovery_ticks << " good solves";
}

TEST(SafeMode, NonFiniteStateEntersDampingWithoutThrowing)
{
    // checkNaN used to throw here and take the process with it.
    ModelProbe c(kMjcf, liveParams());
    State s = referenceState(c.n_joints_);
    s.v[6 + c.ownedJointInds.front()] = std::numeric_limits<double>::quiet_NaN();

    Eigen::VectorXd tau;
    EXPECT_NO_THROW(tau = c.computeControl(s.q, s.v));
    EXPECT_EQ(c.mode(), ArmNMPC::ControlMode::Damping);
    EXPECT_TRUE(tau.allFinite()) << "tau = " << tau.transpose();
}

TEST(SafeMode, DampingBringsTheArmToRest)
{
    // End to end: with the QP unable to solve, closed-loop damping must actually
    // dissipate the joint velocity rather than merely being signed correctly.
    const double dt = 1.0 / 200.0;
    ModelProbe c(kMjcf, failingParams());
    State s = referenceState(c.n_joints_);

    auto ownedSpeed = [&c](const Eigen::VectorXd& v) {
        double sq = 0.0;
        for (int idx : c.ownedJointInds) sq += v[6 + idx] * v[6 + idx];
        return std::sqrt(sq);
    };

    const double v0 = ownedSpeed(s.v);
    double v_mid = 0.0;
    for (int t = 0; t < 1200; ++t) {          // 6 s at 200 Hz
        const Eigen::VectorXd tau_all = c.computeControl(s.q, s.v);
        Eigen::VectorXd tau = Eigen::VectorXd::Zero(c.n_joints_);
        for (int idx : c.ownedJointInds) tau[idx] = tau_all[idx];
        Eigen::VectorXd q1, v1;
        c.integrateStep(s.q, s.v, tau, dt, q1, v1);
        s.q = q1;
        s.v = v1;
        if (t == 599) v_mid = ownedSpeed(s.v);
    }
    const double v_end = ownedSpeed(s.v);

    // Still decaying at the end rather than settling on a floor: the slowest
    // joint's time constant is H_ii/kd_i, so an absolute target would encode
    // this arm's inertia rather than the property being tested.
    EXPECT_LT(v_mid, 0.25 * v0) << "v0=" << v0 << " v_mid=" << v_mid;
    EXPECT_LT(v_end, 0.5 * v_mid)
        << "decay stalled: v_mid=" << v_mid << " v_end=" << v_end;
}

// --- Tiered failure response ---------------------------------------------- //

/*! Stage mapping: the plan's stages are Ts apart, the ticks control_period apart,
 *  so a stage is held for the ticks it spans. Getting this wrong runs through the
 *  plan 2.5x too fast at Ts=25 ms and a 10 ms period. */
TEST(Tiers, PlanStageHoldsForTheTicksAStageSpans)
{
    // Pins its own Ts and period rather than inheriting liveParams(): this is a
    // test of the stage-hold arithmetic, so it must not change meaning when the
    // deployed horizon is retuned.
    NMPCParams p = liveParams();
    p.Ts = 0.025;
    p.control_period = 0.010;             // 2.5 ticks per stage
    ModelProbe c(kMjcf, p);
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    c.computeControl(s.q, s.v);           // produces an accepted plan

    // age 1,2 -> stage 0 (0 and 10 ms into a 25 ms stage); age 3,4,5 -> stage 1; ...
    EXPECT_EQ(c.planStageForAge(1), 0);
    EXPECT_EQ(c.planStageForAge(2), 0);
    EXPECT_EQ(c.planStageForAge(3), 0);
    EXPECT_EQ(c.planStageForAge(4), 1);
    EXPECT_EQ(c.planStageForAge(6), 2);
    EXPECT_EQ(c.planStageForAge(0), -1) << "age 0 is the accepting tick itself";
    // Past the horizon the plan is exhausted and Tier 1 must give up.
    EXPECT_EQ(c.planStageForAge(1 + static_cast<int>(2.5 * p.N) + 5), -1);
}

/*! A single failed solve should replay the plan, not abandon the task. Before the
 *  tiers, one failure cost safe_mode_recovery_ticks of damping. */
TEST(Tiers, OneFailedSolveReplaysThePlanInsteadOfDamping)
{
    NMPCParams p = liveParams();
    p.control_period = 0.010;
    ModelProbe c(kMjcf, p);
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    const Eigen::VectorXd good = c.computeControl(s.q, s.v);
    ASSERT_EQ(c.mode(), ArmNMPC::ControlMode::Nominal);

    // Force exactly one failed solve by starving the solver, then restore it.
    c.qp_->setIterMax(1);
    const Eigen::VectorXd replayed = c.computeControl(s.q, s.v);
    EXPECT_EQ(c.mode(), ArmNMPC::ControlMode::PlanReplay);
    EXPECT_EQ(c.tier1Entries(), 1);
    EXPECT_TRUE(replayed.allFinite());
    // A replay is a task-space command, so it should look nothing like damping,
    // which at near-zero velocity is near-zero torque.
    EXPECT_GT(replayed.norm(), 1e-6);

    c.qp_->setIterMax(p.qp_max_iter);
    c.computeControl(s.q, s.v);
    EXPECT_EQ(c.mode(), ArmNMPC::ControlMode::Nominal)
        << "one good solve should end a Tier 1 replay -- no hysteresis";
}

/*! The replay budget has to be finite: a plan is only valid near the state it was
 *  computed at, so a solver that stays broken must reach damping. */
TEST(Tiers, ReplayBudgetExpiresIntoDamping)
{
    NMPCParams p = liveParams();
    p.control_period = 0.010;
    p.tier1_max_replay_ticks = 3;
    ModelProbe c(kMjcf, p);
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    c.computeControl(s.q, s.v);

    c.qp_->setIterMax(1);
    for (int i = 0; i < 3; ++i) {
        c.computeControl(s.q, s.v);
        EXPECT_EQ(c.mode(), ArmNMPC::ControlMode::PlanReplay) << "replay tick " << i;
    }
    c.computeControl(s.q, s.v);
    EXPECT_EQ(c.mode(), ArmNMPC::ControlMode::Damping) << "budget spent, must damp";
    EXPECT_EQ(c.tier1Ticks(), 3);
}

/*! Tier 1 off must reproduce the old behaviour exactly, so the tiers can be
 *  switched out if they ever misbehave in flight. */
TEST(Tiers, DisablingTier1FallsStraightToDamping)
{
    NMPCParams p = liveParams();
    p.tier1_max_replay_ticks = 0;
    ModelProbe c(kMjcf, p);
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    c.computeControl(s.q, s.v);

    c.qp_->setIterMax(1);
    c.computeControl(s.q, s.v);
    EXPECT_EQ(c.mode(), ArmNMPC::ControlMode::Damping);
    EXPECT_EQ(c.tier1Ticks(), 0);
}

/*! Tier 3 stops the controller flapping between damping and tracking forever, and
 *  must not clear itself. */
TEST(Tiers, LatchesAfterSustainedDampingAndNeedsAnExplicitClear)
{
    NMPCParams p = liveParams();
    p.tier1_max_replay_ticks  = 0;     // go straight to damping
    p.tier3_latch_after_ticks = 4;
    ModelProbe c(kMjcf, p);
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    c.computeControl(s.q, s.v);

    c.qp_->setIterMax(1);
    for (int i = 0; i < 4; ++i) c.computeControl(s.q, s.v);
    ASSERT_TRUE(c.latched());

    // Healthy solves must NOT clear it.
    c.qp_->setIterMax(p.qp_max_iter);
    for (int i = 0; i < 3 * p.safe_mode_recovery_ticks; ++i) c.computeControl(s.q, s.v);
    EXPECT_TRUE(c.latched()) << "a latch that times out is not a latch";

    c.clearLatch();
    EXPECT_FALSE(c.latched());
    for (int i = 0; i < p.safe_mode_recovery_ticks + 1; ++i) c.computeControl(s.q, s.v);
    EXPECT_EQ(c.mode(), ArmNMPC::ControlMode::Nominal)
        << "after clearing, normal Tier 2 recovery should apply";
}

/*! Tier 1 reuses only the stale task WRENCH; the Jacobian and the Cv feedforward
 *  are re-derived at the current measurement. So the same plan stage at a different
 *  state must produce a different joint torque. */
TEST(Tiers, ReplayUsesCurrentKinematicsNotAStaleTorque)
{
    NMPCParams p = liveParams();
    p.control_period = 0.010;
    ModelProbe c(kMjcf, p);
    State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    c.computeControl(s.q, s.v);

    c.qp_->setIterMax(1);
    const Eigen::VectorXd at_rest = c.computeControl(s.q, s.v);

    // Same plan, same stage age, but the arm is now moving: Cv changes, so the
    // commanded torque must too.
    for (int idx : c.ownedJointInds) s.v[6 + idx] = 0.4;
    ModelProbe c2(kMjcf, p);
    c2.setDesiredPos(0.30, 0.25, 0.0);
    c2.computeControl(s.q, Eigen::VectorXd::Zero(s.v.size()));
    c2.qp_->setIterMax(1);
    const Eigen::VectorXd moving = c2.computeControl(s.q, s.v);

    ASSERT_EQ(c2.mode(), ArmNMPC::ControlMode::PlanReplay);
    EXPECT_GT((moving - at_rest).norm(), 1e-9)
        << "replay looks like it is reusing a torque rather than a wrench";
}

TEST(SafeMode, RejectsInvalidParams)
{
    {   // sqp_iters = 0 skipped the rollout loop and left J_task0 at 0x0
        NMPCParams p = liveParams();
        p.sqp_iters = 0;
        EXPECT_THROW(ModelProbe(kMjcf, p), std::runtime_error);
    }
    {
        NMPCParams p = liveParams();
        p.safe_mode_damping_frac = 0.0;
        EXPECT_THROW(ModelProbe(kMjcf, p), std::runtime_error);
    }
    {
        NMPCParams p = liveParams();
        p.safe_mode_recovery_ticks = 0;
        EXPECT_THROW(ModelProbe(kMjcf, p), std::runtime_error);
    }
}

// --- Hard vs soft constraints ---------------------------------------------

TEST(HardConstraints, RolloutClampDoesNotSatisfyTheBoxForFree)
{
    // The rollout clamps its predicted joint positions to keep the open-loop
    // trajectory out of a kinematic singularity. It must clamp to the MECHANICAL
    // range, not to the configured software margin: the box is written on
    // Δq = q_k − q0, so clamping into the margin would put the nominal trajectory
    // inside the box by construction, and the multiple-shooting defect b_k would
    // hand the QP that teleport as free dynamics. The constraint could then never
    // bind, soft or hard.
    //
    // joint2_L sits at 0.90 with q_min raised to 1.60, so the box is [0.70, 2.24]
    // and Δq is pinned to 0 at stage 0. Bounded torque moves Δq by ~0.017 in one
    // 25 ms step, so the plan must stay near that, NOT jump to the box edge.
    NMPCParams p = liveParams();
    p.joint_lims[0].q_min = 1.60;
    p.soft_joint_limits   = true;      // soft, so this stays feasible and observable
    ModelProbe c(kMjcf, p);
    State s = referenceState(c.n_joints_);
    const ArmNMPC::EePose st = c.currentEePose(s.q, s.v);
    c.setDesiredPos(st.pos.x() + 0.06, st.pos.y() + 0.06, st.pos.z());
    c.setDesiredOrient(st.quat(0), st.quat(1), st.quat(2), st.quat(3));

    c.computeControl(s.q, s.v);
    ASSERT_EQ(c.mode(), ArmNMPC::ControlMode::Nominal);

    const int idx   = c.ownedJointInds[0];
    const double lo = c.qlimsByIndex[idx].first - s.q[7 + idx];   // 0.70
    Eigen::VectorXd x1(7 + 2 * c.n_joints_);
    c.qp_->getX(1, x1.data());

    EXPECT_LT(x1[7 + idx], 0.25 * lo)
        << "stage 1 planned Δq = " << x1[7 + idx] << ", suspiciously close to the "
        << "box edge " << lo << " that bounded torque cannot reach in one step";
}

TEST(HardConstraints, SoftBoxCarriesRealSlackWhenTheArmIsOutsideItsLimit)
{
    // The other half: with the box genuinely unreachable, a SOFT box must show
    // slack of the order of the violation. Slack near zero here would mean the
    // constraint is being satisfied by something other than the control.
    NMPCParams p = liveParams();
    p.soft_joint_limits = true;
    const RunMetrics m = runClosedLoop(p, Scenario{0.06, 1.60}, 200);

    EXPECT_GT(m.max_box_slack, 1e-3)
        << "box slack " << m.max_box_slack << " is too small for an arm parked "
        << "0.7 rad outside its limit";
    // Straining the box this hard still costs time, but with HARD torque rows it
    // no longer costs much availability: ~96.7% on both x86-64 and aarch64, with
    // the joint fully recovered. This assertion is the one that caught soft torque
    // rows collapsing on aarch64 (21.7%, joint never recovered) while passing on
    // x86-64 -- keep it, and keep it on availability, because that is the column
    // that moved.
    EXPECT_GT(m.availability, 0.5)
        << "availability " << m.availability << ", longest damping run "
        << m.max_consecutive_damping;
}

TEST(HardConstraints, RawHardBoxFailsWhenTheArmStartsOutsideItsLimit)
{
    // The historical failure, now honest and reproducible: the box cannot contain
    // the measured state, one step cannot reach it, and with no guard the QP has
    // no answer at all. Encoded deliberately so it regresses loudly if the
    // rollout clamp ever starts papering over it again.
    NMPCParams p = liveParams();
    p.soft_joint_limits       = false;
    p.relax_box_to_contain_x0 = false;
    const RunMetrics m = runClosedLoop(p, Scenario{0.06, 1.60}, 200);

    EXPECT_LT(m.availability, 0.5)
        << "expected mostly safe mode, got availability " << m.availability;
    EXPECT_GT(m.max_consecutive_damping, 50)
        << "expected a sustained outage, longest run " << m.max_consecutive_damping;
}

TEST(HardConstraints, GuardKeepsAHardBoxFeasibleFromOutsideTheLimit)
{
    // Same scenario, guard on: the box is widened just enough to contain the
    // measurement, so the solve stays feasible throughout.
    NMPCParams p = liveParams();
    p.soft_joint_limits       = false;
    p.relax_box_to_contain_x0 = true;
    const RunMetrics m = runClosedLoop(p, Scenario{0.06, 1.60}, 200);

    EXPECT_GT(m.availability, 0.99)
        << "availability " << m.availability
        << ", longest damping run " << m.max_consecutive_damping;
    EXPECT_GT(m.guard_relax_frac, 0.0) << "the guard should have been needed here";
}

TEST(HardConstraints, GuardMarginIsFlooredAtOneStepOfDrift)
{
    // A joint at rated speed moves Ts*qd_max in one step whatever the control
    // does, so a guard margin below that re-creates the infeasibility the guard
    // exists to remove. ArmNMPC raises the configured value to that floor — and
    // the raise was silently overwritten once, which nothing could observe.
    NMPCParams p = liveParams();
    p.relax_box_to_contain_x0 = true;
    p.relax_box_margin        = 1e-9;      // far below the floor
    ModelProbe c(kMjcf, p);

    double qd_max_owned = 0.0;
    for (int idx : c.ownedJointInds)
        qd_max_owned = std::max(qd_max_owned, c.vlimsByIndex[idx]);
    const double floor = p.Ts * qd_max_owned;

    ASSERT_GT(floor, p.relax_box_margin) << "test is vacuous if the floor is lower";
    EXPECT_NEAR(c.qp_->relaxMargin(), floor, 1e-12)
        << "configured " << p.relax_box_margin << " should have been raised to "
        << floor;
}

TEST(HardConstraints, GuardMarginHonoursALargerConfiguredValue)
{
    // The floor raises, never lowers: a deliberately generous margin must survive.
    NMPCParams p = liveParams();
    p.relax_box_to_contain_x0 = true;
    p.relax_box_margin        = 0.5;
    ModelProbe c(kMjcf, p);
    EXPECT_NEAR(c.qp_->relaxMargin(), 0.5, 1e-12);
}

TEST(HardConstraints, GuardIsNotNeededWhenTheArmIsInsideItsLimits)
{
    // The guard must not be silently load-bearing in normal operation: if it
    // fired every tick here, a "hard" box would really be no box at all and any
    // comparison against soft would be measuring nothing.
    NMPCParams p = liveParams();
    p.soft_joint_limits       = false;
    p.relax_box_to_contain_x0 = true;
    const RunMetrics m = runClosedLoop(p, Scenario{}, 200);

    EXPECT_LT(m.guard_relax_frac, 0.01)
        << "guard fired on " << (100.0 * m.guard_relax_frac) << "% of nominal ticks";
    EXPECT_GT(m.availability, 0.99);
}

TEST(HardConstraints, TorqueRowHardnessDoesNotChangeTheNominalCommand)
{
    // Hardening the torque rows changes the QP's PLAN, not the applied torque:
    // the controller already clamps per joint before publishing. Nominal
    // behaviour is therefore essentially unchanged either way, which is what
    // makes the hard default free -- it buys robustness near infeasibility
    // without costing anything here.
    //
    // `hard` is the live configuration and `soft` the variant, the opposite way
    // round from when soft torque was the default. Set explicitly rather than
    // taken from liveParams(), so this keeps comparing two different things if
    // the default moves again.
    NMPCParams hard = liveParams();
    NMPCParams soft = liveParams();
    hard.soft_torque_limits = false;
    soft.soft_torque_limits = true;

    const RunMetrics mh = runClosedLoop(hard, Scenario{}, 100);
    const RunMetrics ms = runClosedLoop(soft, Scenario{}, 100);

    EXPECT_GT(mh.availability, 0.99) << "hard torque rows cost availability";
    EXPECT_NEAR(mh.mean_ee_err, ms.mean_ee_err, 0.25 * ms.mean_ee_err + 1e-4)
        << "hard " << mh.mean_ee_err << " vs soft " << ms.mean_ee_err;
}

// --- Realtime profile ----------------------------------------------------- //

/*! The per-tick heap traffic of the live configuration, as a bound rather than
 *  an exact figure: it depends on the Eigen and acados builds, so pinning it
 *  exactly would flake. The bound catches a reintroduced allocation on the hot
 *  path, which is what matters under a realtime policy. */
TEST(RealtimeProfile, ControlTickAllocationCountIsBounded)
{
    constexpr long kTicks       = 10;
    // Measured 1 (was 1369), and that one is computeControl's returned vector:
    // the signature returns by value, so the caller's result is necessarily a
    // heap allocation. Nothing else on the tick allocates -- acados, HPIPM,
    // BLASFEO and MuJoCo contribute none of it, confirmed by resolving every
    // call site to its originating object (DISABLED_AllocStudy.Attribution).
    //
    // The ceiling allows a little slack for a different Eigen or acados build
    // rather than pinning 1 exactly, while still failing if a single per-node
    // allocation comes back -- those arrive 30 at a time.
    constexpr long kMaxPerTick  = 8;

    ModelProbe c(kMjcf, liveParams());
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    c.setDesiredOrient(1.0, 0.0, 0.0, 0.0);

    // Warm up first: the first tick also pays one-off costs (lazily sized
    // workspaces, acados' first solve) that are not per-tick traffic.
    for (int i = 0; i < 3; ++i) c.computeControl(s.q, s.v);

    long allocs = 0;
    {
        AllocScope scope;
        for (long i = 0; i < kTicks; ++i) c.computeControl(s.q, s.v);
        allocs = scope.n();
    }

    const double per_tick = static_cast<double>(allocs) / kTicks;
    std::printf("[alloc] %.0f heap allocations per control tick\n", per_tick);
    EXPECT_GT(allocs, 0) << "counter did not observe any allocation -- the "
                            "malloc interposer is not being reached";
    EXPECT_LT(per_tick, kMaxPerTick);
}

/*! The profiler has to report a worst case, not just a mean, and has to forget
 *  it on reset -- a cumulative max would make every later window report the
 *  first spike it ever saw. */
TEST(RealtimeProfile, ReportCarriesAMaxAndAHistogramAndResets)
{
    ModelProbe c(kMjcf, liveParams());
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    // A budget no tick can plausibly exceed, so the miss count is a fixed
    // expectation rather than a measurement of the machine running the test.
    c.setTickBudgetUs(60e6);
    for (int i = 0; i < 5; ++i) c.computeControl(s.q, s.v);

    const std::string rep = c.profileReport();
    EXPECT_NE(rep.find("over 5 ticks"), std::string::npos) << rep;
    EXPECT_NE(rep.find("max="), std::string::npos) << rep;
    EXPECT_NE(rep.find("hist(tick)"), std::string::npos) << rep;
    EXPECT_NE(rep.find("over_period=0/5"), std::string::npos) << rep;
    EXPECT_NE(rep.find("qpsolve("), std::string::npos) << rep;
    EXPECT_NE(rep.find("unattributed="), std::string::npos) << rep;

    EXPECT_GT(c.tick.max_us, 0.0);
    c.resetProfile();
    EXPECT_EQ(c.tick.count, 0);
    EXPECT_EQ(c.tick.max_us, 0.0);
    EXPECT_EQ(c.qpsolve.count, 0);
    EXPECT_EQ(c.tick.budget_us, 60e6) << "reset dropped the configured deadline";
}

/*! A tick over the budget must be counted exactly. The histogram cannot answer
 *  this: its bins are coarser than any period of interest and the last one is
 *  open-ended, so a budget above 5 ms has no bin boundary to read. */
TEST(RealtimeProfile, DeadlineMissesAreCountedAgainstTheBudget)
{
    ModelProbe c(kMjcf, liveParams());
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    c.setTickBudgetUs(1e-3);       // below any achievable tick: all ticks miss
    for (int i = 0; i < 4; ++i) c.computeControl(s.q, s.v);

    EXPECT_EQ(c.tick.over, 4);
    EXPECT_NE(c.profileReport().find("over_period=4/4"), std::string::npos);

    c.setTickBudgetUs(0.0);        // not a deadline: nothing counted, no column
    c.resetProfile();
    c.computeControl(s.q, s.v);
    EXPECT_EQ(c.tick.over, 0);
    EXPECT_EQ(c.profileReport().find("over_period"), std::string::npos);
}

/*! lastQpIters() sums acados' num_iter over every solve in the tick, which is only
 *  right if acados reports it PER SOLVE rather than cumulatively. If it accumulated
 *  internally the retry would be double counted and every reported figure for a
 *  retried tick would be inflated. Pin it: one capped solve versus the same capped
 *  solve plus its cold retry should differ by exactly one cap's worth. */
TEST(RealtimeProfile, QpIterationsArePerSolveNotCumulative)
{
    constexpr int kCap = 4;        // below the ~8 a healthy solve needs, so it fails

    NMPCParams cold = liveParams();
    cold.qp_warm_start = 0;        // no retry path at all -> exactly one solve
    ModelProbe c0(kMjcf, cold);
    const State s = referenceState(c0.n_joints_);
    c0.setDesiredPos(0.30, 0.25, 0.0);
    c0.qp_->setIterMax(kCap);
    c0.computeControl(s.q, s.v);
    const int one_solve = c0.qp_->lastQpIters();

    NMPCParams warm = liveParams();
    warm.qp_warm_start = 1;        // fails, then retries cold -> two solves
    ModelProbe c1(kMjcf, warm);
    c1.setDesiredPos(0.30, 0.25, 0.0);
    c1.qp_->setIterMax(kCap);
    c1.computeControl(s.q, s.v);
    const int two_solves = c1.qp_->lastQpIters();

    ASSERT_TRUE(c1.qp_->lastRetried()) << "the retry did not fire, so this proves nothing";
    EXPECT_EQ(one_solve, kCap) << "a capped failing solve should report the cap";
    EXPECT_EQ(two_solves, 2 * kCap)
        << "one solve reported " << one_solve << ", two reported " << two_solves
        << " -- not 2x, so acados' num_iter is cumulative and the sum double counts";
}

/*! The iteration count is the link between a slow tick and its cause, so it has
 *  to come back from the solver rather than stay at the "no info" sentinel. */
TEST(RealtimeProfile, QpIterationsAreReported)
{
    ModelProbe c(kMjcf, liveParams());
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    for (int i = 0; i < 5; ++i) c.computeControl(s.q, s.v);

    EXPECT_EQ(c.qp_iters.count, 5) << "one tally per tick";
    EXPECT_GT(c.qp_iters.mean(), 0.0);
    EXPECT_EQ(c.qp_retries, 0) << "the nominal path should not need a cold retry";
    // Without a retry the tick holds exactly one solve, so the cap applies to it.
    EXPECT_LE(c.qp_iters.max, liveParams().qp_max_iter);
    // HPIPM's own time cannot exceed the stopwatch around the call that produced it.
    EXPECT_GT(c.qp_time_ms.max, 0.0);
    EXPECT_LE(c.qp_time_ms.max * 1e3, c.qpsolve.max_us);
    EXPECT_NE(c.profileReport().find("qp_iter/tick mean="), std::string::npos);

    c.resetProfile();
    EXPECT_EQ(c.qp_iters.count, 0);
    EXPECT_EQ(c.qp_time_ms.count, 0);
    EXPECT_EQ(c.profileReport().find("qp_iter"), std::string::npos)
        << "an empty tally should not print a column";
}

/*! The hot path must reach a steady state where a tick allocates nothing new.
 *
 *  Separate from the ceiling test above because the two ask different things: that
 *  one bounds the average, this one asserts the FIRST tick is where sizing happens
 *  and later ticks do not keep growing. A buffer that is re-sized every tick, or a
 *  container that is cleared and refilled, passes a loose average and fails here. */
TEST(RealtimeProfile, SteadyStateTicksDoNotKeepAllocating)
{
    ModelProbe c(kMjcf, liveParams());
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);

    long first = 0, later = 0;
    {
        AllocScope scope;
        c.computeControl(s.q, s.v);
        first = scope.n();
    }
    // Several more, so a per-tick leak shows up multiplied rather than once.
    constexpr long kLater = 20;
    {
        AllocScope scope;
        for (long i = 0; i < kLater; ++i) c.computeControl(s.q, s.v);
        later = scope.n();
    }
    const double per_later = static_cast<double>(later) / kLater;
    std::printf("[alloc] first tick %ld, steady state %.1f per tick\n", first, per_later);

    EXPECT_LT(per_later, static_cast<double>(first))
        << "the first tick should be the expensive one -- it does the sizing";
    // What remains is computeControl's returned vector, one per tick. See
    // ControlTickAllocationCountIsBounded for why that one is the floor here.
    EXPECT_LT(per_later, 8.0);
}

/*! The per-tick trace exists because the report's eight histogram bins cannot
 *  show the shape of the distribution. It has to be exact per tick, bounded, and
 *  free of allocation on the control path. */
TEST(RealtimeProfile, TraceRecordsEveryTickWithoutAllocating)
{
    constexpr std::size_t kCap = 8;
    ModelProbe c(kMjcf, liveParams());
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    c.computeControl(s.q, s.v);            // warm up before the buffer is armed

    c.enableTrace(kCap);
    long allocs = 0;
    {
        AllocScope scope;
        for (std::size_t i = 0; i < kCap; ++i) c.computeControl(s.q, s.v);
        allocs = scope.n();
    }
    ASSERT_EQ(c.trace().size(), kCap);

    // Recording must not be what makes the tick allocate. Compare against the
    // ceiling the un-traced path already holds to, rather than a fresh figure.
    EXPECT_LT(static_cast<double>(allocs) / kCap, 2500.0);

    for (const auto& r : c.trace()) {
        EXPECT_GT(r.tick_us, 0.0f);
        EXPECT_GT(r.qpsolve, 0.0f) << "the SQP path always solves";
        // Phases are this tick's share, so they cannot exceed the tick itself.
        const float summed = r.dynamics + r.jdotqdot + r.lambda + r.qpsolve
                           + r.rollout_dynamics + r.rollout_integrate + r.pinv
                           + r.build_model + r.defect + r.trust_region + r.finalize;
        EXPECT_LE(summed, r.tick_us * 1.05f + 1.0f)
            << "summed phases " << summed << " vs tick " << r.tick_us;
        EXPECT_GE(r.qp_iters, 0);
        EXPECT_EQ(r.mode, 0) << "nominal run should not be in safe mode";
    }
    // Monotonic timestamps, first row at zero.
    EXPECT_FLOAT_EQ(static_cast<float>(c.trace().front().t_us), 0.0f);
    for (std::size_t i = 1; i < c.trace().size(); ++i)
        EXPECT_GT(c.trace()[i].t_us, c.trace()[i - 1].t_us);

    // Bounded: further ticks are dropped, not appended, and never reallocate.
    c.computeControl(s.q, s.v);
    EXPECT_EQ(c.trace().size(), kCap);
    EXPECT_TRUE(c.traceFull());
}

/*! A report window resets the PhaseStat sums the trace differences against, so
 *  the first tick afterwards must not absorb the whole window's total. */
TEST(RealtimeProfile, TraceSurvivesAProfileReset)
{
    ModelProbe c(kMjcf, liveParams());
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    c.enableTrace(16);
    for (int i = 0; i < 4; ++i) c.computeControl(s.q, s.v);
    const float before = c.trace().back().qpsolve;

    c.resetProfile();                      // what the diagnostics timer does
    c.computeControl(s.q, s.v);
    const float after = c.trace().back().qpsolve;

    EXPECT_LT(after, 4.0f * before)
        << "qpsolve " << after << " us after a reset vs " << before
        << " before -- the reset baseline was not cleared";
    EXPECT_LE(after, c.trace().back().tick_us);
}

// --- MuJoCo error / warning hooks ----------------------------------------- //

/*! MuJoCo's default warning handler appends to MUJOCO_LOG.TXT -- a file write on
 *  the control path, from a code path that fires when the solver is struggling.
 *  The hook has to be ours, and it has to record rather than write. */
TEST(MujocoHooks, WarningIsCountedAndNotWrittenToAFile)
{
    ModelProbe c(kMjcf, liveParams());     // the constructor installs the hooks
    const long before = c.mjWarningCount();

    mju_warning("pat test warning %d", 42);

    EXPECT_EQ(c.mjWarningCount(), before + 1);
    EXPECT_NE(std::string(c.lastMjWarning()).find("pat test warning 42"),
              std::string::npos)
        << "got: " << c.lastMjWarning();
}

/*! The default error handler calls exit(), which would kill the process mid-motion
 *  with a torque latched in the actuators. Ours throws, so ArmNMPC's existing
 *  guard can turn it into safe-mode damping. */
TEST(MujocoHooks, ErrorThrowsInsteadOfTerminating)
{
    ModelProbe c(kMjcf, liveParams());
    const long before = c.mjErrorCount();

    EXPECT_THROW(mju_error("pat test error"), std::runtime_error);

    EXPECT_EQ(c.mjErrorCount(), before + 1);
    EXPECT_NE(std::string(c.lastMjError()).find("pat test error"),
              std::string::npos)
        << "got: " << c.lastMjError();
}

/*! Both counts have to reach the report, or redirecting MuJoCo's logging would
 *  trade noisy-but-visible for silent. */
TEST(MujocoHooks, CountsReachTheProfileReport)
{
    ModelProbe c(kMjcf, liveParams());
    mju_warning("pat report probe");
    EXPECT_NE(c.profileReport().find("MuJoCo warnings:"), std::string::npos);
    EXPECT_NE(c.profileReport().find("pat report probe"), std::string::npos);
}

/*! Every phase the report lists must actually be timed on the active path --
 *  an untimed phase shows up as unattributed time and the profile cannot then
 *  point at the cost. */
TEST(RealtimeProfile, TheSqpPathAttributesMostOfItsTick)
{
    ModelProbe c(kMjcf, liveParams());
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    for (int i = 0; i < 10; ++i) c.computeControl(s.q, s.v);

    const PhaseStat* required[] = {&c.rollout_dynamics, &c.rollout_integrate,
                                   &c.pinv, &c.build_model, &c.defect,
                                   &c.qpsolve, &c.trust_region, &c.finalize};
    double attributed = 0.0;
    for (const PhaseStat* p : required) {
        EXPECT_GT(p->count, 0) << "a phase the report lists is never timed";
        attributed += p->sum_us;
    }
    EXPECT_GT(attributed, 0.6 * c.tick.sum_us)
        << "attributed " << attributed << " of " << c.tick.sum_us << " us";
}

// --- Warm start vs cold start --------------------------------------------- //

/*! The scenario that separates the two: settle, then jump the target. A single
 *  fixed setpoint never invalidates the warm-start iterate, so it cannot show the
 *  behaviour this is about. */
Scenario warmStartScenario()
{
    Scenario sc;
    sc.setpoint_offset = 0.06;
    sc.step_at_tick    = 150;
    sc.step_offset     = 0.12;
    return sc;
}

/*! Cold-starting every solve trades mean cost for a bounded worst case.
 *
 *  Warm-starting is cheaper per tick while the previous iterate stays a good
 *  guess, but a target jump invalidates it and HPIPM then runs to the iteration
 *  cap before the cold-start retry rescues it -- a tick an order of magnitude over
 *  the period. Cold-starting pays more iterations every tick and never approaches
 *  the cap. For a deadline the second shape is the useful one, so the invariant
 *  worth holding is that cold stays far clear of the cap. */
TEST(WarmStart, ColdStartingStaysWellClearOfTheIterationCap)
{
    NMPCParams p = liveParams();
    p.qp_warm_start = 0;
    const RunMetrics m = runClosedLoop(p, warmStartScenario(), 400);

    EXPECT_EQ(m.qp_retries, 0) << "warm_start=0 disables the retry path entirely";
    EXPECT_GT(m.availability, 0.99) << "cold starts should not cost availability";
    EXPECT_LT(m.qp_iter_max, 0.5 * p.qp_max_iter)
        << "cold solves ran to " << m.qp_iter_max << " of a " << p.qp_max_iter
        << " cap; the cap is supposed to be slack, not a bound being reached";
}

/*! Warm-starting's cost is concentrated in the ticks right after a target jump,
 *  which is what makes a mean-only profile miss it. */
TEST(WarmStart, WarmStartingCostsMoreInTheWorstTickThanCold)
{
    NMPCParams warm = liveParams();   // qp_warm_start defaults to 2
    NMPCParams cold = liveParams();
    cold.qp_warm_start = 0;

    const RunMetrics mw = runClosedLoop(warm, warmStartScenario(), 400);
    const RunMetrics mc = runClosedLoop(cold, warmStartScenario(), 400);

    EXPECT_LT(mc.qp_iter_max, mw.qp_iter_max)
        << "warm " << mw.qp_iter_max << " vs cold " << mc.qp_iter_max;
    EXPECT_GT(mc.qp_iter_mean, mw.qp_iter_mean)
        << "cold is expected to pay more iterations on the average tick";
    // Both must still control the arm; a bounded tail is worthless if it tracks worse.
    EXPECT_NEAR(mc.mean_ee_err, mw.mean_ee_err, 0.25 * mw.mean_ee_err + 1e-4)
        << "warm " << mw.mean_ee_err << " vs cold " << mc.mean_ee_err;
}

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

/*! The experiment. Disabled so it never runs in CI:
 *    ./build/pat_arm_control/test_pat_arm_control \
 *        --gtest_also_run_disabled_tests --gtest_filter='DISABLED_ConstraintStudy.*'
 */
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
 *  x86-64 and fails on aarch64, with everything else in the suite green and the
 *  same commit on both. Measured, same scenario, soft box + soft torque +
 *  terminal_velocity_constraint:
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

// --- The control loop's input hand-off ------------------------------------ //

namespace {
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
}  // namespace

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

// === Disturbance feedforward =============================================== //
// Compensating an estimate closes a loop around it, so each guard between the
// estimate and the actuators gets its own test: the default must be a literal
// no-op, the compensation must be the stated fraction and nothing more, and
// the cap, the slew limit and the safe-mode reset must each bind.

namespace {
//! liveParams with the feedforward enabled, so each test states only what it
//! is actually varying.
NMPCParams ffParams(double gain, double max_frac = 0.3, double slew = 0.0)
{
    NMPCParams p = liveParams();
    p.ff_gain     = gain;
    p.ff_max_frac = max_frac;
    p.ff_slew     = slew;
    return p;
}

//! A disturbance on the owned joints only, which is what the reduction yields.
Eigen::VectorXd ffEstimate(int n_joints, double value)
{
    Eigen::VectorXd d = Eigen::VectorXd::Zero(n_joints);
    for (int i = 0; i < n_joints && i < 3; ++i) d(i) = value * (i == 1 ? -1.0 : 1.0);
    return d;
}

/*! An estimate sized as a fraction of each joint's feedforward CAP, so a test
 *  that wants to stay clear of the saturation says so instead of guessing a
 *  magnitude. The harness runs 0.25 N*m limits on every owned joint where the
 *  shipped config runs 10/10/5, so a hardcoded N*m value means something quite
 *  different in each and silently saturates in one of them. */
Eigen::VectorXd ffEstimateBelowCap(const ModelProbe& c, double frac_of_cap)
{
    Eigen::VectorXd d = Eigen::VectorXd::Zero(c.n_joints_);
    for (int i = 0; i < c.n_joints_ && i < 3; ++i) {
        const double cap = c.params_.ff_max_frac *
                           c.torqueLimsByIndex[static_cast<std::size_t>(i)];
        d(i) = frac_of_cap * cap * (i == 1 ? -1.0 : 1.0);
    }
    return d;
}
}  // namespace

TEST(Feedforward, GainOfZeroIsABitExactNoOp)
{
    // The gate that lets the observer run on a configuration whose numbers are
    // already committed: offering an estimate at gain 0 must change nothing at
    // all, not merely change it by a little.
    const State s = referenceState(6);

    ModelProbe baseline(kMjcf, liveParams());
    baseline.setDesiredPos(0.30, 0.25, 0.0);
    const Eigen::VectorXd tau_ref = baseline.computeControl(s.q, s.v);

    ModelProbe offered(kMjcf, ffParams(0.0));
    offered.setDesiredPos(0.30, 0.25, 0.0);
    offered.setDisturbanceFeedforward(ffEstimate(offered.n_joints_, 2.0), 0.010);
    const Eigen::VectorXd tau = offered.computeControl(s.q, s.v);

    ASSERT_EQ(tau.size(), tau_ref.size());
    for (int i = 0; i < tau.size(); ++i) EXPECT_EQ(tau(i), tau_ref(i)) << "joint " << i;
    EXPECT_EQ(offered.disturbanceFeedforward().cwiseAbs().maxCoeff(), 0.0);
}

TEST(Feedforward, CommandCancelsExactlyTheGainTimesTheEstimate)
{
    // Checked at finalizeJointTorques rather than through computeControl. The
    // bounds shift changes the QP's feasible set, so when the torque constraint
    // is ACTIVE the solver's u legitimately moves and the end-to-end command is
    // not the baseline minus d -- this harness runs 0.25 N*m limits, where it
    // is active. The claim being made here is about the command the controller
    // forms from a given solution, and that is exact.
    const State s = referenceState(6);

    auto commandWith = [&](double gain, const Eigen::VectorXd* d) {
        ModelProbe c(kMjcf, ffParams(gain));
        c.loadLiveState(s.q, s.v);
        Eigen::MatrixXd H_g;
        Eigen::VectorXd Cv;
        c.getDynamics(H_g, Cv);
        const Eigen::MatrixXd J = c.gjm6(c.ee_site_id_);
        // A fixed task wrench, so the only thing differing between the two
        // calls is the feedforward.
        Eigen::VectorXd tau_task = Eigen::VectorXd::Zero(J.rows());
        tau_task(0) = 0.4;
        tau_task(1) = -0.25;
        if (d) c.setDisturbanceFeedforward(*d, 0.010);
        Eigen::VectorXd out;
        c.finalizeJointTorques(J, tau_task, Cv, out);
        return out;
    };

    ModelProbe probe(kMjcf, ffParams(1.0));
    const Eigen::VectorXd d = ffEstimateBelowCap(probe, 0.5);

    const Eigen::VectorXd tau_ref = commandWith(1.0, nullptr);
    const Eigen::VectorXd tau_ff  = commandWith(1.0, &d);
    const Eigen::VectorXd tau_half = [&] {
        const Eigen::VectorXd dh = ffEstimateBelowCap(probe, 0.5);
        return commandWith(0.5, &dh);
    }();

    for (int i = 0; i < tau_ref.size(); ++i) {
        // Only where the per-joint clamp is not what decided the answer.
        const double lim = probe.torqueLimsByIndex[static_cast<std::size_t>(i)];
        if (std::fabs(tau_ref(i)) >= lim - 1e-9) continue;
        EXPECT_NEAR(tau_ff(i), tau_ref(i) - d(i), 1e-9)
            << "joint " << i << ": full gain did not cancel the estimate";
        EXPECT_NEAR(tau_half(i), tau_ref(i) - 0.5 * d(i), 1e-9)
            << "joint " << i << ": half gain did not cancel half of it";
    }
}

TEST(Feedforward, TorqueBoundsShiftWithTheCommand)
{
    // The other half of the change, and the reason a torque limit still means
    // what it says: the QP's general constraint bounds the APPLIED torque, so
    // when the command gains a -d_ff term both bounds must move by +d_ff. If
    // they did not, the solver would plan against a budget the command does not
    // have and the clamp in finalizeJointTorques would quietly take it back.
    const State s = referenceState(6);

    auto boundsWith = [&](const Eigen::VectorXd* d) {
        ModelProbe c(kMjcf, ffParams(1.0));
        c.loadLiveState(s.q, s.v);
        Eigen::MatrixXd H_g;
        Eigen::VectorXd Cv;
        c.getDynamics(H_g, Cv);
        const Eigen::MatrixXd J = c.gjm6(c.ee_site_id_);
        if (d) c.setDisturbanceFeedforward(*d, 0.010);
        Eigen::MatrixXd D;
        Eigen::VectorXd lg, ug;
        c.buildTorqueGeneralConstraint(J, Cv, D, lg, ug);
        return std::make_pair(lg, ug);
    };

    ModelProbe probe(kMjcf, ffParams(1.0));
    const Eigen::VectorXd d = ffEstimateBelowCap(probe, 0.5);

    const auto ref = boundsWith(nullptr);
    const auto shifted = boundsWith(&d);

    for (int i = 0; i < d.size(); ++i) {
        EXPECT_NEAR(shifted.first(i), ref.first(i) + d(i), 1e-12)
            << "lower bound, joint " << i;
        EXPECT_NEAR(shifted.second(i), ref.second(i) + d(i), 1e-12)
            << "upper bound, joint " << i;
    }
    // The width is untouched: the feedforward moves the window, it does not
    // widen it, so the arm keeps exactly the authority it had.
    for (int i = 0; i < d.size(); ++i) {
        EXPECT_NEAR(shifted.second(i) - shifted.first(i),
                    ref.second(i) - ref.first(i), 1e-12) << "joint " << i;
    }
}

TEST(Feedforward, SaturationBoundsAWildEstimate)
{
    // The estimate is unbounded by construction -- a momentum residual during
    // hard contact is whatever the contact was -- and it goes to the actuators.
    ModelProbe ff(kMjcf, ffParams(1.0, 0.25));
    const Eigen::VectorXd d = Eigen::VectorXd::Constant(ff.n_joints_, 1000.0);
    ff.setDisturbanceFeedforward(d, 0.010);

    const Eigen::VectorXd& applied = ff.disturbanceFeedforward();
    for (int i = 0; i < ff.n_joints_; ++i) {
        const double cap = 0.25 * ff.torqueLimsByIndex[static_cast<std::size_t>(i)];
        EXPECT_LE(std::fabs(applied(i)), cap + 1e-12)
            << "joint " << i << " exceeded its cap";
    }
    EXPECT_GT(ff.ffSaturated(), 0) << "the saturation did not report itself";
    // The cap is derived per joint rather than being one global number. It
    // cannot be checked by comparing two joints here -- this harness gives
    // every owned joint the same 0.25 N*m limit, where the shipped config uses
    // 10/10/5 -- so check it against the limit each joint actually has.
    for (int i = 0; i < ff.n_joints_; ++i) {
        const double lim = ff.torqueLimsByIndex[static_cast<std::size_t>(i)];
        if (lim <= 0.0) continue;        // a joint this controller does not own
        EXPECT_NEAR(std::fabs(applied(i)), 0.25 * lim, 1e-12) << "joint " << i;
    }
}

TEST(Feedforward, SlewLimitBoundsHowFastItMoves)
{
    // Contact makes the residual step within a tick or two; handing that step
    // to a motor is a torque discontinuity.
    constexpr double kDt = 0.010;
    constexpr double kSlew = 1.0;                  // N*m/s
    const double kStep = kSlew * kDt;              // 0.01 N*m per tick

    ModelProbe ff(kMjcf, ffParams(1.0, 1.0, kSlew));
    // A target inside the cap, so the slew is what bounds the approach and not
    // the saturation. ff_max_frac is 1.0 here, so the cap is the torque limit.
    const Eigen::VectorXd d = ffEstimateBelowCap(ff, 0.8);
    const double target = d(0);
    ASSERT_GT(target, 12 * kStep) << "the target must be several ticks away";

    double prev = 0.0;
    for (int tick = 0; tick < 10; ++tick) {
        ff.setDisturbanceFeedforward(d, kDt);
        const double now = ff.disturbanceFeedforward()(0);
        EXPECT_LE(now - prev, kStep + 1e-12)
            << "tick " << tick << " moved " << (now - prev) << " N*m";
        prev = now;
    }
    EXPECT_GT(ff.ffSlewLimited(), 0);
    // Ten ticks of 0.01 cannot have arrived -- that is the point.
    EXPECT_LT(prev, target);
    EXPECT_NEAR(prev, 10 * kStep, 1e-9);

    // Given enough ticks it does arrive, and stops there rather than
    // overshooting past the value it was asked for.
    for (int tick = 0; tick < 500; ++tick) ff.setDisturbanceFeedforward(d, kDt);
    EXPECT_NEAR(ff.disturbanceFeedforward()(0), target, 1e-9);
}

TEST(Feedforward, SafeModeEntryClearsIt)
{
    // Safe mode means something already went wrong, and a bad state is exactly
    // what produces a bad estimate. What the reset really protects is recovery:
    // without it the first tracking tick afterwards applies the stale value as
    // a step.
    ModelProbe ff(kMjcf, ffParams(1.0));
    ff.setDisturbanceFeedforward(ffEstimateBelowCap(ff, 0.8), 0.010);
    ASSERT_GT(ff.disturbanceFeedforward().cwiseAbs().maxCoeff(), 0.0);

    ff.enterDamping();
    EXPECT_EQ(ff.disturbanceFeedforward().cwiseAbs().maxCoeff(), 0.0)
        << "a stale feedforward survived safe-mode entry";
}

TEST(Feedforward, HoldsItsValueWhenNoFreshEstimateArrives)
{
    // A tick with no new estimate keeps the last one rather than dropping to
    // zero, because dropping to zero is itself a torque step. The node relies
    // on this when a mailbox read loses its race.
    ModelProbe ff(kMjcf, ffParams(1.0));
    const Eigen::VectorXd d = ffEstimateBelowCap(ff, 0.6);
    ff.setDisturbanceFeedforward(d, 0.010);
    const Eigen::VectorXd held = ff.disturbanceFeedforward();

    // No further call: computeControl must still see the retained value.
    const State s = referenceState(6);
    ff.setDesiredPos(0.30, 0.25, 0.0);
    ff.computeControl(s.q, s.v);
    for (int i = 0; i < held.size(); ++i) {
        EXPECT_DOUBLE_EQ(ff.disturbanceFeedforward()(i), held(i));
    }
}

TEST(Feedforward, NonFiniteEstimateIsIgnoredPerJoint)
{
    // A NaN must not reach an actuator, and must not poison the retained value
    // either: that joint holds what it had.
    ModelProbe ff(kMjcf, ffParams(1.0));
    Eigen::VectorXd good = ffEstimateBelowCap(ff, 0.6);
    ff.setDisturbanceFeedforward(good, 0.010);
    const double before = ff.disturbanceFeedforward()(0);
    ASSERT_NE(before, 0.0);

    Eigen::VectorXd poisoned = good;
    poisoned(0) = std::nan("");
    poisoned(1) = std::numeric_limits<double>::infinity();
    ff.setDisturbanceFeedforward(poisoned, 0.010);

    EXPECT_TRUE(ff.disturbanceFeedforward().allFinite());
    EXPECT_DOUBLE_EQ(ff.disturbanceFeedforward()(0), before)
        << "a NaN overwrote a good value instead of being skipped";
}

TEST(Feedforward, AppliedTorqueStillRespectsTheLimit)
{
    // The bounds shift exists so a torque limit keeps meaning what it says. A
    // large estimate plus an aggressive setpoint must still produce a command
    // inside the per-joint limits.
    const State s = referenceState(6);
    ModelProbe ff(kMjcf, ffParams(1.0, 0.5));
    ff.setDesiredPos(1.20, 0.80, 0.0);          // far away, so the law pushes hard
    ff.setDisturbanceFeedforward(
        Eigen::VectorXd::Constant(ff.n_joints_, 50.0), 0.010);

    for (int tick = 0; tick < 20; ++tick) {
        const Eigen::VectorXd tau = ff.computeControl(s.q, s.v);
        for (int i = 0; i < tau.size(); ++i) {
            EXPECT_LE(std::fabs(tau(i)),
                      ff.torqueLimsByIndex[static_cast<std::size_t>(i)] + 1e-9)
                << "tick " << tick << " joint " << i;
        }
    }
}

TEST(Feedforward, RejectsBadConfiguration)
{
    auto bad = [](const NMPCParams& p) {
        EXPECT_THROW((ModelProbe(kMjcf, p)), std::runtime_error);
    };
    bad(ffParams(-0.1));                  // negative gain
    bad(ffParams(1.5));                   // gain above full compensation
    bad(ffParams(0.5, 0.0));              // an unbounded feedforward
    bad(ffParams(0.5, 1.5));              // a cap above the torque limit
    bad(ffParams(0.5, 0.3, -1.0));        // negative slew
}

TEST(Feedforward, RejectsAMisSizedEstimate)
{
    ModelProbe ff(kMjcf, ffParams(1.0));
    EXPECT_THROW(ff.setDisturbanceFeedforward(
                     Eigen::VectorXd::Zero(ff.n_joints_ + 1), 0.010),
                 std::runtime_error);
}

TEST(RealtimeProfile, FeedforwardHandoffDoesNotAllocate)
{
    // It runs on the control thread, once per tick, before the solve.
    ModelProbe ff(kMjcf, ffParams(1.0, 0.3, 5.0));
    const Eigen::VectorXd d = ffEstimateBelowCap(ff, 0.6);
    ff.setDisturbanceFeedforward(d, 0.010);      // first call, sizing already done

    long allocs = 0;
    {
        AllocScope scope;
        for (int i = 0; i < 50; ++i) ff.setDisturbanceFeedforward(d, 0.010);
        allocs = scope.n();
    }
    EXPECT_EQ(allocs, 0);
}

// === FloatingBaseDynamics::fullDynamics ==================================== //
// The query the observer thread runs on. Its whole reason for existing is that
// dynamicsAt() and integrateStep() evaluate hypothetical states on mutable
// members, so a second thread cannot use them; these tests check that the
// Workspace form really does avoid that, because the alternative design (a
// second FloatingBaseDynamics, i.e. a second MJCF parse and a second model) was
// rejected on the strength of it.

TEST(FullDynamics, AgreesWithTheLiveStatePath)
{
    ModelProbe c(kMjcf, liveParams());
    const State s = referenceState(c.n_joints_);
    const int nv = c.numDof();
    ASSERT_EQ(nv, 6 + c.n_joints_);
    ASSERT_EQ(c.numConfig(), 7 + c.n_joints_);

    // The live path: load the state the way a control tick does, then read the
    // base-eliminated dynamics off it.
    c.loadLiveState(s.q, s.v);
    Eigen::MatrixXd H_g_live;
    Eigen::VectorXd Cv_live;
    c.getDynamics(H_g_live, Cv_live);

    // The Workspace path, at the same state.
    FloatingBaseDynamics::Workspace ws;
    Eigen::MatrixXd M, R;
    Eigen::VectorXd bias;
    c.fullDynamics(s.q, s.v, ws, M, bias, R);

    ASSERT_EQ(M.rows(), nv);
    ASSERT_EQ(M.cols(), nv);
    ASSERT_EQ(bias.size(), nv);
    ASSERT_EQ(R.rows(), 6);
    ASSERT_EQ(R.cols(), c.n_joints_);

    // The mass matrix must be symmetric and positive definite, or nothing
    // downstream of it means anything.
    EXPECT_LT((M - M.transpose()).cwiseAbs().maxCoeff(), 1e-9);
    EXPECT_GT(M.ldlt().vectorD().minCoeff(), 0.0);

    // Reconstruct the reduced quantities from the full ones and check they
    // match what the live path produced. This is the real equivalence claim:
    // H_g = H_mm - H_bm^T H_bb^-1 H_bm, and Cv = c_m - H_bm^T H_bb^-1 c_b.
    const Eigen::MatrixXd H_bm = M.topRightCorner(6, c.n_joints_);
    const Eigen::MatrixXd H_g_rebuilt =
        M.bottomRightCorner(c.n_joints_, c.n_joints_) - H_bm.transpose() * R;
    const Eigen::VectorXd Cv_rebuilt =
        bias.tail(c.n_joints_) - R.transpose() * bias.head(6);

    EXPECT_LT((H_g_rebuilt - H_g_live).cwiseAbs().maxCoeff(), 1e-9)
        << "the full mass matrix does not reduce to the live H_g";
    EXPECT_LT((Cv_rebuilt - Cv_live).cwiseAbs().maxCoeff(), 1e-9)
        << "qfrc_bias does not reduce to the live Cv_joints";
}

TEST(FullDynamics, DoesNotDisturbTheLiveStateOrTheRollout)
{
    // The observer thread must not change what the next control tick computes.
    //
    // Compared against a SECOND controller rather than against this one's own
    // previous tick: the NMPC warm-starts, so two identical computeControl
    // calls legitimately differ by a long way (0.19 N*m here), and asserting
    // they match would be testing the warm start, not the interference.
    auto drive = [](bool interleave) {
        ModelProbe c(kMjcf, liveParams());
        const State s = referenceState(c.n_joints_);
        c.setDesiredPos(0.30, 0.25, 0.0);
        c.computeControl(s.q, s.v);          // establishes the warm start

        if (interleave) {
            // Evaluate somewhere else entirely, repeatedly, exactly as the
            // observer thread would between two control ticks.
            FloatingBaseDynamics::Workspace ws;
            Eigen::MatrixXd M, R;
            Eigen::VectorXd bias;
            Eigen::VectorXd q2 = s.q, v2 = s.v;
            for (int i = 0; i < 20; ++i) {
                q2(7) += 0.01;
                v2(6) = 0.3;
                c.fullDynamics(q2, v2, ws, M, bias, R);
            }
        }
        return c.computeControl(s.q, s.v);   // the tick that must be unaffected
    };

    const Eigen::VectorXd clean = drive(false);
    const Eigen::VectorXd interleaved = drive(true);
    EXPECT_LT((interleaved - clean).cwiseAbs().maxCoeff(), 1e-12)
        << "fullDynamics perturbed the controller's own state\nclean       "
        << clean.transpose() << "\ninterleaved " << interleaved.transpose();
}

TEST(FullDynamics, TwoThreadsWithSeparateWorkspacesAgreeWithSerial)
{
    // The claim that justifies sharing one FloatingBaseDynamics between the
    // control thread and the observer thread. mjModel is read-only once loaded;
    // what must not be shared is the mjData, and a Workspace is exactly that.
    ModelProbe c(kMjcf, liveParams());
    const State s = referenceState(c.n_joints_);
    const int nv = c.numDof();

    // Two different states, so a workspace mix-up produces a wrong answer
    // rather than coincidentally the right one.
    Eigen::VectorXd qA = s.q, vA = s.v;
    Eigen::VectorXd qB = s.q, vB = s.v;
    qB(7) += 0.25;
    vB(6) = -0.4;
    vB(7) = 0.2;

    // Serial references.
    Eigen::MatrixXd MA_ref, RA_ref, MB_ref, RB_ref;
    Eigen::VectorXd bA_ref, bB_ref;
    {
        FloatingBaseDynamics::Workspace ws;
        c.fullDynamics(qA, vA, ws, MA_ref, bA_ref, RA_ref);
        c.fullDynamics(qB, vB, ws, MB_ref, bB_ref, RB_ref);
    }
    ASSERT_GT((MA_ref - MB_ref).cwiseAbs().maxCoeff(), 1e-6)
        << "the two states must differ, or this test cannot detect a mix-up";

    // Now concurrently, each thread with its own Workspace, hammering the
    // shared model for long enough to interleave.
    constexpr int kIters = 400;
    std::atomic<long> mismatches{0};
    auto worker = [&](const Eigen::VectorXd& q, const Eigen::VectorXd& v,
                      const Eigen::MatrixXd& M_ref, const Eigen::VectorXd& b_ref,
                      const Eigen::MatrixXd& R_ref) {
        FloatingBaseDynamics::Workspace ws;
        Eigen::MatrixXd M, R;
        Eigen::VectorXd bias;
        for (int i = 0; i < kIters; ++i) {
            c.fullDynamics(q, v, ws, M, bias, R);
            if ((M - M_ref).cwiseAbs().maxCoeff() > 1e-12 ||
                (bias - b_ref).cwiseAbs().maxCoeff() > 1e-12 ||
                (R - R_ref).cwiseAbs().maxCoeff() > 1e-12) {
                mismatches.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };

    std::thread ta([&] { worker(qA, vA, MA_ref, bA_ref, RA_ref); });
    std::thread tb([&] { worker(qB, vB, MB_ref, bB_ref, RB_ref); });
    ta.join();
    tb.join();

    std::printf("[fulldyn] %d concurrent evaluations per thread, nv=%d, "
                "%ld mismatches\n", kIters, nv, mismatches.load());
    EXPECT_EQ(mismatches.load(), 0)
        << "concurrent fullDynamics with separate workspaces diverged from "
           "serial -- something mutable is still shared";
}

TEST(FullDynamics, SteadyStateCallsDoNotAllocate)
{
    ModelProbe c(kMjcf, liveParams());
    const State s = referenceState(c.n_joints_);

    FloatingBaseDynamics::Workspace ws;
    Eigen::MatrixXd M, R;
    Eigen::VectorXd bias;
    // First call creates the mjData and sizes the outputs; that one is allowed
    // to allocate, and the observer thread makes it during bring-up.
    c.fullDynamics(s.q, s.v, ws, M, bias, R);

    long allocs = 0;
    {
        AllocScope scope;
        for (int i = 0; i < 20; ++i) c.fullDynamics(s.q, s.v, ws, M, bias, R);
        allocs = scope.n();
    }
    std::printf("[alloc] fullDynamics steady state %.2f per call\n",
                static_cast<double>(allocs) / 20.0);
    EXPECT_EQ(allocs, 0);
}

TEST(FullDynamics, RejectsAMisSizedState)
{
    ModelProbe c(kMjcf, liveParams());
    FloatingBaseDynamics::Workspace ws;
    Eigen::MatrixXd M, R;
    Eigen::VectorXd bias;
    const Eigen::VectorXd q_bad = Eigen::VectorXd::Zero(3);
    const Eigen::VectorXd v_ok = Eigen::VectorXd::Zero(c.numDof());
    EXPECT_THROW(c.fullDynamics(q_bad, v_ok, ws, M, bias, R), std::runtime_error);
}

/*! The observer is destined for its own realtime thread, so it is held to the
 *  same standard as the control tick -- and to a stricter one, since it has no
 *  returned-vector excuse: every buffer is sized in the constructor, so even
 *  the first update must reach the allocator zero times. */
TEST(RealtimeProfile, ObserverUpdateDoesNotAllocate)
{
    constexpr int kNv = 12;
    MomentumObserverParams p;
    p.nv = kNv;
    p.n_joints = kNv - 6;
    p.dt = 0.01;
    p.gain = Eigen::VectorXd::Constant(1, 20.0);
    p.mdot_clamp = 50.0;
    p.estimate_clamp = 5.0;
    MomentumObserver obs(p);

    // Inputs built outside the counted scope. Configuration-dependent, so the
    // Mdot difference is non-trivial and the clamp paths are reachable.
    Eigen::MatrixXd M = Eigen::MatrixXd::Identity(kNv, kNv) * 2.0;
    Eigen::VectorXd bias = Eigen::VectorXd::Constant(kNv, 0.01);
    Eigen::VectorXd v = Eigen::VectorXd::Constant(kNv, 0.3);
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(kNv);
    for (int i = 6; i < kNv; ++i) tau(i) = 0.5;
    Eigen::VectorXd reduced(p.n_joints);
    const Eigen::MatrixXd H_bb_inv_H_bm = Eigen::MatrixXd::Constant(6, p.n_joints, 0.1);

    long first = 0, steady = 0;
    {
        AllocScope scope;
        obs.update(M, bias, v, tau);
        first = scope.n();
    }

    constexpr long kTicks = 50;
    {
        AllocScope scope;
        for (long i = 0; i < kTicks; ++i) {
            // Vary M so the divided difference does real work each step.
            for (int k = 0; k < kNv; ++k) {
                M(k, k) = 2.0 + 0.1 * std::sin(0.01 * static_cast<double>(i * kNv + k));
            }
            obs.update(M, bias, v, tau);
            obs.reduceToJoints(H_bb_inv_H_bm, reduced);
        }
        steady = scope.n();
    }
    std::printf("[alloc] observer first update %ld, steady %.2f per update\n",
                first, static_cast<double>(steady) / kTicks);

    EXPECT_EQ(first, 0) << "the constructor is supposed to do all the sizing";
    EXPECT_EQ(steady, 0);
}

}  // namespace
}  // namespace pat_arm_control

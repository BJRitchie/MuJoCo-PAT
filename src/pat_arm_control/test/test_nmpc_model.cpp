/*! The NMPC's augmented prediction model against the plant.
 *
 *  Builds A/B at a known state through the probe's re-exported seams and
 *  compares the model's one-step prediction with MuJoCo's true one-step
 *  response, in the same augmented coordinates. Nothing here touches the QP:
 *  the question is only "does the linearised model predict the plant it is
 *  supposed to predict".
 */
#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "support/nmpc_fixtures.h"

namespace pat_arm_control {
namespace {
using namespace test_support;

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

}  // namespace
}  // namespace pat_arm_control

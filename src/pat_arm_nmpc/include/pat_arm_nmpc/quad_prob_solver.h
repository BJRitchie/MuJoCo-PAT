#pragma once

#include <numeric>  // std::iota
#include <stdexcept>
#include <vector>
#include <Eigen/Dense>

extern "C" {
#include "acados_c/ocp_qp_interface.h"
#include "acados/ocp_qp/ocp_qp_common.h"
}

namespace quad_prob_solver {

struct QuadProbSolverParams {
    int nx_task = 0;   //!< task-space error state dim (7 or 13)
    int nu      = 0;   //!< control dim (3 or 6)
    // Total state dim. For the bare task-space model this equals nx_task;
    // if the caller augments the state with Δq/qdot joint-limit rows (see
    // armConstrainedNMPController.cpp's reverted idxbx_joint design), this
    // is nx_task + 2*n_joints instead. Not derived automatically since this
    // class has no notion of "joints" -- the caller owns that math and just
    // tells the solver how big the state vector it will push actually is.
    int nx      = 0;
    int N       = 0;

    // --- QP solver tuning ----------------------------------------------------
    int    qp_max_iter = 50;      //!< HPIPM iteration cap ("iter_max" field)
    double qp_tol      = 1e-6;    //!< HPIPM stationarity convergence tolerance ("tol_stat" field)

    // Primal-Hessian ridge added inside HPIPM's own IPM factorization at every
    // iteration ("reg_prim" field) — independent of R's regularization of the
    // closed-form KKT system, this keeps the IPM's per-iteration Riccati
    // factorization well-conditioned when the caller's B is near-singular
    // (e.g. near a kinematic singularity, or under the worse GJM conditioning
    // seen in dual-arm setups). See armConstrainedNMPController.cpp for the
    // ill-conditioning diagnostic (Lambda_inv6 condition-number check) this
    // is meant to harden against.
    double qp_reg_prim = 1e-6;

    // HPIPM warm-start mode ("warm_start" field): 0 = cold start every solve,
    // 1 = warm-start primal variables only, 2 = warm-start primal + dual.
    // Reuses qp_out's iterate from the PREVIOUS solve as this solve's IPM
    // starting point — valid whenever consecutive solve() calls use only
    // slightly-perturbed A/B/Q/R (e.g. successive linearization).
    int qp_warm_start = 2;

    // Indices (into the nx-dim state) that are box-constrained at stages
    // 1..N. Empty means "no constraints beyond the stage-0 initial-condition
    // pin" (only stage 0 is ever pinned to x0, matching
    // ArmConstrainedNMPController's current MVP behavior).
    std::vector<int> idxbx_k;

    // Every row in idxbx_k is enforced as a SOFT constraint (slack s >= 0,
    // penalized z*s + 1/2*Z*s^2), not hard -- so the QP can never go
    // infeasible purely because the state is currently outside a box bound
    // (e.g. a joint that's strayed past its limit). The penalty still
    // creates strong pressure back toward the true bound; it just can't
    // lock the solver up the way a hard bound can when the correction isn't
    // reachable within one horizon step. Stage 0 (the initial-condition
    // pin) is never softened -- it's an equality on the actual measurement,
    // not a limit under negotiation.
    // Defaults are large relative to typical Q/R scales (see e.g. acados'
    // own mass_spring_qp.c soft-constraint example, which uses the same
    // 1e2/1e3 order of magnitude) -- respect the bound under normal
    // operation, but stay finite so violation is a fallback, not a wall.
    double slack_penalty_linear    = 1e2;  //!< z: cost per unit of slack
    double slack_penalty_quadratic = 1e3;  //!< Z: cost per unit^2 of slack

    // Number of general (polytopic) constraint rows: lg <= C*x + D*u <= ug,
    // pushed fresh every solve() call via D_data/lg_data/ug_data (C is
    // always the zero matrix in this class -- no caller currently needs a
    // state-dependent general constraint, only ones on a linear transform
    // of the control, e.g. tau_joints = J_g^T*tau_task). Active at stages
    // 0..N-1 (wherever a control decision exists) -- NOT the same stage
    // range as idxbx_k (1..N, since that bounds STATE rows and stage 0's
    // state is the hard measurement pin, while stage 0's CONTROL is a free
    // decision variable that a torque-style bound still applies to). ng=0
    // (the default) disables general constraints entirely.
    //
    // KNOWN LIMITATION (confirmed via isolated testing, not a design
    // choice): unlike soft idxbx_k box rows, which tolerate a genuinely
    // inverted bound (lbx > ubx) via slack just fine, soft general rows
    // return a NaN solve status for ANY lg > ug, even by 1e-6 -- lg == ug
    // (zero-width, still valid) works correctly, only strict inversion
    // breaks. Callers must ensure lg <= ug always holds. For
    // ArmConstrainedNMPController's torque-limit use (lg = -tauLim -
    // Cv_joints, ug = tauLim - Cv_joints), this is automatically guaranteed
    // since ug - lg = 2*tauLim is always positive for tauLim >= 0,
    // regardless of Cv_joints -- but a future caller constructing lg/ug
    // differently should keep this in mind.
    int ng = 0;

    // Every general-constraint row is soft too, same rationale as
    // slack_penalty_linear/quadratic above -- kept as SEPARATE weights
    // since general (e.g. torque) constraints typically live on a very
    // different natural scale than the box (e.g. joint position/velocity)
    // ones and want independent tuning.
    double general_slack_penalty_linear    = 1e2;
    double general_slack_penalty_quadratic = 1e3;

    // Indices (into the nx-dim state) that are HARD-equality-constrained to
    // ZERO at stage N ONLY (the terminal stage) -- e.g. a task-space
    // velocity-error row, so a feasible solve implies "this MPC has a
    // trajectory that ends at zero velocity error." Empty (default) means
    // no terminal constraint beyond the running idxbx_k bounds, i.e.
    // today's behavior exactly.
    //
    // UNLIKE idxbx_k (soft everywhere it applies), these rows are HARD --
    // the classical MPC recursive-feasibility argument (if this tick's
    // solve is feasible, next tick's solve is GUARANTEED feasible too,
    // given accurate dynamics) only holds for a true equality terminal
    // constraint; a soft one is just a strong incentive, not a guarantee.
    // The cost of that guarantee: solve()/solveMultiStage() CAN now return
    // false if the terminal velocity genuinely isn't reachable within N
    // steps (e.g. too short a horizon, or right after an already-strained
    // tick) -- same as any other hard infeasibility, the caller's existing
    // solve-failure fallback handles this safely.
    //
    // Recursive feasibility ALSO requires whatever Jacobian/inertia term
    // the caller's dynamics model uses to construct the "hold the terminal
    // state" control (e.g. Lambda_inv in ArmConstrainedNMPController's
    // buildAugmentedModel) to stay INVERTIBLE at the shifted terminal
    // state -- near a kinematic singularity this can break down even with
    // this constraint in place. And it's a clean guarantee for solve()'s
    // single-linearization-per-tick usage; for solveMultiStage()'s true
    // nonlinear multi-shooting path, the per-node defect (b_k) breaks the
    // classical theorem's exact preconditions, so treat it as a strong
    // heuristic there, not a proof.
    std::vector<int> idxbx_terminal;

    QuadProbSolverParams() = default;
};


/*! @brief Thin, reusable wrapper around the acados_c + HPIPM box-constrained
 *  QP solve sequence used by ArmConstrainedNMPController — see
 *  armConstrainedNMPController.cpp's file-level comment for the full
 *  rationale on this specific hybrid use of the acados_c and HPIPM layers.
 *
 *  Owns one HPIPM/acados solver instance for the lifetime of the object
 *  (dims/config/opts fixed at construction, matching how many joints/what
 *  horizon length are being solved for); solve() pushes fresh numeric
 *  A/B/Q/R/bounds every call and returns the receding-horizon control.
 */
class QuadProbSolver {
public: 
    explicit QuadProbSolver(const QuadProbSolverParams& params);
    ~QuadProbSolver();

    // Not copyable (owns raw acados/HPIPM handles with no reference
    // counting); moving is not implemented either since nothing in this
    // codebase currently needs it.
    QuadProbSolver(const QuadProbSolver&) = delete;
    QuadProbSolver& operator=(const QuadProbSolver&) = delete;

    // Solve the QP based on input linearized matrices (column-major storage,
    // matches Eigen's default -- pass Eigen matrix/vector .data() directly).
    // lbx_data/ubx_data must have params.idxbx_k.size() elements (the
    // stage 1..N box bounds); D_data/lg_data/ug_data must have
    // params.ng*nu / params.ng / params.ng elements respectively (the
    // stage 0..N-1 general constraints -- pass nullptr for all three when
    // params.ng == 0); x0_current must have params.nx elements (the
    // stage-0 initial-condition pin). Returns false (without touching the
    // previously stored solution) if the solve fails to converge.
    //
    // Q_N_data (optional, default nullptr): a distinct nx*nx cost matrix
    // (same column-major layout as Q_data) used ONLY at the terminal stage
    // N, in place of Q_data -- a standard MPC "terminal cost" (e.g. a
    // heavier weight on position error to counter steady-state tracking
    // offset), deliberately a soft cost rather than a hard terminal
    // constraint (see idxbx_terminal's doc comment above for why a hard
    // terminal constraint is only safe for quantities -- like velocity --
    // that are reachable from ANY state given enough control authority;
    // position generally isn't, within one horizon). When null (the
    // default), stage N reuses Q_data exactly as before this parameter
    // existed -- zero behavior change for callers that don't pass it.
    bool solve(
        double A_data[],   double B_data[], double b_data[],
        double Q_data[],   double R_data[],
        double lbx_data[], double ubx_data[],
        double D_data[],   double lg_data[], double ug_data[],
        double x0_current[],
        double Q_N_data[] = nullptr );

    // Per-stage generalization of solve() for true multiple-shooting NMPC:
    // A_k/B_k/b_k/D_k/lg_k/ug_k are genuinely re-linearized at EACH shooting
    // node, unlike solve()'s single A/B/D repeated across the whole horizon
    // -- b_k in particular is the direct-multiple-shooting linearization
    // defect (solve() always passes a zero vector here; see
    // armConstrainedNMPController.cpp's solveNonlinearMPC()). Q/R/lbx/ubx
    // stay stage-constant, same rationale solve() already has (cost weights
    // and static joint limits don't vary by linearization node). Each
    // vector below must have exactly N entries (stage k = 0..N-1), sized
    // A_k[k]: nx x nx, B_k[k]: nx x nu, b_k[k]: nx, D_k[k]: ng x nu,
    // lg_k[k]/ug_k[k]: ng (D_k/lg_k/ug_k may be empty vectors when
    // params.ng == 0, matching solve()'s nullptr convention); x0_current
    // has params.nx elements, as in solve(). Returns false (without
    // touching the previously stored solution) if the solve fails to
    // converge -- same contract as solve().
    //
    // Q_N_data: same optional terminal-cost override as solve()'s -- see
    // that function's doc comment.
    bool solveMultiStage(
        const std::vector<Eigen::MatrixXd>& A_k,
        const std::vector<Eigen::MatrixXd>& B_k,
        const std::vector<Eigen::VectorXd>& b_k,
        double Q_data[], double R_data[],
        double lbx_data[], double ubx_data[],
        const std::vector<Eigen::MatrixXd>& D_k,
        const std::vector<Eigen::VectorXd>& lg_k,
        const std::vector<Eigen::VectorXd>& ug_k,
        double x0_current[],
        double Q_N_data[] = nullptr );

    // Retrieve the optimal control at a given stage (0 <= stage < N) from
    // the most recent successful solve, into caller-owned storage of size
    // params.nu.
    void getU(int stage, double* u_out) const;
    // Retrieve the predicted state at a given stage (0 <= stage <= N) from
    // the most recent successful solve, into caller-owned storage of size
    // params.nx.
    void getX(int stage, double* x_out) const;

    // Number of soft-constraint rows (and hence the required size of
    // getSlack's sl_out/su_out buffers) at a given stage: idxbx_k.size()
    // box rows plus ng general rows, MINUS whichever of the two isn't
    // active at that stage -- stage 0 has general rows only (ng), stage N
    // has box rows only (idxbx_k.size()), stages 1..N-1 have both. Returns
    // 0 if stage is out of [0, N] or neither constraint type is in use.
    int nsAt(int stage) const;

    // Retrieve the lower/upper slack values (how far the soft rows at this
    // stage exceeded their bound, 0 if within bound) from the most recent
    // successful solve, into caller-owned storage of size nsAt(stage). When
    // both box and general rows are soft at this stage (see nsAt), the
    // first idxbx_k.size() entries are the box (e.g. joint position/
    // velocity) slack and the remaining ng are the general (e.g. torque)
    // slack -- same order the constructor wires idxs in. A large slack
    // means the solve succeeded but is leaning heavily on the penalty
    // rather than respecting the real bound -- worth surfacing as a
    // diagnostic (see armConstrainedNMPController.cpp).
    void getSlack(int stage, double* sl_out, double* su_out) const;

    // Convenience accessor for the stage-0 control from the most recent
    // successful solve (the receding-horizon control ArmConstrainedNMPController
    // actually applies each tick).
    const Eigen::VectorXd& lastTauTask() const { return last_tau_task; }

private:
    QuadProbSolverParams params_;
    ocp_qp_solver_plan_t plan;

    ocp_qp_dims*                dims       = nullptr;
    ocp_qp_xcond_solver_config* config     = nullptr;
    ocp_qp_xcond_solver_dims*   xcond_dims = nullptr;
    ocp_qp_xcond_solver_opts*   opts       = nullptr;
    ocp_qp_solver*              solver     = nullptr;

    ocp_qp_in*  qp_in  = nullptr;
    ocp_qp_out* qp_out = nullptr;

    // Stage-0 index set (pins the full state -- always {0, ..., nx-1}).
    std::vector<int> idxbx_initial;

    // Fixed at construction: the terminal constraint's bound is always
    // exactly 0 and never varies tick-to-tick (same idea as
    // idxbx_initial for stage 0's pin). Concatenated onto the caller's
    // idxbx_k-sized lbx/ubx before pushing stage N's bounds in solve()/
    // solveMultiStage(). Empty when params_.idxbx_terminal is empty.
    Eigen::VectorXd terminal_zero_bound_;

    Eigen::VectorXd last_tau_task;
};

}  // namespace quad_prob_solver

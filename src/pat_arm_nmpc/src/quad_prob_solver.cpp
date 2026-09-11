#include "pat_arm_nmpc/quad_prob_solver.h"

#include <iostream>

namespace quad_prob_solver {

// === Construction =============================================================
QuadProbSolver::QuadProbSolver( const QuadProbSolverParams& params )
    : params_(params)
{
    if (params_.nx_task <= 0) {
        throw std::runtime_error("QuadProbSolver: QuadProbSolverParams.nx_task must be > 0");
    }
    if (params_.N < 1) {
        throw std::runtime_error("QuadProbSolver: QuadProbSolverParams.N must be >= 1");
    }
    if (params_.nx <= 0) {
        throw std::runtime_error("QuadProbSolver: QuadProbSolverParams.nx must be > 0");
    }
    if (params_.nu <= 0) {
        throw std::runtime_error("QuadProbSolver: QuadProbSolverParams.nu must be > 0");
    }

    // 1. Problem Specification
    const int N  = params_.N;    // Horizon steps (N control intervals, N+1 stages)
    const int nx = params_.nx;
    const int nu = params_.nu;

    // Generate Stage 0 state index sequence: [0, 1, 2, ..., nx - 1]
    idxbx_initial.resize(nx);
    std::iota(idxbx_initial.begin(), idxbx_initial.end(), 0);

    // Stage 1..N bounded states and count
    const std::vector<int>& idxbx_k = params_.idxbx_k;
    const int nbx_k = static_cast<int>(idxbx_k.size());

    // Stage-N-ONLY hard terminal indices (see QuadProbSolverParams::
    // idxbx_terminal's doc comment for the recursive-feasibility rationale).
    const std::vector<int>& idxbx_terminal = params_.idxbx_terminal;
    const int n_terminal = static_cast<int>(idxbx_terminal.size());
    terminal_zero_bound_ = Eigen::VectorXd::Zero(n_terminal);

    // Stage N's combined bounded-index set: idxbx_k's rows (still soft)
    // FIRST, idxbx_terminal's rows (hard) appended after -- this ordering
    // is what lets the existing idxsN/ZlN/ZuN/zlN/zuN soft-mapping below
    // (which only marks the first nbx_k LOCAL rows at stage N as soft)
    // automatically leave the terminal rows hard, with no changes to that
    // block needed.
    std::vector<int> idxbx_N_combined;
    idxbx_N_combined.reserve(nbx_k + n_terminal);
    idxbx_N_combined.insert(idxbx_N_combined.end(), idxbx_k.begin(), idxbx_k.end());
    idxbx_N_combined.insert(idxbx_N_combined.end(), idxbx_terminal.begin(), idxbx_terminal.end());

    // 2. One-time setup (ACADOS layer 1)
    plan.qp_solver = PARTIAL_CONDENSING_HPIPM;

    // The config type is ocp_qp_xcond_solver_config, not "ocp_qp_solver_config"
    // -- "xcond" ("x-conditioning") is the layer that lets the same generic
    // API target both partial- and full-condensing solvers.
    config = ocp_qp_xcond_solver_config_create(plan);

    // Plain problem dims -- IS d_ocp_qp_dim under the hood
    // so it's built with HPIPM's own direct setters.
    const int ng = params_.ng;

    dims = ocp_qp_dims_create(N);
    for (int k = 0; k <= N; ++k) {
        // States and control inputs
        d_ocp_qp_dim_set_nx(k, nx, dims);
        d_ocp_qp_dim_set_nu(k, (k < N) ? nu : 0, dims);    // terminal stage: no control

        // Stage 0: pin full state (nx bounds); stages 1..N-1: nbx_k bounds;
        // stage N: nbx_k PLUS n_terminal (the hard terminal-only rows
        // appended after, see idxbx_N_combined above) -- NOT a hardcoded
        // constant, since idxbx_k/idxbx_terminal's sizes vary per caller.
        const int nbx_this_stage = (k == 0) ? nx : (k == N ? nbx_k + n_terminal : nbx_k);
        d_ocp_qp_dim_set_nbx(k, nbx_this_stage, dims);
        d_ocp_qp_dim_set_nbu(k, 0, dims);

        // General (polytopic) constraints: lg <= C*x + D*u <= ug. Active at
        // stages 0..N-1 only (wherever a control decision exists -- the
        // terminal stage has nu=0, nothing to constrain there), which is a
        // DIFFERENT stage range than idxbx_k's 1..N (state rows: stage 0's
        // state is the hard measurement pin, but stage 0's CONTROL is a
        // free decision variable a general/torque-style bound still
        // applies to).
        const int ng_k = (k < N) ? ng : 0;
        d_ocp_qp_dim_set_ng(k, ng_k, dims);

        // Every idxbx_k row at stages 1..N and every general-constraint row
        // at stages 0..N-1 is SOFT -- see QuadProbSolverParams::
        // slack_penalty_linear/general_slack_penalty_linear's doc comments
        // for why. Stage 0's box rows stay hard (nsbx=0 there): it's the
        // measured initial condition, not a negotiable limit.
        const int nsbx_k = (k == 0) ? 0 : nbx_k;
        const int nsg_k  = ng_k;
        d_ocp_qp_dim_set_nsbx(k, nsbx_k, dims);
        d_ocp_qp_dim_set_nsg(k, nsg_k, dims);
        d_ocp_qp_dim_set_ns(k, nsbx_k + nsg_k, dims);
    }

    // xcond_dims is a SEPARATE dims representation derived from the plain
    // dims above -- required by the opts/solver constructors below.
    xcond_dims = ocp_qp_xcond_solver_dims_create_from_ocp_qp_dims(config, dims);

    // Solver options (iteration cap, tolerances, regularization, warm-start).
    // Field names here are HPIPM's OWN option names -- opts_set forwards
    // them straight down to HPIPM's arg-struct setter (see
    // armConstrainedNMPController.cpp's constructor for the same pattern,
    // already verified against hpipm_d_ocp_qp_ipm.h).
    opts = static_cast<ocp_qp_xcond_solver_opts*>(
        ocp_qp_xcond_solver_opts_create(config, xcond_dims));
    ocp_qp_xcond_solver_opts_set(config, opts, "iter_max", &params_.qp_max_iter);
    ocp_qp_xcond_solver_opts_set(config, opts, "tol_stat", &params_.qp_tol);
    ocp_qp_xcond_solver_opts_set(config, opts, "reg_prim", &params_.qp_reg_prim);
    ocp_qp_xcond_solver_opts_set(config, opts, "warm_start", &params_.qp_warm_start);

    // qp_in/qp_out are sized from the PLAIN dims, not xcond_dims.
    qp_in  = ocp_qp_in_create(dims);
    qp_out = ocp_qp_out_create(dims);

    // The actual solver instance -- takes config + xcond_dims + opts (three
    // args, not two -- "ocp_qp_solver_create" doesn't exist). Assign
    // directly to the MEMBER (not a local shadowing it, which would leave
    // this object's own `solver` null and leak the one actually created).
    solver = ocp_qp_create(config, xcond_dims, opts);

    // 3. Fixed Index Sets
    // Stage 0: full state index vector [0 ... nx-1]
    d_ocp_qp_set_idxbx(0, idxbx_initial.data(), qp_in);

    // Stages 1..N-1: user-defined state bound indices (skipped entirely when
    // idxbx_k is empty -- matches nbx=0 at those stages above, so no bounds
    // array is required or expected by HPIPM).
    if (nbx_k > 0) {
        for (int k = 1; k <= N - 1; ++k) {
            d_ocp_qp_set_idxbx(k, const_cast<int*>(idxbx_k.data()), qp_in);
        }
    }
    // Stage N: idxbx_k's rows PLUS the hard terminal rows appended after
    // (idxbx_N_combined, built above) -- a DIFFERENT (longer, when
    // n_terminal>0) index array than stages 1..N-1 use, so it needs its own
    // push rather than being folded into the loop above.
    if (nbx_k > 0 || n_terminal > 0) {
        d_ocp_qp_set_idxbx(N, idxbx_N_combined.data(), qp_in);
    }

    // General constraints: C is always the zero matrix in this class (no
    // caller currently needs a state-dependent general constraint -- see
    // QuadProbSolverParams::ng's doc comment), so it's fixed here at
    // construction rather than re-pushed every solve() call.
    if (ng > 0) {
        Eigen::MatrixXd C_zero = Eigen::MatrixXd::Zero(ng, nx);
        for (int k = 0; k < N; ++k) {
            d_ocp_qp_set_C(k, C_zero.data(), qp_in);
        }
    }

    // Soft-constraint wiring. idxs[j] maps slack j into the stage's
    // box-then-general constraint index space ([0, nbx) for state boxes,
    // [nbx, nbx+ng) for general rows, since nbu is always 0 in this class)
    // -- confirmed against acados' own soft-constraint example
    // (examples/c/no_interface_examples/mass_spring_model/mass_spring_qp.c),
    // which uses this same box-then-general numbering. The three stage
    // ranges have different soft-row compositions (see the dims loop
    // above), so each needs its own idxs/Zl/Zu/zl/zu built once and pushed
    // to its applicable stages:
    //   stage 0      : general rows only (box is hard there)
    //   stages 1..N-1: box rows then general rows, combined
    //   stage N      : box rows only (no control there, so no general rows)
    if (ng > 0) {
        std::vector<int> idxs0(ng);
        std::iota(idxs0.begin(), idxs0.end(), 0);
        Eigen::VectorXd Zl0 = Eigen::VectorXd::Constant(ng, params_.general_slack_penalty_quadratic);
        Eigen::VectorXd Zu0 = Eigen::VectorXd::Constant(ng, params_.general_slack_penalty_quadratic);
        Eigen::VectorXd zl0 = Eigen::VectorXd::Constant(ng, params_.general_slack_penalty_linear);
        Eigen::VectorXd zu0 = Eigen::VectorXd::Constant(ng, params_.general_slack_penalty_linear);
        d_ocp_qp_set_idxs(0, idxs0.data(), qp_in);
        d_ocp_qp_set_Zl(0, Zl0.data(), qp_in);
        d_ocp_qp_set_Zu(0, Zu0.data(), qp_in);
        d_ocp_qp_set_zl(0, zl0.data(), qp_in);
        d_ocp_qp_set_zu(0, zu0.data(), qp_in);
    }

    if (nbx_k > 0 || ng > 0) {
        const int ns_mid = nbx_k + ng;
        std::vector<int> idxs_mid(ns_mid);
        std::iota(idxs_mid.begin(), idxs_mid.end(), 0);
        Eigen::VectorXd Zl_mid(ns_mid), Zu_mid(ns_mid), zl_mid(ns_mid), zu_mid(ns_mid);
        Zl_mid.head(nbx_k).setConstant(params_.slack_penalty_quadratic);
        Zu_mid.head(nbx_k).setConstant(params_.slack_penalty_quadratic);
        zl_mid.head(nbx_k).setConstant(params_.slack_penalty_linear);
        zu_mid.head(nbx_k).setConstant(params_.slack_penalty_linear);
        Zl_mid.tail(ng).setConstant(params_.general_slack_penalty_quadratic);
        Zu_mid.tail(ng).setConstant(params_.general_slack_penalty_quadratic);
        zl_mid.tail(ng).setConstant(params_.general_slack_penalty_linear);
        zu_mid.tail(ng).setConstant(params_.general_slack_penalty_linear);

        for (int k = 1; k <= N - 1; ++k) {
            d_ocp_qp_set_idxs(k, idxs_mid.data(), qp_in);
            d_ocp_qp_set_Zl(k, Zl_mid.data(), qp_in);
            d_ocp_qp_set_Zu(k, Zu_mid.data(), qp_in);
            d_ocp_qp_set_zl(k, zl_mid.data(), qp_in);
            d_ocp_qp_set_zu(k, zu_mid.data(), qp_in);
        }
    }

    if (nbx_k > 0) {
        std::vector<int> idxsN(nbx_k);
        std::iota(idxsN.begin(), idxsN.end(), 0);
        Eigen::VectorXd ZlN = Eigen::VectorXd::Constant(nbx_k, params_.slack_penalty_quadratic);
        Eigen::VectorXd ZuN = Eigen::VectorXd::Constant(nbx_k, params_.slack_penalty_quadratic);
        Eigen::VectorXd zlN = Eigen::VectorXd::Constant(nbx_k, params_.slack_penalty_linear);
        Eigen::VectorXd zuN = Eigen::VectorXd::Constant(nbx_k, params_.slack_penalty_linear);
        d_ocp_qp_set_idxs(N, idxsN.data(), qp_in);
        d_ocp_qp_set_Zl(N, ZlN.data(), qp_in);
        d_ocp_qp_set_Zu(N, ZuN.data(), qp_in);
        d_ocp_qp_set_zl(N, zlN.data(), qp_in);
        d_ocp_qp_set_zu(N, zuN.data(), qp_in);
    }

    last_tau_task = Eigen::VectorXd::Zero(nu);
}

QuadProbSolver::~QuadProbSolver() {
    if (qp_out)     ocp_qp_out_free(qp_out);
    if (qp_in)      ocp_qp_in_free(qp_in);
    if (solver)     ocp_qp_solver_destroy(solver);
    if (opts)       ocp_qp_xcond_solver_opts_free(opts);
    if (xcond_dims) ocp_qp_xcond_solver_dims_free(xcond_dims);
    if (config)     ocp_qp_xcond_solver_config_free(config);
    if (dims)       ocp_qp_dims_free(dims);
}

bool QuadProbSolver::solve(
    double A_data[],   double B_data[], double b_data[],
    double Q_data[],   double R_data[],
    double lbx_data[], double ubx_data[],
    double D_data[],   double lg_data[], double ug_data[],
    double x0_current[],
    double Q_N_data[]
) {
    const int N  = params_.N;
    const int nx = params_.nx;
    const int nu = params_.nu;
    const int nbx_k = static_cast<int>(params_.idxbx_k.size());
    const int ng = params_.ng;

    Eigen::VectorXd zero_vec = Eigen::VectorXd::Zero(nx);
    Eigen::VectorXd zero_u   = Eigen::VectorXd::Zero(nu);

    for (int k = 0; k < N; ++k) {
        d_ocp_qp_set_A(k, A_data, qp_in);
        d_ocp_qp_set_B(k, B_data, qp_in);
        d_ocp_qp_set_b(k, b_data, qp_in);

        d_ocp_qp_set_Q(k, Q_data, qp_in);
        d_ocp_qp_set_R(k, R_data, qp_in);
        d_ocp_qp_set_q(k, zero_vec.data(), qp_in);
        d_ocp_qp_set_r(k, zero_u.data(), qp_in);

        if (k > 0 && nbx_k > 0) {
            d_ocp_qp_set_lbx(k, lbx_data, qp_in);
            d_ocp_qp_set_ubx(k, ubx_data, qp_in);
        }
        if (ng > 0) {
            // D (the linear map from control to the constrained quantity,
            // e.g. J_g^T for torque) is re-pushed every tick since it's
            // derived from the current linearization, same as A/B. C stays
            // the fixed zero matrix set once at construction.
            d_ocp_qp_set_D(k, D_data, qp_in);
            d_ocp_qp_set_lg(k, lg_data, qp_in);
            d_ocp_qp_set_ug(k, ug_data, qp_in);
        }
    }

    // Terminal stage: state cost + box bound only -- no A/B/R, since
    // nothing transitions OUT of the terminal stage. The terminal cost
    // matrix is Q_N_data when the caller supplied one, else Q_data (today's
    // behavior) -- see solve()'s doc comment. When idxbx_terminal is
    // non-empty, stage N's bound array is LONGER than lbx_data/ubx_data
    // (nbx_k + n_terminal, matching idxbx_N_combined built in the
    // constructor) -- concatenate the caller's idxbx_k-sized bounds with
    // the fixed terminal_zero_bound_ (always exactly 0) before pushing.
    d_ocp_qp_set_Q(N, Q_N_data ? Q_N_data : Q_data, qp_in);
    d_ocp_qp_set_q(N, zero_vec.data(), qp_in);
    const int n_terminal = static_cast<int>(params_.idxbx_terminal.size());
    if (n_terminal > 0) {
        Eigen::VectorXd lbxN(nbx_k + n_terminal), ubxN(nbx_k + n_terminal);
        if (nbx_k > 0) {
            lbxN.head(nbx_k) = Eigen::Map<Eigen::VectorXd>(lbx_data, nbx_k);
            ubxN.head(nbx_k) = Eigen::Map<Eigen::VectorXd>(ubx_data, nbx_k);
        }
        lbxN.tail(n_terminal) = terminal_zero_bound_;
        ubxN.tail(n_terminal) = terminal_zero_bound_;
        d_ocp_qp_set_lbx(N, lbxN.data(), qp_in);
        d_ocp_qp_set_ubx(N, ubxN.data(), qp_in);
    } else if (nbx_k > 0) {
        d_ocp_qp_set_lbx(N, lbx_data, qp_in);
        d_ocp_qp_set_ubx(N, ubx_data, qp_in);
    }

    // Initial-condition pin: lbx == ubx == measured x0, over ALL nx rows
    // -- the standard HPIPM idiom for pinning the OCP's initial state
    // (a box constraint collapsed to a single point).
    d_ocp_qp_set_lbx(0, x0_current, qp_in);
    d_ocp_qp_set_ubx(0, x0_current, qp_in);

    // 5. Solve (Acados Layer 1)
    // On the very first call, qp_out's iterate is whatever ocp_qp_out_create
    // left it as -- not guaranteed zeroed -- so warm-starting off it before
    // any real solve has run can seed HPIPM's IPM with garbage and produce
    // a NaN solution (status NAN_SOL). A failed solve leaves qp_out
    // untouched (see this function's returning branch below), so that same
    // garbage would otherwise keep getting warm-started into NaN on every
    // subsequent call forever -- confirmed via isolated testing on a small
    // (nx_task=5) problem. Guard against this generally (not just the first
    // call, since the same self-perpetuating-NaN failure mode could in
    // principle recur after any solve that doesn't converge): if the
    // configured warm_start mode fails, retry once with a forced cold start
    // before giving up -- cold-starting reliably recovers (verified: zero
    // failures running entirely cold), and this keeps the fast path
    // (warm-started) for every normal tick.
    int qp_status = ocp_qp_solve(solver, qp_in, qp_out);
    if (qp_status != 0 && params_.qp_warm_start != 0) {
        int cold = 0;
        ocp_qp_xcond_solver_opts_set(config, opts, "warm_start", &cold);
        qp_status = ocp_qp_solve(solver, qp_in, qp_out);
        ocp_qp_xcond_solver_opts_set(config, opts, "warm_start", &params_.qp_warm_start);
    }
    if (qp_status == 0) {
        // 6. Extract solution -- stash stage-0 control on the object so
        // callers that only care about the receding-horizon control (the
        // common case) can use lastTauTask()/getU(0, ...) without having to
        // know acados/HPIPM types at all. On failure, the PREVIOUS solution
        // is left untouched rather than overwritten with garbage.
        Eigen::VectorXd u0_sol(nu);
        d_ocp_qp_sol_get_u(0, qp_out, u0_sol.data());
        last_tau_task = u0_sol;

        return true;
    } else {
        std::cerr << "QuadProbSolver: QP solve failed with status " << qp_status << std::endl;
        return false;
    }
}

bool QuadProbSolver::solveMultiStage(
    const std::vector<Eigen::MatrixXd>& A_k,
    const std::vector<Eigen::MatrixXd>& B_k,
    const std::vector<Eigen::VectorXd>& b_k,
    double Q_data[], double R_data[],
    double lbx_data[], double ubx_data[],
    const std::vector<Eigen::MatrixXd>& D_k,
    const std::vector<Eigen::VectorXd>& lg_k,
    const std::vector<Eigen::VectorXd>& ug_k,
    double x0_current[],
    double Q_N_data[] 
) {
    const int N  = params_.N;
    const int nbx_k = static_cast<int>(params_.idxbx_k.size());
    const int ng = params_.ng;

    if (static_cast<int>(A_k.size()) != N || static_cast<int>(B_k.size()) != N ||
        static_cast<int>(b_k.size()) != N) {
        throw std::invalid_argument(
            "QuadProbSolver::solveMultiStage: A_k/B_k/b_k must each have exactly N entries");
    }
    if (ng > 0 && (static_cast<int>(D_k.size()) != N || static_cast<int>(lg_k.size()) != N ||
                   static_cast<int>(ug_k.size()) != N)) {
        throw std::invalid_argument(
            "QuadProbSolver::solveMultiStage: D_k/lg_k/ug_k must each have exactly N entries when ng > 0");
    }

    Eigen::VectorXd zero_vec = Eigen::VectorXd::Zero(params_.nx);
    Eigen::VectorXd zero_u   = Eigen::VectorXd::Zero(params_.nu);

    for (int k = 0; k < N; ++k) {
        if (A_k[k].rows() != params_.nx || A_k[k].cols() != params_.nx ||
            B_k[k].rows() != params_.nx || B_k[k].cols() != params_.nu ||
            b_k[k].size() != params_.nx) {
            throw std::invalid_argument(
                "QuadProbSolver::solveMultiStage: A_k/B_k/b_k[" + std::to_string(k) +
                "] size mismatch against params.nx/nu");
        }
        if (ng > 0 && (D_k[k].rows() != ng || D_k[k].cols() != params_.nu ||
                       lg_k[k].size() != ng || ug_k[k].size() != ng)) {
            throw std::invalid_argument(
                "QuadProbSolver::solveMultiStage: D_k/lg_k/ug_k[" + std::to_string(k) +
                "] size mismatch against params.ng/nu");
        }
        d_ocp_qp_set_A(k, const_cast<double*>(A_k[k].data()), qp_in);
        d_ocp_qp_set_B(k, const_cast<double*>(B_k[k].data()), qp_in);
        d_ocp_qp_set_b(k, const_cast<double*>(b_k[k].data()), qp_in);

        d_ocp_qp_set_Q(k, Q_data, qp_in);
        d_ocp_qp_set_R(k, R_data, qp_in);
        d_ocp_qp_set_q(k, zero_vec.data(), qp_in);
        d_ocp_qp_set_r(k, zero_u.data(), qp_in);

        if (k > 0 && nbx_k > 0) {
            d_ocp_qp_set_lbx(k, lbx_data, qp_in);
            d_ocp_qp_set_ubx(k, ubx_data, qp_in);
        }
        if (ng > 0) {
            d_ocp_qp_set_D(k, const_cast<double*>(D_k[k].data()), qp_in);
            d_ocp_qp_set_lg(k, const_cast<double*>(lg_k[k].data()), qp_in);
            d_ocp_qp_set_ug(k, const_cast<double*>(ug_k[k].data()), qp_in);
        }
    }

    // Terminal stage: state cost + box bound only, same as solve() --
    // including solve()'s identical idxbx_terminal concatenation when
    // non-empty, and the same optional Q_N_data terminal-cost override
    // (see that function's identical comments).
    d_ocp_qp_set_Q(N, Q_N_data ? Q_N_data : Q_data, qp_in);
    d_ocp_qp_set_q(N, zero_vec.data(), qp_in);
    const int n_terminal = static_cast<int>(params_.idxbx_terminal.size());
    if (n_terminal > 0) {
        Eigen::VectorXd lbxN(nbx_k + n_terminal), ubxN(nbx_k + n_terminal);
        if (nbx_k > 0) {
            lbxN.head(nbx_k) = Eigen::Map<Eigen::VectorXd>(lbx_data, nbx_k);
            ubxN.head(nbx_k) = Eigen::Map<Eigen::VectorXd>(ubx_data, nbx_k);
        }
        lbxN.tail(n_terminal) = terminal_zero_bound_;
        ubxN.tail(n_terminal) = terminal_zero_bound_;
        d_ocp_qp_set_lbx(N, lbxN.data(), qp_in);
        d_ocp_qp_set_ubx(N, ubxN.data(), qp_in);
    } else if (nbx_k > 0) {
        d_ocp_qp_set_lbx(N, lbx_data, qp_in);
        d_ocp_qp_set_ubx(N, ubx_data, qp_in);
    }

    // Initial-condition pin, same idiom as solve().
    d_ocp_qp_set_lbx(0, x0_current, qp_in);
    d_ocp_qp_set_ubx(0, x0_current, qp_in);

    // Solve, same warm-start-then-cold-retry-on-failure logic as solve().
    int qp_status = ocp_qp_solve(solver, qp_in, qp_out);
    if (qp_status != 0 && params_.qp_warm_start != 0) {
        int cold = 0;
        ocp_qp_xcond_solver_opts_set(config, opts, "warm_start", &cold);
        qp_status = ocp_qp_solve(solver, qp_in, qp_out);
        ocp_qp_xcond_solver_opts_set(config, opts, "warm_start", &params_.qp_warm_start);
    }
    if (qp_status == 0) {
        Eigen::VectorXd u0_sol(params_.nu);
        d_ocp_qp_sol_get_u(0, qp_out, u0_sol.data());
        last_tau_task = u0_sol;
        return true;
    } else {
        std::cerr << "QuadProbSolver: multi-stage QP solve failed with status " << qp_status << std::endl;
        return false;
    }
}

void QuadProbSolver::getU(int stage, double* u_out) const {
    if (stage < 0 || stage >= params_.N) {
        throw std::out_of_range("QuadProbSolver::getU: stage out of range [0, N)");
    }
    d_ocp_qp_sol_get_u(stage, qp_out, u_out);
}

void QuadProbSolver::getX(int stage, double* x_out) const {
    if (stage < 0 || stage > params_.N) {
        throw std::out_of_range("QuadProbSolver::getX: stage out of range [0, N]");
    }
    d_ocp_qp_sol_get_x(stage, qp_out, x_out);
}

int QuadProbSolver::nsAt(int stage) const {
    if (stage < 0 || stage > params_.N) return 0;
    const int nbx_k  = static_cast<int>(params_.idxbx_k.size());
    const int nsbx_k = (stage == 0) ? 0 : nbx_k;             // box rows soft at stages 1..N
    const int nsg_k  = (stage < params_.N) ? params_.ng : 0; // general rows soft at stages 0..N-1
    return nsbx_k + nsg_k;
}

void QuadProbSolver::getSlack(int stage, double* sl_out, double* su_out) const {
    if (stage < 0 || stage > params_.N) {
        throw std::out_of_range("QuadProbSolver::getSlack: stage out of range [0, N]");
    }
    d_ocp_qp_sol_get_sl(stage, qp_out, sl_out);
    d_ocp_qp_sol_get_su(stage, qp_out, su_out);
}

} // namespace quad_prob_solver
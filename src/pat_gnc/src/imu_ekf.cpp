#include "pat_gnc/navigation/imu_ekf.hpp"

#include <cmath>
#include <stdexcept>

namespace pat_gnc {

namespace {

//! d/dth R(th) = R(th) * J, with J the planar cross-product matrix.
inline Eigen::Matrix2d rot(double th) {
    const double c = std::cos(th), s = std::sin(th);
    Eigen::Matrix2d R;
    R << c, -s,
         s,  c;
    return R;
}

inline Eigen::Matrix2d skewJ() {
    Eigen::Matrix2d J;
    J << 0.0, -1.0,
         1.0,  0.0;
    return J;
}

}  // namespace

ImuEkf::ImuEkf(const Params& p) : p_(p) {
    if (p_.history_len < 2) {
        throw std::invalid_argument("ImuEkf: history_len must be at least 2");
    }
    // The only allocation the filter ever makes. Everything after this --
    // propagation, updates, replays -- reuses these slots.
    hist_.resize(p_.history_len);
}

ImuEkf::Step& ImuEkf::stepAt(std::size_t i) {
    return hist_[(hist_head_ + i) % hist_.size()];
}

void ImuEkf::pushStep(bool integrated) {
    std::size_t idx;
    if (hist_size_ < hist_.size()) {
        idx = (hist_head_ + hist_size_) % hist_.size();
        ++hist_size_;
    } else {
        // Full: the oldest step is overwritten, and a measurement older than
        // the new oldest is from then on too old to place.
        idx = hist_head_;
        hist_head_ = (hist_head_ + 1) % hist_.size();
    }
    Step& s = hist_[idx];
    s.t = t_;
    s.x = x_;
    s.P = P_;
    s.accel = last_accel_;
    s.gyro = last_gyro_;
    s.integrated = integrated;
    s.n_meas = 0;
}

void ImuEkf::initialise(double t, double px, double py, double yaw) {
    x_.setZero();
    x_(PX) = px;
    x_(PY) = py;
    x_(TH) = wrapAngle(yaw);

    P_.setZero();
    const auto var = [](double s) { return s * s; };
    P_(PX, PX) = P_(PY, PY) = var(p_.init_pos_std);
    P_(TH, TH) = var(p_.init_yaw_std);
    P_(VX, VX) = P_(VY, VY) = var(p_.init_vel_std);
    P_(BG, BG) = var(p_.init_gyro_bias_std);
    P_(BAX, BAX) = P_(BAY, BAY) = var(p_.init_accel_bias_std);

    t_ = t;
    initialised_ = true;
    // The history restarts here: nothing before initialisation is a state the
    // filter could return to.
    hist_head_ = 0;
    hist_size_ = 0;
    pushStep(false);
}

void ImuEkf::propagate(double t, const Eigen::Vector2d& accel, double gyro_z) {
    ++diag_.imu_samples;

    // Before initialisation there is no state to move; only keep the clock and
    // the sample, so the first interval after initialise() starts from a real
    // reading.
    if (!initialised_) {
        t_ = t;
        last_gyro_ = gyro_z;
        last_accel_ = accel;
        have_sample_ = true;
        return;
    }
    const double dt = t - t_;
    if (dt <= 0.0) {
        // A repeated or reordered sample. Integrating it would run time
        // backwards; dropping it costs one sample of information.
        ++diag_.imu_out_of_order;
        return;
    }
    // The interval runs between two samples. Without a previous one (the
    // filter was initialised from a pose before any IMU arrived) the current
    // sample stands in for both ends.
    const Eigen::Vector2d a0 = have_sample_ ? last_accel_ : accel;
    const double g0 = have_sample_ ? last_gyro_ : gyro_z;
    t_ = t;
    last_gyro_ = gyro_z;
    last_accel_ = accel;
    have_sample_ = true;
    bool integrated = true;
    if (dt > p_.max_imu_gap) {
        // Bridging a long gap from its two end samples extrapolates them far
        // past where they were valid. Skip the interval and let the covariance
        // be grown honestly by the next real ones.
        ++diag_.imu_gaps;
        integrated = false;
    } else {
        integrate(dt, a0, g0, accel, gyro_z);
    }
    pushStep(integrated);
}

void ImuEkf::integrate(double dt, const Eigen::Vector2d& a0, double g0,
                       const Eigen::Vector2d& a1, double g1) {
    // --- Nominal state: integration of the bias-corrected IMU ---------------
    // The IMU reads instantaneous values, so the interval's input is the mean
    // of its two end samples (trapezoidal), and the rotation is evaluated at
    // the interval's midpoint attitude. Both make the step second-order
    // accurate in dt; taking either end alone is first-order and leaves a
    // drift proportional to the rate of change of the input.
    const double w = 0.5 * (g0 + g1) - x_(BG);
    const Eigen::Vector2d a_b = 0.5 * (a0 + a1) - x_.segment<2>(BAX);
    const double th_mid = x_(TH) + 0.5 * w * dt;
    const Eigen::Matrix2d R = rot(th_mid);
    const Eigen::Vector2d a_w = R * a_b;

    x_.segment<2>(PX) += x_.segment<2>(VX) * dt + 0.5 * a_w * dt * dt;
    x_.segment<2>(VX) += a_w * dt;
    x_(TH) = wrapAngle(x_(TH) + w * dt);

    // --- Error dynamics: d(dx)/dt = A dx + G n ------------------------------
    //   dp' = dv
    //   dv' = (dR/dth a_b) dth - R db_a - R n_a
    //   dth' = -db_g - n_g
    //   db_g' = n_wg,  db_a' = n_wa
    // (Sola 2017, sec. 5.3, the error-state kinematics, restricted to the
    // plane: the skew-symmetric term collapses to J and the error angle is a
    // scalar.)
    Mat A = Mat::Zero();
    A.block<2, 2>(PX, VX) = Eigen::Matrix2d::Identity();
    A.block<2, 1>(VX, TH) = R * skewJ() * a_b;
    A.block<2, 2>(VX, BAX) = -R;
    A(TH, BG) = -1.0;

    // Transition to second order in dt. The A^2 term carries the coupling of
    // attitude and accelerometer-bias errors into POSITION within one step
    // (A^2 maps them through velocity), which a first-order I + A dt drops.
    const Mat Adt = A * dt;
    const Mat F = Mat::Identity() + Adt + 0.5 * Adt * Adt;

    // --- Process noise, discretised for sampled white noise -----------------
    // Each sample's noise enters velocity as n * dt and position as
    // n * dt^2 / 2. Averaging adjacent samples splits each one across two
    // intervals, but the noise still accumulates by one sample per interval
    // over any run longer than a step, so the per-interval variance is
    // sigma^2 dt^2 on velocity and sigma^2 dt^4 / 4 on position, correlated by
    // sigma^2 dt^3 / 2. Rotation leaves an isotropic covariance unchanged, so
    // R drops out. The bias walks are continuous densities and accumulate as
    // sigma^2 dt.
    Mat Q = Mat::Zero();
    const double sa2 = p_.accel_noise_std * p_.accel_noise_std;
    const double sg2 = p_.gyro_noise_std * p_.gyro_noise_std;
    const double dt2 = dt * dt;
    for (int i = 0; i < 2; ++i) {
        Q(PX + i, PX + i) = sa2 * dt2 * dt2 / 4.0;
        Q(PX + i, VX + i) = Q(VX + i, PX + i) = sa2 * dt2 * dt / 2.0;
        Q(VX + i, VX + i) = sa2 * dt2;
        Q(BAX + i, BAX + i) = p_.accel_bias_walk * p_.accel_bias_walk * dt;
    }
    Q(TH, TH) = sg2 * dt2;
    Q(BG, BG) = p_.gyro_bias_walk * p_.gyro_bias_walk * dt;

    P_ = F * P_ * F.transpose() + Q;
    P_ = 0.5 * (P_ + P_.transpose());
}

double ImuEkf::yawRateVariance() const noexcept {
    return p_.gyro_noise_std * p_.gyro_noise_std + P_(BG, BG);
}

double ImuEkf::updateYaw(double t_meas, double yaw) {
    Meas m;
    m.kind = Kind::Yaw;
    m.a = yaw;
    return insert(t_meas, m);
}

double ImuEkf::updatePosition(double t_meas, const Eigen::Vector2d& p) {
    Meas m;
    m.kind = Kind::Pos;
    m.a = p.x();
    m.b = p.y();
    return insert(t_meas, m);
}

double ImuEkf::updateVelocity(double t_meas, const Eigen::Vector2d& v) {
    Meas m;
    m.kind = Kind::Vel;
    m.a = v.x();
    m.b = v.y();
    return insert(t_meas, m);
}

double ImuEkf::insert(double t_meas, const Meas& m) {
    if (!initialised_) {
        ++diag_.not_ready;
        return -1.0;
    }
    // The kept step a measurement belongs to: the one NEAREST its time.
    // Searched from the newest end, since almost every measurement belongs
    // there or a few steps back. Nearest rather than at-or-before for two
    // reasons. It halves the worst timing error, to half an IMU interval. And
    // a stamp computed as (message time - latency) routinely lands a rounding
    // error below the step it was taken at; at-or-before then applies it a
    // whole interval early, which while turning is a systematic yaw error
    // several times the star tracker's own noise.
    std::size_t k = hist_size_;
    for (std::size_t i = hist_size_; i-- > 0;) {
        if (stepAt(i).t <= t_meas) {
            k = i;
            break;
        }
    }
    if (k == hist_size_) {
        // Before every kept step: nearest is the oldest, if it is within half
        // an interval of it; otherwise there is no state to apply it to.
        const bool near_oldest = hist_size_ >= 2 &&
            (stepAt(0).t - t_meas) < 0.5 * (stepAt(1).t - stepAt(0).t);
        if (!near_oldest) {
            ++diag_.too_old;
            return -1.0;
        }
        k = 0;
    } else if (k + 1 < hist_size_ &&
               (stepAt(k + 1).t - t_meas) < (t_meas - stepAt(k).t)) {
        ++k;
    }

    Step& home = stepAt(k);
    const bool stored = home.n_meas < kMeasPerStep;
    if (stored) {
        home.meas[home.n_meas++] = m;
    } else {
        ++diag_.slot_overflow;
    }

    // The common case: it belongs to the newest step, so the current state IS
    // that step's state with its earlier measurements applied, and the new one
    // just goes on top.
    if (k + 1 == hist_size_) return apply(m, true);

    // A delayed measurement. Return to the step it belongs to and recompute
    // forward from there. Every measurement kept at that step is re-applied,
    // in arrival order, with the new one -- the last in the list -- counted.
    ++diag_.replays;
    const std::size_t replay_steps = hist_size_ - 1 - k;
    if (replay_steps > diag_.max_replay_steps) diag_.max_replay_steps = replay_steps;

    x_ = home.x;
    P_ = home.P;
    double nis = -1.0;
    for (int j = 0; j < home.n_meas; ++j) {
        const bool is_new = stored && j == home.n_meas - 1;
        const double r = apply(home.meas[j], is_new);
        if (is_new) nis = r;
    }
    if (!stored) nis = apply(m, true);   // applied, but not kept for replay

    for (std::size_t i = k + 1; i < hist_size_; ++i) {
        const Step& prev = stepAt(i - 1);
        Step& s = stepAt(i);
        if (s.integrated) integrate(s.t - prev.t, prev.accel, prev.gyro, s.accel, s.gyro);
        // Overwrite the kept state with the corrected one, so a later delayed
        // measurement builds on this correction instead of undoing it.
        s.x = x_;
        s.P = P_;
        for (int j = 0; j < s.n_meas; ++j) apply(s.meas[j], false);
    }
    return nis;
}

double ImuEkf::apply(const Meas& m, bool first) {
    bool accepted = false;
    double nis = 0.0;
    switch (m.kind) {
    case Kind::Yaw: {
        Eigen::Matrix<double, 1, 1> y;
        // Wrapped, so a measurement of +3.13 against an estimate of -3.13 is a
        // 0.02 rad innovation and not a 6.26 rad one.
        y(0) = wrapAngle(m.a - x_(TH));
        Eigen::Matrix<double, 1, N> H = Eigen::Matrix<double, 1, N>::Zero();
        H(0, TH) = 1.0;
        Eigen::Matrix<double, 1, 1> R;
        R(0, 0) = p_.yaw_meas_std * p_.yaw_meas_std;
        nis = correct<1>(y, H, R, p_.gate_yaw, accepted);
        if (first) {
            ++(accepted ? diag_.yaw_updates : diag_.yaw_rejected);
            diag_.last_nis_yaw = nis;
        }
        break;
    }
    case Kind::Pos: {
        const Eigen::Vector2d y = Eigen::Vector2d(m.a, m.b) - x_.segment<2>(PX);
        Eigen::Matrix<double, 2, N> H = Eigen::Matrix<double, 2, N>::Zero();
        H.block<2, 2>(0, PX) = Eigen::Matrix2d::Identity();
        const Eigen::Matrix2d R =
            Eigen::Matrix2d::Identity() * (p_.pos_meas_std * p_.pos_meas_std);
        nis = correct<2>(y, H, R, p_.gate_pos, accepted);
        if (first) {
            ++(accepted ? diag_.pos_updates : diag_.pos_rejected);
            diag_.last_nis_pos = nis;
        }
        break;
    }
    case Kind::Vel: {
        const Eigen::Vector2d y = Eigen::Vector2d(m.a, m.b) - x_.segment<2>(VX);
        Eigen::Matrix<double, 2, N> H = Eigen::Matrix<double, 2, N>::Zero();
        H.block<2, 2>(0, VX) = Eigen::Matrix2d::Identity();
        const Eigen::Matrix2d R =
            Eigen::Matrix2d::Identity() * (p_.vel_meas_std * p_.vel_meas_std);
        nis = correct<2>(y, H, R, p_.gate_vel, accepted);
        if (first) ++(accepted ? diag_.vel_updates : diag_.vel_rejected);
        break;
    }
    }
    return nis;
}

template <int M>
double ImuEkf::correct(const Eigen::Matrix<double, M, 1>& y,
                       const Eigen::Matrix<double, M, N>& H,
                       const Eigen::Matrix<double, M, M>& R, double gate,
                       bool& accepted) {
    using MatMN = Eigen::Matrix<double, M, N>;
    const Eigen::Matrix<double, M, M> S = H * P_ * H.transpose() + R;
    const Eigen::LDLT<Eigen::Matrix<double, M, M>> S_ldlt(S);
    const double nis = y.dot(S_ldlt.solve(y));

    // Gate before touching anything: a rejected measurement leaves the state
    // and covariance exactly as they were.
    if (gate > 0.0 && nis > gate) {
        accepted = false;
        return nis;
    }
    accepted = true;

    // K = P H^T S^-1, formed as (S^-1 H P)^T since S and P are symmetric:
    // a solve rather than an explicit inverse.
    const MatMN HP = H * P_;
    const Eigen::Matrix<double, N, M> K = S_ldlt.solve(HP).transpose();

    // Joseph form, Bar-Shalom, Li & Kirubarajan, "Estimation with Applications
    // to Tracking and Navigation" (2001), ch. 5: stays symmetric positive
    // semi-definite for ANY gain, where (I - KH)P loses that under rounding --
    // which matters for a filter that runs for hours.
    const Mat IKH = Mat::Identity() - K * H;
    P_ = IKH * P_ * IKH.transpose() + K * R * K.transpose();
    P_ = 0.5 * (P_ + P_.transpose());

    // Inject the error into the nominal state; the reset is the identity in
    // the plane (see the header), so P needs no reset transform.
    x_ += K * y;
    x_(TH) = wrapAngle(x_(TH));
    return nis;
}

double ImuEkf::wrapAngle(double a) noexcept { return std::remainder(a, 2.0 * M_PI); }

}  // namespace pat_gnc

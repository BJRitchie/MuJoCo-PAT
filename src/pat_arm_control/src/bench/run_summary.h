#pragma once
/*! One bench run, recorded tick by tick, and the scalar figures of merit that
 *  summarise it.
 *
 *  The bench fills a RunSeries as it runs (the scored arm's applied torque and
 *  joint velocity, where its end effector actually was against where it was
 *  asked to be, how the base moved, and how long each tick took). summarise()
 *  turns that into the named numbers a comparison needs, and the two writers
 *  put them in tidy CSV: one row per run for the summary, one row per tick for
 *  the series. Nothing here knows about the controller or the plant, so it is
 *  tested against synthetic series whose metrics are known.
 *
 *  Units in the series are SI (metres, radians, seconds, N*m, rad/s); the
 *  summary's names carry the unit they are reported in (_mm, _mrad, _us).
 */
#include <string>
#include <utility>
#include <vector>

namespace pat_arm_control {
namespace bench {

struct RunSeries {
    std::vector<double> t;                       //!< [s] simulation time at each tick
    std::vector<double> rx, ry;                  //!< [m] what the EE was asked to reach
    std::vector<double> rvx, rvy;                //!< [m/s] reference velocity (0 for a setpoint)
    std::vector<double> ex, ey;                  //!< [m] where the EE actually was (plant truth)
    std::vector<double> eth;                     //!< [rad] EE yaw error against its reference
    std::vector<std::vector<double>> tau;        //!< [N*m] applied torque, [owned joint][tick]
    std::vector<std::vector<double>> qd;         //!< [rad/s] owned joint velocity, same layout
    std::vector<double> base_x, base_y;          //!< [m] chaser position
    std::vector<double> base_yaw;                //!< [rad] chaser yaw, wrapped as the plant reports it
    std::vector<double> base_wz;                 //!< [rad/s] chaser yaw rate
    std::vector<double> tick_us;                 //!< [us] wall time of the control computation
    std::vector<int> mode;                       //!< IArmController::ControlMode at each tick
    std::vector<double> tau_limit;               //!< [N*m] per owned joint, for the rail fraction
    std::vector<std::vector<double>> q;          //!< [rad] owned joint position (truth), [joint][tick]
    //! [rad] each owned joint's CONFIGURED (software) position limits, the box the
    //! constrained NMPC enforces; the plant's own hard stops lie outside it.
    std::vector<double> q_min, q_max;
    std::vector<int> ncon;                       //!< plant constraint count per tick (a hard stop or contact)

    //! The base state the controller was given minus the plant's truth, when a
    //! measurement model or an estimator was in the loop (else left empty):
    //! x, y [m], yaw [rad], vx, vy [m/s], yaw rate [rad/s], one entry per tick.
    std::vector<double> est_x, est_y, est_yaw, est_vx, est_vy, est_wz;

    //! Planar momentum of the whole chaser (mom_*) and of the bus alone (bus_*),
    //! both about the chaser's centre of mass: px, py [kg*m/s], lz [kg*m^2/s].
    //! Written when filled, one entry per tick.
    std::vector<double> mom_px, mom_py, mom_lz, bus_px, bus_py, bus_lz;
    //! [J] kinetic energy of the whole chaser, written when filled.
    std::vector<double> ke;

    std::size_t size() const { return t.size(); }
    //! Size the per-joint tracks for `n_owned` joints.
    void setJoints(std::size_t n_owned) {
        tau.assign(n_owned, {});
        qd.assign(n_owned, {});
        q.assign(n_owned, {});
    }
};

//! Everything about a run that the series does not hold: what it was, and the
//! counters the controller kept.
struct RunContext {
    std::string scenario;     //!< e.g. "circle", "step", "osc"
    std::string law;          //!< nmpc | ompc | smc
    std::string tag;          //!< free-form configuration label, e.g. "sqp"
    int seed = 0;
    //! Further identifying columns, written after `seed` in this order: the
    //! settings that distinguish one cell of a study from another, so a plot
    //! script can group on them without parsing the tag. Every run of one
    //! binary writes the same names.
    std::vector<std::pair<std::string, std::string>> labels;
    double window_start_s = 0.0;   //!< scoring window for the tracking and effort metrics
    //! A setpoint step: the EE was told at `step_t` to move (dx, dy) metres from
    //! where it started. Enables the step-response columns (NaN otherwise).
    bool has_step = false;
    double step_dx = 0.0, step_dy = 0.0, step_t = 0.0;
    long safe_ticks = 0;
    long tier1_entries = 0;
    long ref_outside = 0;
    long ff_saturated = 0;
    long ff_slew_limited = 0;
    double qp_iter_mean = 0.0;
    double qp_iter_max = 0.0;
    bool has_solver = true;
};

using NamedValues = std::vector<std::pair<std::string, double>>;

/*! The scalar metrics of a run, in a fixed order. Tracking and effort are taken
 *  over the scoring window; the base-attitude and timing figures are also given
 *  over the whole run (suffix _total), since the approach is part of what the
 *  base felt. Missing values are NaN. */
NamedValues summarise(const RunSeries& s, const RunContext& ctx);

//! The full per-tick record, one row per tick, header first.
void writeSeriesCsv(const std::string& path, const RunSeries& s);

/*! Append one row (scenario, law, tag, seed, then every metric) to a tidy CSV,
 *  writing the header first if the file is new or empty. Appending is what
 *  lets a driver collect a whole study into one file. */
void appendSummaryCsv(const std::string& path, const RunContext& ctx,
                      const NamedValues& values);

}  // namespace bench
}  // namespace pat_arm_control

/*! series_summary: score a recorded run the way the bench scores its own.
 *
 *  Reads a per-tick series CSV (the bench's --series output, or the live
 *  recorder's) and appends one summary row, through the same summarise() the
 *  bench uses, so a live figure of merit and a bench one are one definition.
 *  ROS-free and plant-free: everything about the run that the CSV does not hold
 *  is on the command line.
 *
 *    series_summary --series run.csv --summary-csv summary.csv \
 *        --scenario circle --law nmpc --tag sqp --seed 3 \
 *        --window-start 5 --tau-limit 0.25 --q-min -1,-2,-3 --q-max 1,2,3 \
 *        --label state_source=ekf --label traj_speed=1
 */
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "run_summary.h"
#include "series_io.h"

namespace b = pat_arm_control::bench;

namespace {

std::vector<double> numbers(const std::string& text) {
    std::vector<double> out;
    std::istringstream ss(text);
    std::string tok;
    while (std::getline(ss, tok, ',')) out.push_back(std::stod(tok));
    return out;
}

void usage() {
    std::cerr <<
        "usage: series_summary --series CSV --summary-csv CSV --scenario NAME --law LAW\n"
        "         [--tag T] [--seed N] [--window-start S] [--tau-limit X]\n"
        "         [--q-min a,b,..] [--q-max a,b,..] [--label KEY=VALUE]...\n"
        "         [--tier1 N] [--ref-outside N] [--ff-saturated N] [--ff-slew-limited N]\n"
        "         [--qp-iter-mean X] [--qp-iter-max X] [--no-solver]\n"
        "  --tau-limit applies to every owned joint. Safe-mode ticks are counted\n"
        "  from the series' mode column (a ControlMode of 2 or more).\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string series_path, summary_path;
    b::RunContext ctx;
    double tau_limit = 0.0;
    std::vector<double> q_min, q_max;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            const auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
                return argv[++i];
            };
            if (a == "--series") series_path = next();
            else if (a == "--summary-csv") summary_path = next();
            else if (a == "--scenario") ctx.scenario = next();
            else if (a == "--law") ctx.law = next();
            else if (a == "--tag") ctx.tag = next();
            else if (a == "--seed") ctx.seed = std::stoi(next());
            else if (a == "--window-start") ctx.window_start_s = std::stod(next());
            else if (a == "--tau-limit") tau_limit = std::stod(next());
            else if (a == "--q-min") q_min = numbers(next());
            else if (a == "--q-max") q_max = numbers(next());
            else if (a == "--tier1") ctx.tier1_entries = std::stol(next());
            else if (a == "--ref-outside") ctx.ref_outside = std::stol(next());
            else if (a == "--ff-saturated") ctx.ff_saturated = std::stol(next());
            else if (a == "--ff-slew-limited") ctx.ff_slew_limited = std::stol(next());
            else if (a == "--qp-iter-mean") ctx.qp_iter_mean = std::stod(next());
            else if (a == "--qp-iter-max") ctx.qp_iter_max = std::stod(next());
            else if (a == "--no-solver") ctx.has_solver = false;
            else if (a == "--label") {
                const std::string kv = next();
                const auto eq = kv.find('=');
                if (eq == std::string::npos) throw std::runtime_error("--label wants KEY=VALUE");
                ctx.labels.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
            } else if (a == "-h" || a == "--help") {
                usage();
                return 0;
            } else {
                throw std::runtime_error("unknown argument " + a);
            }
        }
        if (series_path.empty() || summary_path.empty() || ctx.scenario.empty() ||
            ctx.law.empty()) {
            usage();
            return 2;
        }
        if (q_min.size() != q_max.size()) {
            throw std::runtime_error("--q-min and --q-max must list the same joints");
        }

        b::RunSeries s = b::readSeriesCsv(series_path);
        s.tau_limit.assign(s.tau.size(), tau_limit);
        s.q_min = q_min;
        s.q_max = q_max;
        for (const int m : s.mode) {
            if (m >= 2) ++ctx.safe_ticks;
        }
        b::appendSummaryCsv(summary_path, ctx, b::summarise(s, ctx));
    } catch (const std::exception& e) {
        std::cerr << "series_summary: " << e.what() << '\n';
        return 1;
    }
    return 0;
}

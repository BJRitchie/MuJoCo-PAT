#include "series_io.h"

#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace pat_arm_control {
namespace bench {

namespace {

std::vector<std::string> splitCsv(std::string line) {
    std::vector<std::string> out;
    std::string cell;
    // A file that went through a CRLF writer would otherwise carry the \r into
    // its last column's name.
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::istringstream ss(line);
    while (std::getline(ss, cell, ',')) out.push_back(cell);
    // getline drops a trailing empty field; a line ending in a comma has one.
    if (!line.empty() && line.back() == ',') out.emplace_back();
    return out;
}

//! Index of every column called `<prefix><number>`, in number order.
std::vector<std::size_t> numbered(const std::map<std::string, std::size_t>& col,
                                  const std::string& prefix) {
    std::vector<std::size_t> out;
    for (std::size_t j = 0;; ++j) {
        const auto it = col.find(prefix + std::to_string(j));
        if (it == col.end()) break;
        out.push_back(it->second);
    }
    return out;
}

}  // namespace

RunSeries readSeriesCsv(const std::string& path) {
    std::ifstream is(path);
    if (!is) throw std::runtime_error("cannot read " + path);

    std::string line;
    if (!std::getline(is, line)) throw std::runtime_error(path + " is empty");
    const std::vector<std::string> head = splitCsv(line);
    std::map<std::string, std::size_t> col;
    for (std::size_t i = 0; i < head.size(); ++i) col[head[i]] = i;

    const auto need = [&](const char* name) {
        const auto it = col.find(name);
        if (it == col.end()) {
            throw std::runtime_error(path + ": no column '" + name + "'");
        }
        return it->second;
    };
    const auto have = [&](const char* name) { return col.count(name) > 0; };

    const std::size_t c_t = need("t_s"), c_rx = need("ref_x"), c_ry = need("ref_y"),
                      c_rvx = need("ref_vx"), c_rvy = need("ref_vy"), c_ex = need("ee_x"),
                      c_ey = need("ee_y"), c_eth = need("eth_rad"), c_bx = need("base_x"),
                      c_by = need("base_y"), c_byaw = need("base_yaw"),
                      c_bwz = need("base_wz"), c_tick = need("tick_us"), c_mode = need("mode");
    const std::vector<std::size_t> c_tau = numbered(col, "tau");
    const std::vector<std::size_t> c_qd = numbered(col, "qd");
    const std::vector<std::size_t> c_q = numbered(col, "q");
    const bool have_ncon = have("ncon");
    const bool have_est = have("est_x");

    RunSeries s;
    s.setJoints(c_tau.size());
    if (c_qd.size() != c_tau.size()) {
        throw std::runtime_error(path + ": torque and joint-velocity columns differ in number");
    }
    // Joint positions are optional; without one per owned joint they are left out.
    const bool use_q = !c_q.empty() && c_q.size() == c_tau.size();
    if (!use_q) s.q.clear();

    std::size_t row = 1;
    while (std::getline(is, line)) {
        ++row;
        if (line.empty()) continue;
        const std::vector<std::string> f = splitCsv(line);
        if (f.size() != head.size()) {
            throw std::runtime_error(path + ": row " + std::to_string(row) + " has " +
                                     std::to_string(f.size()) + " fields, expected " +
                                     std::to_string(head.size()));
        }
        const auto num = [&f](std::size_t i) { return std::strtod(f[i].c_str(), nullptr); };
        s.t.push_back(num(c_t));
        s.rx.push_back(num(c_rx));   s.ry.push_back(num(c_ry));
        s.rvx.push_back(num(c_rvx)); s.rvy.push_back(num(c_rvy));
        s.ex.push_back(num(c_ex));   s.ey.push_back(num(c_ey));
        s.eth.push_back(num(c_eth));
        for (std::size_t j = 0; j < c_tau.size(); ++j) {
            s.tau[j].push_back(num(c_tau[j]));
            s.qd[j].push_back(num(c_qd[j]));
        }
        if (use_q) {
            for (std::size_t j = 0; j < c_q.size(); ++j) s.q[j].push_back(num(c_q[j]));
        }
        if (have_ncon) s.ncon.push_back(static_cast<int>(num(col.at("ncon"))));
        s.base_x.push_back(num(c_bx));   s.base_y.push_back(num(c_by));
        s.base_yaw.push_back(num(c_byaw)); s.base_wz.push_back(num(c_bwz));
        s.tick_us.push_back(num(c_tick));
        s.mode.push_back(static_cast<int>(num(c_mode)));
        if (have_est) {
            s.est_x.push_back(num(col.at("est_x")));     s.est_y.push_back(num(col.at("est_y")));
            s.est_yaw.push_back(num(col.at("est_yaw")));
            s.est_vx.push_back(num(col.at("est_vx")));   s.est_vy.push_back(num(col.at("est_vy")));
            s.est_wz.push_back(num(col.at("est_wz")));
        }
    }
    if (s.size() == 0) throw std::runtime_error(path + " has no rows");
    return s;
}

}  // namespace bench
}  // namespace pat_arm_control

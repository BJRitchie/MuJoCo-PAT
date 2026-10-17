#pragma once
/*! The navigator factory, shared by every pat_gnc node that runs a navigator.
 *
 *  Node-side (it reads ROS parameters), so it lives here and not in the
 *  zero-ROS library. One factory means the 'navigator' choice and every
 *  imu_ekf.* parameter name mean the same thing in gnc_node and nav_node.
 *  Add a navigator as one more else-if; nothing else changes.
 */
#include <memory>
#include <stdexcept>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "pat_gnc/interfaces/i_navigator.hpp"
#include "pat_gnc/navigation/direct_navigator.hpp"
#include "pat_gnc/navigation/ekf_navigator.hpp"
#include "pat_gnc/navigation/imu_ekf_navigator.hpp"

namespace pat_gnc {

//! imu_ekf.* parameters -> the navigator's configuration. Defaults are
//! ImuEkf::Params' own, so an unset parameter changes nothing.
inline ImuEkfNavigator::Config declareImuEkfConfig(rclcpp::Node& node) {
    ImuEkfNavigator::Config c;
    ImuEkf::Params& e = c.ekf;
    const auto d = [&node](const std::string& name, double def) {
        return node.declare_parameter<double>("imu_ekf." + name, def);
    };
    e.gyro_noise_std      = d("gyro_noise_std", e.gyro_noise_std);
    e.accel_noise_std     = d("accel_noise_std", e.accel_noise_std);
    e.gyro_bias_walk      = d("gyro_bias_walk", e.gyro_bias_walk);
    e.accel_bias_walk     = d("accel_bias_walk", e.accel_bias_walk);
    e.yaw_meas_std        = d("yaw_meas_std", e.yaw_meas_std);
    e.pos_meas_std        = d("pos_meas_std", e.pos_meas_std);
    e.vel_meas_std        = d("vel_meas_std", e.vel_meas_std);
    e.init_pos_std        = d("init_pos_std", e.init_pos_std);
    e.init_yaw_std        = d("init_yaw_std", e.init_yaw_std);
    e.init_vel_std        = d("init_vel_std", e.init_vel_std);
    e.init_gyro_bias_std  = d("init_gyro_bias_std", e.init_gyro_bias_std);
    e.init_accel_bias_std = d("init_accel_bias_std", e.init_accel_bias_std);
    e.max_imu_gap         = d("max_imu_gap", e.max_imu_gap);
    e.gate_yaw            = d("gate_yaw", e.gate_yaw);
    e.gate_pos            = d("gate_pos", e.gate_pos);
    e.gate_vel            = d("gate_vel", e.gate_vel);
    const int hist = node.declare_parameter<int>(
        "imu_ekf.history_len", static_cast<int>(e.history_len));
    if (hist < 2) throw std::runtime_error("imu_ekf.history_len must be at least 2");
    e.history_len = static_cast<std::size_t>(hist);

    c.latency_yaw = d("latency_yaw", c.latency_yaw);
    c.latency_pos = d("latency_pos", c.latency_pos);
    c.latency_vel = d("latency_vel", c.latency_vel);
    c.use_twist   = node.declare_parameter<bool>("imu_ekf.use_twist", c.use_twist);
    if (c.latency_yaw < 0.0 || c.latency_pos < 0.0 || c.latency_vel < 0.0) {
        throw std::runtime_error("imu_ekf.latency_* must be non-negative");
    }
    return c;
}

/*! Build the navigator named by the 'navigator' parameter.
 *
 *  @param nav_hz  the fixed predict rate of the constant-velocity "ekf"
 *                 navigator; the others take their timing from the stamps.
 */
inline std::unique_ptr<INavigator> makeNavigator(rclcpp::Node& node, double nav_hz,
                                                 std::string& type_out) {
    type_out = node.declare_parameter<std::string>("navigator", "direct");
    if (type_out == "direct") {
        return std::make_unique<DirectNavigator>();
    }
    if (type_out == "ekf") {
        PlanarEKF::Params ep;
        ep.dt = 1.0 / nav_hz;
        return std::make_unique<EkfNavigator>(ep);
    }
    if (type_out == "imu_ekf") {
        return std::make_unique<ImuEkfNavigator>(declareImuEkfConfig(node));
    }
    RCLCPP_WARN(node.get_logger(), "Unknown navigator '%s', using 'direct'",
                type_out.c_str());
    type_out = "direct";
    return std::make_unique<DirectNavigator>();
}

}  // namespace pat_gnc

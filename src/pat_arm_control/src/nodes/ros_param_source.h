#pragma once
/*! ParamSource over a node's ROS parameters, for the shared config loaders.
 *
 *  Each key is declared on first read with the loader's default, so `ros2
 *  param list` shows every parameter the controller understands, at the value
 *  it is actually using. A key that is already declared is read back instead of
 *  declared again (which rclcpp refuses), so the node and a loader can both
 *  read one key.
 */
#include <cstdint>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "pat_arm_control/common/param_source.h"

class RosParamSource final : public pat_arm_control::ParamSource {
public:
    explicit RosParamSource(rclcpp::Node& node) : node_(node) {}

    double getDouble(const std::string& key, double def) override { return get(key, def); }
    int getInt(const std::string& key, int def) override { return get(key, def); }
    bool getBool(const std::string& key, bool def) override { return get(key, def); }
    std::string getString(const std::string& key, const std::string& def) override {
        return get(key, def);
    }
    std::vector<double> getDoubles(const std::string& key,
                                   const std::vector<double>& def) override {
        return get(key, def);
    }
    std::vector<std::string> getStrings(const std::string& key,
                                        const std::vector<std::string>& def) override {
        return get(key, def);
    }

private:
    template <typename T>
    T get(const std::string& key, const T& def) {
        if (node_.has_parameter(key)) return node_.get_parameter(key).get_value<T>();
        return node_.declare_parameter<T>(key, def);
    }

    rclcpp::Node& node_;
};

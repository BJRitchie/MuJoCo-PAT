#pragma once
/*! Where configuration comes from, without saying how.
 *
 *  The node reads its parameters through ROS (declare_parameter); the benches
 *  read the same nmpc.yaml with yaml-cpp. Both go through this interface, so the
 *  loaders that turn keys into controller parameters (controller_config.h,
 *  control/nmpc/nmpc_config.h) are written once and are the only place a key
 *  name appears. A key the node and a bench spelled differently, or mapped onto
 *  different fields, can then not exist.
 *
 *  Keys are dotted ROS parameter names ("nmpc.Ts", "weights.Q_pos"). Each getter
 *  returns the configured value, or `def` when the key is absent. Reading a key
 *  twice returns the same value both times. Zero ROS.
 */
#include <string>
#include <vector>

namespace pat_arm_control {

class ParamSource {
public:
    virtual ~ParamSource() = default;

    virtual double getDouble(const std::string& key, double def) = 0;
    virtual int getInt(const std::string& key, int def) = 0;
    virtual bool getBool(const std::string& key, bool def) = 0;
    virtual std::string getString(const std::string& key, const std::string& def) = 0;
    virtual std::vector<double> getDoubles(const std::string& key,
                                           const std::vector<double>& def) = 0;
    virtual std::vector<std::string> getStrings(const std::string& key,
                                                const std::vector<std::string>& def) = 0;
};

}  // namespace pat_arm_control

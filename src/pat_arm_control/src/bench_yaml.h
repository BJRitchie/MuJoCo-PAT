#pragma once
/*! nmpc.yaml -> NMPCParams, for the ROS-free benchmarks.
 *
 *  Shared by nmpc_bench and contact_bench. Shared rather than copied because
 *  this is already a second mapping of keys the node reads through
 *  declare_parameter -- a third would be a third thing to drift, and the
 *  unknown-key guard below only works if there is exactly one list of what a
 *  bench understands.
 *
 *  Reads the real config file rather than carrying its own copy, so a bench
 *  measures the configuration that actually gets deployed.
 */
#include <map>
#include <set>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "pat_arm_control/control/arm_nmpc.h"

namespace pat_arm_control {
namespace bench {

using Flat = std::map<std::string, YAML::Node>;

//! Nested YAML maps -> the dotted keys the ROS parameter names use.
void flatten(const YAML::Node& node, const std::string& prefix, Flat& out);

double asDouble(const Flat& f, const std::string& k, double def,
                std::set<std::string>& used);
int    asInt(const Flat& f, const std::string& k, int def,
             std::set<std::string>& used);
bool   asBool(const Flat& f, const std::string& k, bool def,
              std::set<std::string>& used);
std::vector<double> asVec(const Flat& f, const std::string& k,
                          std::vector<double> def, std::set<std::string>& used);
std::vector<std::string> asStrVec(const Flat& f, const std::string& k,
                                  std::set<std::string>& used);

struct Loaded {
    NMPCParams  params;
    std::string mjcf;
    std::string ee_site;

    //! ALL model joints in MJCF body-tree order. Both benches need it: one to
    //! size the torque-limit array, the other to map named joints onto the
    //! scene's index-named DOF rows.
    std::vector<std::string> model_joints;

    //! Observer configuration. Mapped here, rather than left to each bench, so
    //! the two agree on the key names and neither can drift from nmpc.yaml.
    //! nmpc_bench ignores these; contact_bench runs the observer.
    std::string observer_type   = "none";
    double      observer_hz     = 100.0;
    double      observer_gain   = 20.0;
    double      observer_gain_base = -1.0;
    double      observer_mdot_clamp = 0.0;
    double      observer_estimate_clamp = 0.0;

    //! The node's own frame correction, applied so a setpoint means the same
    //! thing to a bench as it does to the running stack.
    double base_anchor_x = 0.0;
    double base_anchor_y = 0.0;
};

/*! Merge the "/**" block with pat_arm_nmpc_<arm>'s and map it onto NMPCParams.
 *
 *  @param not_ours        keys this bench knowingly ignores; anything neither
 *                         mapped nor listed is a hard error, so adding a knob
 *                         to nmpc.yaml without teaching it to the bench says so
 *                         instead of quietly measuring something else
 *  @param default_model   used when neither `mjcf_override` nor the file's
 *                         mjcf_path gives one
 *  @param tool            named in the error message
 */
Loaded loadConfig(const std::string& yaml_path, const std::string& arm,
                  const std::string& mjcf_override,
                  const std::string& default_model,
                  const std::set<std::string>& not_ours,
                  const std::string& tool);

}  // namespace bench
}  // namespace pat_arm_control

#pragma once
/*! nmpc.yaml -> the controller's configuration, for the ROS-free benchmarks.
 *
 *  Shared by nmpc_bench and contact_bench. The key-to-field mapping itself is
 *  not here: it is the library's (controller_config.h, nmpc_config.h), the same
 *  code the node runs through ROS parameters. This file only supplies the
 *  values -- a ParamSource over the YAML file -- and the guard that refuses a
 *  key nothing mapped.
 *
 *  Reads the real config file rather than carrying its own copy, so a bench
 *  measures the configuration that actually gets deployed.
 */
#include <map>
#include <set>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "pat_arm_control/common/param_source.h"
#include "pat_arm_control/control/controller_config.h"
#include "pat_arm_control/control/nmpc/arm_nmpc.h"

namespace pat_arm_control {
namespace bench {

using Flat = std::map<std::string, YAML::Node>;

//! Nested YAML maps -> the dotted keys the ROS parameter names use.
void flatten(const YAML::Node& node, const std::string& prefix, Flat& out);

/*! A ParamSource over flattened YAML that records every key it is asked for,
 *  so the caller can tell which keys in the file nothing read. */
class YamlParamSource final : public ParamSource {
public:
    explicit YamlParamSource(const Flat& f) : f_(f) {}

    double getDouble(const std::string& key, double def) override;
    int getInt(const std::string& key, int def) override;
    bool getBool(const std::string& key, bool def) override;
    std::string getString(const std::string& key, const std::string& def) override;
    std::vector<double> getDoubles(const std::string& key,
                                   const std::vector<double>& def) override;
    std::vector<std::string> getStrings(const std::string& key,
                                        const std::vector<std::string>& def) override;

    //! Every key read so far, present in the file or not.
    const std::set<std::string>& used() const noexcept { return used_; }

private:
    const YAML::Node* find(const std::string& key);

    const Flat& f_;
    std::set<std::string> used_;
};

struct Loaded {
    NMPCParams     params;
    std::string    mjcf;
    //! Both benches need the model joints: one to size the torque-limit array,
    //! the other to map named joints onto the scene's index-named DOF rows.
    ArmJoints      joints;
    //! Loaded even by nmpc_bench, which runs no observer, so the observer's
    //! keys count as mapped; contact_bench runs it.
    ObserverConfig observer;
    //! The node's own frame correction, applied so a setpoint means the same
    //! thing to a bench as it does to the running stack.
    BaseAnchor     anchor;
};

/*! Merge the files the node loads, in the order it loads them, and map the
 *  result onto the controller's configuration with the library's loaders.
 *
 *  Each file's wildcard ("/" "**") block is merged first, in order, then each
 *  file's pat_arm_nmpc_<arm> block -- a later value replacing an earlier one,
 *  and a node block beating the wildcard, as ROS resolves them. The node loads
 *  arm_common.yaml and then nmpc.yaml, so pass {common, nmpc}.
 *
 *  @param not_ours        keys this bench knowingly ignores; anything neither
 *                         mapped nor listed is a hard error, so adding a knob
 *                         to nmpc.yaml without teaching it to the loaders says so
 *                         instead of quietly measuring something else
 *  @param default_model   used when neither `mjcf_override` nor the file's
 *                         mjcf_path gives one
 *  @param tool            named in the error message
 */
Loaded loadConfig(const std::vector<std::string>& yaml_paths, const std::string& arm,
                  const std::string& mjcf_override,
                  const std::string& default_model,
                  const std::set<std::string>& not_ours,
                  const std::string& tool);

}  // namespace bench
}  // namespace pat_arm_control

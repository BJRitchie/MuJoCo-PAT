#include "bench_yaml.h"

#include <stdexcept>

#include "pat_arm_control/control/nmpc/nmpc_config.h"

namespace pat_arm_control {
namespace bench {

void flatten(const YAML::Node& node, const std::string& prefix, Flat& out) {
    if (node.IsMap()) {
        for (const auto& kv : node) {
            flatten(kv.second, prefix.empty()
                                   ? kv.first.as<std::string>()
                                   : prefix + "." + kv.first.as<std::string>(), out);
        }
    } else {
        out[prefix] = node;
    }
}

const YAML::Node* YamlParamSource::find(const std::string& key) {
    used_.insert(key);
    auto it = f_.find(key);
    return it == f_.end() ? nullptr : &it->second;
}

double YamlParamSource::getDouble(const std::string& key, double def) {
    const YAML::Node* n = find(key);
    return n ? n->as<double>() : def;
}

int YamlParamSource::getInt(const std::string& key, int def) {
    const YAML::Node* n = find(key);
    return n ? n->as<int>() : def;
}

bool YamlParamSource::getBool(const std::string& key, bool def) {
    const YAML::Node* n = find(key);
    return n ? n->as<bool>() : def;
}

std::string YamlParamSource::getString(const std::string& key, const std::string& def) {
    const YAML::Node* n = find(key);
    return n ? n->as<std::string>() : def;
}

std::vector<double> YamlParamSource::getDoubles(const std::string& key,
                                                const std::vector<double>& def) {
    const YAML::Node* n = find(key);
    if (!n) return def;
    std::vector<double> v;
    for (const auto& e : *n) v.push_back(e.as<double>());
    return v;
}

std::vector<std::string> YamlParamSource::getStrings(const std::string& key,
                                                     const std::vector<std::string>& def) {
    const YAML::Node* n = find(key);
    if (!n) return def;
    std::vector<std::string> v;
    for (const auto& e : *n) v.push_back(e.as<std::string>());
    return v;
}

Loaded loadConfig(const std::vector<std::string>& yaml_paths, const std::string& arm,
                  const std::string& mjcf_override,
                  const std::string& default_model,
                  const std::set<std::string>& not_ours,
                  const std::string& tool) {
    std::vector<YAML::Node> roots;
    std::string files;   // for messages
    for (const auto& path : yaml_paths) {
        roots.push_back(YAML::LoadFile(path));
        files += (files.empty() ? "" : ", ") + path;
    }

    // The shared blocks, then this arm's, merged the way ROS merges them: the
    // per-node block wins, and a later file wins over an earlier one. "/**" is
    // the wildcard every node loads.
    Flat f;
    const char* shared = "/**";
    for (const auto& root : roots) {
        if (root[shared] && root[shared]["ros__parameters"])
            flatten(root[shared]["ros__parameters"], "", f);
    }
    const std::string node_name = "pat_arm_nmpc_" + arm;
    bool have_node = false;
    for (const auto& root : roots) {
        if (root[node_name] && root[node_name]["ros__parameters"]) {
            flatten(root[node_name]["ros__parameters"], "", f);
            have_node = true;
        }
    }
    if (!have_node) {
        throw std::runtime_error("no '" + node_name + ": ros__parameters:' block in " +
                                 files);
    }

    // The library's loaders, the same ones the node runs.
    YamlParamSource src(f);
    Loaded L;
    L.joints   = loadArmJoints(src);
    L.params   = loadNmpcParams(src, L.joints);
    L.observer = loadObserverConfig(src);
    L.anchor   = loadBaseAnchor(src);

    // --model wins, then the params file, then the installed model. nmpc.yaml
    // ships mjcf_path as "" because the launch file substitutes it from
    // FindPackageShare, so in practice the compiled-in default is what gets used.
    // Read even when overridden, so the key always counts as mapped.
    const std::string from_file = src.getString("mjcf_path", "");
    L.mjcf = !mjcf_override.empty() ? mjcf_override
           : !from_file.empty()     ? from_file
                                    : default_model;

    // The drift guard: anything in the file this binary neither mapped nor
    // knowingly skipped. A new knob in nmpc.yaml lands here rather than being
    // silently dropped, which would make the bench measure a configuration
    // nobody deploys.
    std::vector<std::string> unknown;
    for (const auto& [k, v] : f) {
        (void)v;
        if (!src.used().count(k) && !not_ours.count(k)) unknown.push_back(k);
    }
    if (!unknown.empty()) {
        std::string msg = tool + " does not know these keys in " + files + ":";
        for (const auto& k : unknown) msg += "\n    " + k;
        msg += "\n  Map them in the library's config loaders (controller_config.cpp, "
               "nmpc_config.cpp), or add them to the tool's not_ours set if they "
               "cannot affect what it measures.";
        throw std::runtime_error(msg);
    }
    return L;
}

}  // namespace bench
}  // namespace pat_arm_control

// ompc_node: ArmControlNode running the one-step MPC task-space law (ArmOMPC).
//
// Its tag makes the node pat_arm_ompc (launched per arm as pat_arm_ompc_left /
// _right) and its log lines pat_arm_ompc[side]. Everything else is the shared
// node: the factory below only builds the law from the node's parameters.
#include <cstdio>
#include <memory>
#include <sstream>

#include "arm_control_node.h"
#include "pat_arm_control/control/ompc/arm_ompc.h"
#include "pat_arm_control/control/ompc/ompc_config.h"

namespace {

ArmControlNode::LawInstance makeOmpc(const std::string& mjcf_path,
                                     pat_arm_control::ParamSource& params,
                                     const pat_arm_control::ArmJoints& joints,
                                     double control_period) {
    pat_arm_control::OMPCParams p = pat_arm_control::loadOmpcParams(params, joints);
    p.control_period = control_period;

    ArmControlNode::LawInstance inst;
    inst.controller = std::make_unique<pat_arm_control::ArmOMPC>(mjcf_path, p);
    inst.ee_site = p.ee_site_name;

    char summary[64];
    std::snprintf(summary, sizeof summary, "Ts=%.3f", p.Ts);
    inst.summary = summary;

    std::ostringstream meta;
    meta << " Ts=" << p.Ts;
    inst.trace_meta = meta.str();
    return inst;
}

}  // namespace

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    // Construction throws stay fatal: a misconfigured controller must refuse to
    // run rather than degrade.
    std::shared_ptr<ArmControlNode> node;
    try {
        node = std::make_shared<ArmControlNode>(ArmControlNode::ControlLaw{"ompc", makeOmpc});
    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("pat_arm_ompc"),
                     "failed to construct node: %s", e.what());
        rclcpp::shutdown();
        return 1;
    }
    // After construction, so DDS has already created its threads at normal
    // priority and cannot inherit the control thread's policy.
    node->start();
    rclcpp::spin(node);
    rclcpp::shutdown();
}

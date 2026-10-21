// smc_node: ArmControlNode running the sliding-mode task-space law (ArmSMC).
//
// Its tag makes the node pat_arm_smc (launched per arm as pat_arm_smc_left /
// _right) and its log lines pat_arm_smc[side]. Everything else is the shared
// node: the factory below only builds the law from the node's parameters.
#include <cstdio>
#include <memory>
#include <sstream>

#include "arm_control_node.h"
#include "pat_arm_control/control/smc/arm_smc.h"
#include "pat_arm_control/control/smc/smc_config.h"

namespace {

ArmControlNode::LawInstance makeSmc(const std::string& mjcf_path,
                                    pat_arm_control::ParamSource& params,
                                    const pat_arm_control::ArmJoints& joints,
                                    double control_period) {
    pat_arm_control::SMCParams p = pat_arm_control::loadSmcParams(params, joints);
    p.control_period = control_period;

    ArmControlNode::LawInstance inst;
    inst.controller = std::make_unique<pat_arm_control::ArmSMC>(mjcf_path, p);
    inst.ee_site = p.ee_site_name;

    char summary[64];
    std::snprintf(summary, sizeof summary, "eps=%.3f", p.epsilon);
    inst.summary = summary;

    std::ostringstream meta;
    meta << " eps=" << p.epsilon;
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
        node = std::make_shared<ArmControlNode>(ArmControlNode::ControlLaw{"smc", makeSmc});
    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("pat_arm_smc"),
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

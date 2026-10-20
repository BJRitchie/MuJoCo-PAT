// nmpc_node: ArmControlNode running the task-space NMPC (ArmNMPC).
//
// Everything a law needs on the ROS side: its tag (the node is pat_arm_nmpc,
// its log lines pat_arm_nmpc[side]) and a factory that builds it from the
// node's parameters. A new law's executable is this file with its own params
// loader and controller class.
#include <cstdio>
#include <memory>
#include <sstream>

#include "arm_control_node.h"
#include "pat_arm_control/control/nmpc/arm_nmpc.h"
#include "pat_arm_control/control/nmpc/nmpc_config.h"

namespace {

ArmControlNode::LawInstance makeNmpc(const std::string& mjcf_path,
                                     pat_arm_control::ParamSource& params,
                                     const pat_arm_control::ArmJoints& joints,
                                     double control_period) {
    pat_arm_control::NMPCParams p = pat_arm_control::loadNmpcParams(params, joints);
    // Tier 1 replays a plan whose stages are Ts apart on ticks this far apart.
    p.control_period = control_period;

    ArmControlNode::LawInstance inst;
    inst.controller = std::make_unique<pat_arm_control::ArmNMPC>(mjcf_path, p);
    inst.ee_site = p.ee_site_name;

    char summary[64];
    std::snprintf(summary, sizeof summary, "N=%d Ts=%.3f", p.N, p.Ts);
    inst.summary = summary;

    std::ostringstream meta;
    meta << " N=" << p.N << " Ts=" << p.Ts << " warm_start=" << p.qp_warm_start
         << " qp_max_iter=" << p.qp_max_iter;
    inst.trace_meta = meta.str();
    return inst;
}

}  // namespace

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    // Construction throws stay fatal — a misconfigured controller must refuse to
    // run rather than degrade — but report them through the logger and a clean
    // exit code instead of an uncaught terminate, which prints nothing useful.
    std::shared_ptr<ArmControlNode> node;
    try {
        node = std::make_shared<ArmControlNode>(ArmControlNode::ControlLaw{"nmpc", makeNmpc});
    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("pat_arm_nmpc"),
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

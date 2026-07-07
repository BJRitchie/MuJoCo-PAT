#ifndef SIMULATION_NODE_HPP_
#define SIMULATION_NODE_HPP_


#include <array>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <pat_msgs/msg/thruster_command.hpp>

#include "pat_simulation/mujoco_sim.hpp"
#include "pat_simulation/header.h"

using namespace std::chrono_literals;

namespace simulation_node 
{

class SimulationNode : public rclcpp::Node 
{
public:
    SimulationNode(); 
    virtual ~SimulationNode() = default;


private:
    void publish(); 

    void publishOdom(rclcpp::Publisher<nav_msgs::msg::Odometry>& pub,
                     const std::string& frame, const std::string& child,
                     const pat_simulation::PlanarState& s, const rclcpp::Time& t); 

    void publishImu(const pat_simulation::PlanarState& s, const rclcpp::Time& t); 
      
    void broadcastTF(const std::string& parent, const std::string& child,
                     const pat_simulation::PlanarState& s, const rclcpp::Time& t);

    void publishArmState(const rclcpp::Time& t);

    // Callbacks
    void thrusterSubscriberCallback(const pat_msgs::msg::ThrusterCommand::SharedPtr msg);
    void armTorqueSubscriberCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
    void simTimerCallback();

    std::unique_ptr<pat_simulation::MuJoCoSim> sim_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr ch_odom_, tg_odom_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr   ch_imu_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr arm_state_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster>        tf_br_;
    rclcpp::Subscription<pat_msgs::msg::ThrusterCommand>::SharedPtr thr_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr arm_torque_sub_;
    rclcpp::TimerBase::SharedPtr sim_tmr_, pub_tmr_;
    std::vector<double> ctrl_;
    std::vector<std::string> arm_joint_names_;
    std::vector<int> arm_actuator_ids_;
    std::mutex mu_;
    double imu_noise_{0.005};
    std::mt19937 rng_{std::random_device{}()};
};

} // end namespace simulation_node



#endif  // SIMULATION_NODE_HPP_
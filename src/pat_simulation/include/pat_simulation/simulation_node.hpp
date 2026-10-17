#ifndef SIMULATION_NODE_HPP_
#define SIMULATION_NODE_HPP_


#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <pat_msgs/msg/thruster_command.hpp>

#include "pat_simulation/mujoco_sim.hpp"
#include "pat_simulation/sensors/sensor_model.hpp"
#include "pat_simulation/sensors/disturbance_model.hpp"
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

    /// `h_pose` is the first of three consecutive SensorModel handles for
    /// x/y/theta, `h_vel` the first of three for vx/vy/omega. Pass -1 for
    /// either to publish that part as exact truth.
    void publishOdom(rclcpp::Publisher<nav_msgs::msg::Odometry>& pub,
                     const std::string& frame, const std::string& child,
                     const pat_simulation::PlanarState& s, const rclcpp::Time& t,
                     int h_pose, int h_vel);

    void publishImu(const rclcpp::Time& t);

    //! MuJoCo's simulated time as a ROS stamp. Every message this node
    //! publishes is stamped with it, never with the node's own ROS clock: that
    //! clock only learns the time from the /clock published here, so it would
    //! lag the physics. Stamping from the physics makes a stamp say exactly
    //! when the simulated measurement was taken, which is what an estimator
    //! integrating IMU intervals between stamps needs.
    rclcpp::Time simStamp() const;

    //! Publish /clock at `t`, unless it was already published at `t`. Called
    //! before every stamped publish, so no subscriber ever receives a message
    //! stamped ahead of its own clock.
    void publishClock(const rclcpp::Time& t);
      
    void broadcastTF(const std::string& parent, const std::string& child,
                     const pat_simulation::PlanarState& s, const rclcpp::Time& t);

    void publishArmState(const rclcpp::Time& t);

    // Callbacks
    void thrusterSubscriberCallback(const pat_msgs::msg::ThrusterCommand::SharedPtr msg);
    void armTorqueSubscriberCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
    void eeSetpointMarkerCallback(int mocap_id, const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void simTimerCallback();

    std::unique_ptr<pat_simulation::MuJoCoSim> sim_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr ch_odom_, tg_odom_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr   ch_imu_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr arm_state_pub_;

    /*! /clock, from MuJoCo's own simulated time.
     *
     *  The simulator is the workspace's time source: launch files default
     *  use_sim_time:=true, so every node's ROS clock follows this. Published
     *  immediately before any message stamped with a new time (publishClock),
     *  so it runs at the faster of pub_rate_hz and imu_rate_hz and never at
     *  the full physics rate: every node with use_sim_time receives it, and
     *  the stamps the estimators need come from the messages themselves. */
    rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr clock_pub_;
    std::int64_t last_clock_ns_{-1};

    /*! The IMU's own rate. /chaser/imu is published from the physics loop
     *  every imu_every_steps_ steps rather than from publish(), so it can run
     *  up to the 500 Hz physics rate while odometry and joint states stay at
     *  pub_rate_hz -- which the arm NMPC's control_hz is matched to. */
    int imu_every_steps_{5};
    int imu_step_{0};
    std::unique_ptr<tf2_ros::TransformBroadcaster>        tf_br_;
    rclcpp::Subscription<pat_msgs::msg::ThrusterCommand>::SharedPtr thr_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr arm_torque_sub_;
    // Cosmetic-only: mirrors /chaser/arm/<side>/ee_setpoint onto a mocap
    // marker body for eyeballing controller performance in the viewer (see
    // MuJoCoSim::setMocapPose). Absent from a model without the marker
    // bodies — mocapId() returns -1 and the callback is then a no-op.
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr ee_setpoint_marker_sub_L_, ee_setpoint_marker_sub_R_;
    // Z height the ee_setpoint markers are drawn at -- the arms only rotate
    // about world Z, so their EE sites stay at a fixed height regardless of
    // configuration; read once from the "chaser" body's world pos at
    // startup (see eeSetpointMarkerCallback) rather than hardcoded, so it
    // can't drift out of sync with air_bearing_table.xacro's own anchor.
    double marker_height_{0.0};
    rclcpp::TimerBase::SharedPtr sim_tmr_, pub_tmr_;
    std::vector<double> ctrl_;

    // Arm-torque watchdog. ctrl_ is otherwise held indefinitely, so a
    // controller that dies or stalls would leave its last torque applied for
    // the rest of the run — and a latched constant torque does not merely
    // coast the arm, it accelerates it without bound once the joint reverses.
    // Zeroing on silence is also what a real motor driver's watchdog does, so
    // the simulated actuators behave like the hardware they stand in for.
    // <= 0 disables the check.
    //
    // Counted in sim steps rather than off a clock, which keeps the window in
    // SIMULATED time -- the time the arm dynamics actually experience,
    // whatever wall rate the sim happens to run at -- and needs nothing from
    // this node's own ROS clock, which only learns the time from the /clock
    // this node publishes.
    double arm_torque_timeout_{0.0};        //!< [s], as configured
    int  arm_torque_timeout_steps_{0};      //!< the above, in sim steps
    int  arm_torque_silent_steps_{0};
    bool arm_torque_seen_{false};
    std::vector<std::string> arm_joint_names_;
    std::vector<int> arm_actuator_ids_;
    std::mutex mu_;

    // Measurement imperfection. Every published channel goes through this, and
    // every channel defaults to a perfect pass-through, so a run with no sensor
    // configuration publishes exactly what MuJoCo computed. 
    pat_simulation::SensorModel sensors_;

    // Handles into sensors_, resolved once at startup. Vector-valued channels
    // are registered consecutively so applyVector can take the first handle.
    int h_imu_accel_{-1};     //!< x, y
    int h_imu_gyro_{-1};      //!< z only; the table is planar
    int h_star_yaw_{-1};
    int h_nav_{-1};           //!< chaser x, y
    int h_chaser_vel_{-1};    //!< vx, vy, omega
    int h_target_{-1};        //!< x, y, theta
    int h_target_vel_{-1};    //!< vx, vy, omega
    int h_joint_pos_{-1};     //!< one per arm joint, arm_joint_names order
    int h_joint_vel_{-1};     //!< likewise

    // MuJoCo <sensor> indices from the MJCF, resolved once at startup.
    int sid_accel_{-1}, sid_gyro_{-1}, sid_attitude_{-1}, sid_position_{-1};

    /// Declare and read one channel's eight parameters under
    /// "sensors.<name>.*", register it, and return its handle. `default_sigma`
    /// seeds noise_std only, for the one channel that has a pre-existing
    /// non-zero default to preserve.
    //! `dt` is the period the channel is actually advanced at -- the IMU's
    //! period for IMU channels, the publish period for the rest. Every
    //! time-based channel setting is counted in calls through it.
    int declareChannel(const std::string& name, double default_sigma, double dt);

    /// Declare and read a wrench profile under "disturbance.<name>.*".
    pat_simulation::WrenchProfile declareWrench(const std::string& name);

    // Commanded external disturbances, one per body that can take one. Applied
    // every sim step, since xfrc_applied persists and a time-varying profile
    // has to be rewritten. Inactive by default, in which case the simulator's
    // xfrc_applied is never touched at all.
    pat_simulation::DisturbanceModel dist_chaser_, dist_target_;
    int body_chaser_{-1}, body_target_{-1};
    bool any_disturbance_{false};
    // What was last written to the chaser's xfrc_applied. Recorded rather than
    // re-evaluated for the ground-truth topic, because the profiles advance a
    // random walk per call and asking twice would both double-step the walk
    // and report a value that was never applied.
    double last_wrench_chaser_[3]{0.0, 0.0, 0.0};

    // Ground truth for scoring a disturbance/contact estimate. Published on a
    // deliberately sim-only topic: nothing like it exists on hardware, so an
    // algorithm package subscribing to it would be a sim-to-real violation of
    // exactly the kind this workspace is built to avoid.
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr truth_pub_;
    bool publish_truth_{false};
    void publishTruth(const rclcpp::Time& t);

    //! The clock the contact-pair log is throttled against. Steady, NOT the
    //! node clock: with use_sim_time the node clock runs at whatever rate the
    //! simulation does, and rate-limiting output is a wall-clock concern.
    rclcpp::Clock log_clock_{RCL_STEADY_TIME};
};

} // end namespace simulation_node



#endif  // SIMULATION_NODE_HPP_
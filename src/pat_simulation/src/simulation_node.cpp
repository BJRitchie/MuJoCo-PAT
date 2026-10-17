#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <random>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <pat_msgs/msg/thruster_command.hpp>


#include "pat_simulation/mujoco_sim.hpp"
#include "pat_simulation/simulation_node.hpp" 

namespace simulation_node 
{

// === Public Interface ===========================================================
SimulationNode::SimulationNode() : Node("pat_simulation") {
    // Parameters 
    const auto model_path = declare_parameter<std::string>("model_path", "");
    const double sim_hz   = declare_parameter<double>("sim_rate_hz",  500.0);
    const double pub_hz   = declare_parameter<double>("pub_rate_hz",  100.0);
    // Off for headless use (CI, batch tools like tools/pinn_datagen/, a
    // server with no GLX/DRM device) -- GLFW window creation is otherwise
    // unconditional and fails loudly (though non-fatally) without one.
    const bool visualise  = declare_parameter<bool>("visualise", true);

    // Check the model path
    if (model_path.empty())
        throw std::runtime_error("'model_path' parameter must be set via launch file");

    // Occupy sim class pointer
    sim_ = std::make_unique<pat_simulation::MuJoCoSim>(model_path, visualise);
    RCLCPP_INFO(get_logger(), "MuJoCo loaded. dt=%.4f s", sim_->dt());
    ctrl_.assign(sim_->model()->nu, 0.0);

    // Arm joints/actuators are entirely opt-in via config — empty by default.
    // Adding a joint (or a whole second arm) means extending these two lists
    // and the MJCF; no source changes here are needed.
    arm_joint_names_ = declare_parameter<std::vector<std::string>>(
        "arm_joint_names", std::vector<std::string>{});
    arm_torque_timeout_ = declare_parameter<double>("arm_torque_timeout", 0.1);
    arm_torque_timeout_steps_ =
        static_cast<int>(std::lround(arm_torque_timeout_ * sim_hz));
    const auto arm_actuator_names = declare_parameter<std::vector<std::string>>(
        "arm_actuator_names", std::vector<std::string>{});
    for (const auto& n : arm_actuator_names)
        arm_actuator_ids_.push_back(sim_->actuatorId(n));

    // Per-launch initial arm configuration — offline dataset generation
    // (tools/pinn_datagen/) needs a different qpos0 per trajectory, which
    // MuJoCoSim's construction path (mj_resetData -> mj_forward, always
    // qpos0=0) cannot otherwise produce, and MuJoCoSim::reset() only restores
    // qpos0, not an arbitrary pose. Empty (default) => untouched all-zero
    // pose, fully backward compatible.
    const auto initial_arm_qpos = declare_parameter<std::vector<double>>(
        "initial_arm_qpos", std::vector<double>{});
    if (!initial_arm_qpos.empty()) {
        if (initial_arm_qpos.size() != arm_joint_names_.size())
            throw std::runtime_error(
                "'initial_arm_qpos' size must match 'arm_joint_names' "
                "(or be left empty for the default zero pose)");
        sim_->setJointPositions(arm_joint_names_, initial_arm_qpos);
        RCLCPP_INFO(get_logger(), "Applied initial_arm_qpos override (%zu joints)",
                    initial_arm_qpos.size());
    }

    // Same mechanism as initial_arm_qpos, for the target's 3 planar joints
    // (target_x, target_y, target_yaw) -- e.g. to park the target somewhere
    // a trial's sampled arm configs can't reach. Empty (default) => untouched
    // at its xacro pos.
    const auto initial_target_qpos = declare_parameter<std::vector<double>>(
        "initial_target_qpos", std::vector<double>{});
    if (!initial_target_qpos.empty()) {
        if (initial_target_qpos.size() != 3)
            throw std::runtime_error(
                "'initial_target_qpos' must have exactly 3 entries "
                "(target_x, target_y, target_yaw), or be left empty");
        sim_->setJointPositions({"target_x", "target_y", "target_yaw"}, initial_target_qpos);
        RCLCPP_INFO(get_logger(), "Applied initial_target_qpos override");
    }

    // Table drag on the planar platforms. The air-bearing table is modelled as
    // per-DOF joint damping (viscous) + frictionloss (dry) - the bodies don't
    // contact the table geom (different collision group, and gravity is off),
    // so the surface geom's own friction never applies. Order: x, y, yaw.
    // Defaults reproduce the previously hardcoded xacro values.
    const auto chaser_damping  = declare_parameter<std::vector<double>>(
        "chaser_damping", std::vector<double>{0.05, 0.05, 0.02});
    const auto chaser_floss    = declare_parameter<std::vector<double>>(
        "chaser_frictionloss", std::vector<double>{0.0, 0.0, 0.0});
    const auto target_damping  = declare_parameter<std::vector<double>>(
        "target_damping", std::vector<double>{0.01, 0.01, 0.005});
    const auto target_floss    = declare_parameter<std::vector<double>>(
        "target_frictionloss", std::vector<double>{0.0, 0.0, 0.0});
    for (const auto* v : {&chaser_damping, &chaser_floss, &target_damping, &target_floss})
        if (v->size() != 3)
            throw std::runtime_error(
                "chaser/target damping and frictionloss must each have exactly 3 entries (x, y, yaw)");
    sim_->setJointDrag({"chaser_x", "chaser_y", "chaser_yaw"}, chaser_damping, chaser_floss);
    sim_->setJointDrag({"target_x", "target_y", "target_yaw"}, target_damping, target_floss);

    // --- Measurement model -------------------------------------------------
    // The MJCF's <sensor> block computes the flight-representative signals
    // exactly; SensorModel then corrupts them by however much the config asks
    // for, which is nothing by default. MuJoCo 3.7 cannot do this itself --
    // mjModel.sensor_noise exists but has no enable flag.
    sid_accel_    = sim_->sensorId("chaser_accel");
    sid_gyro_     = sim_->sensorId("chaser_gyro");
    sid_attitude_ = sim_->sensorId("chaser_attitude");
    sid_position_ = sim_->sensorId("chaser_position");

    // Deterministic by default, and logged below, so a trial can be replayed.
    // The previous code seeded from std::random_device, which made a run that
    // produced an interesting number impossible to reproduce.
    const auto seed = declare_parameter<int64_t>("sensor_seed", 20251014);
    sensors_ = pat_simulation::SensorModel(sim_->dt(),
                                           static_cast<std::uint64_t>(seed));

    // The gyro keeps a non-zero default because it is the one channel with
    // prior behaviour to preserve: /chaser/imu has always carried this much
    // noise on angular_velocity.z and pat_gnc's navigator consumes it. This
    // was the `imu_noise_std` parameter, which is now just this channel's
    // noise_std -- one name for one knob rather than two for the same one.
    // The accelerometer gets no such default: it used to publish pure noise
    // with no signal underneath, so there is no behaviour there worth keeping.
    constexpr double kGyroSigmaDefault = 0.005;

    h_imu_accel_  = declareChannel("imu_accel_x");
                    declareChannel("imu_accel_y");
    h_imu_gyro_   = declareChannel("imu_gyro_z", kGyroSigmaDefault);
    // The chaser's pose triple, registered consecutively as x, y, theta so one
    // handle covers it. Position is the relative-navigation channel and
    // attitude is the star tracker's, which is why they are named differently
    // even though odom presents them together -- degrade them independently.
    h_nav_        = declareChannel("nav_x");
                    declareChannel("nav_y");
    h_star_yaw_   = declareChannel("star_yaw");
    h_chaser_vel_ = declareChannel("chaser_vx");
                    declareChannel("chaser_vy");
                    declareChannel("chaser_omega");
    h_target_     = declareChannel("target_x");
                    declareChannel("target_y");
                    declareChannel("target_yaw");
    h_target_vel_ = declareChannel("target_vx");
                    declareChannel("target_vy");
                    declareChannel("target_omega");
    // Arm encoders, one channel per joint so a single bad joint can be
    // simulated. Registered in arm_joint_names_ order, matching publishArmState.
    for (std::size_t i = 0; i < arm_joint_names_.size(); ++i) {
        const int h = declareChannel("joint_pos." + arm_joint_names_[i]);
        if (i == 0) h_joint_pos_ = h;
    }
    for (std::size_t i = 0; i < arm_joint_names_.size(); ++i) {
        const int h = declareChannel("joint_vel." + arm_joint_names_[i]);
        if (i == 0) h_joint_vel_ = h;
    }

    // The shipped config is the baseline: every channel exact except the gyro
    // noise this node has always published. Warning about that would be crying
    // wolf on every launch, so the WARN is reserved for a config that actually
    // departs from what the committed figures were measured against.
    pat_simulation::ChannelConfig gyro_baseline;
    gyro_baseline.noise_std = kGyroSigmaDefault;
    bool departs = false;
    for (int i = 0; i < static_cast<int>(sensors_.size()); ++i) {
        const auto& c = sensors_.channel(i).config();
        if (c.perfect()) continue;
        if (i == h_imu_gyro_ && c == gyro_baseline) continue;
        departs = true;
        break;
    }

    if (departs) {
        RCLCPP_WARN(get_logger(),
                    "sensors: IMPERFECT, seed %ld -- measurements are NOT ground "
                    "truth, and the committed timing and constraint figures do "
                    "not apply to this run:%s",
                    static_cast<long>(seed), sensors_.report().c_str());
    } else {
        RCLCPP_INFO(get_logger(),
                    "sensors: baseline (%zu channels exact, gyro sigma %.4f), "
                    "seed %ld", sensors_.size(), kGyroSigmaDefault,
                    static_cast<long>(seed));
    }

    // --- Disturbances ------------------------------------------------------
    // Three mechanisms, all off by default. The first two are already
    // available without any of this: chaser_damping/frictionloss above is
    // unmodelled drag as far as pat_arm_control is concerned, since its
    // internal model has none.
    //
    // Mass error makes the controller's model genuinely wrong rather than
    // merely coarse: the factors apply to the SCENE only, and
    // pat_arm_control's own MJCF is left alone. Scaling at startup from config
    // rather than in the xacro means a sweep needs no rebuild.
    const auto mass_bodies = declare_parameter<std::vector<std::string>>(
        "mass_scale_bodies", std::vector<std::string>{});
    const auto mass_factors = declare_parameter<std::vector<double>>(
        "mass_scale_factors", std::vector<double>{});
    if (!mass_bodies.empty()) {
        sim_->setBodyMassScale(mass_bodies, mass_factors);
        RCLCPP_WARN(get_logger(),
                    "mass scaling applied to %zu bodies -- the simulated "
                    "vehicle no longer matches any controller's model",
                    mass_bodies.size());
    }

    // A commanded external wrench is the only disturbance whose value is known
    // exactly, which is what makes it the instrument for scoring an observer.
    body_chaser_ = sim_->bodyId("chaser");
    body_target_ = sim_->bodyId("target");
    const auto seed_u = static_cast<std::uint64_t>(seed);
    dist_chaser_ = pat_simulation::DisturbanceModel(
        declareWrench("chaser"), sim_->dt(), seed_u ^ 0x1111ULL);
    dist_target_ = pat_simulation::DisturbanceModel(
        declareWrench("target"), sim_->dt(), seed_u ^ 0x2222ULL);
    any_disturbance_ = dist_chaser_.active() || dist_target_.active();
    if (any_disturbance_) {
        RCLCPP_WARN(get_logger(), "external wrench: chaser [%s], target [%s]",
                    dist_chaser_.describe().c_str(),
                    dist_target_.describe().c_str());
    }

    // Publishers
    // ClockQoS: best-effort, depth 1 -- the standard profile for /clock, so
    // every node's built-in time-source subscription matches it.
    clock_pub_ = create_publisher<rosgraph_msgs::msg::Clock>("/clock", rclcpp::ClockQoS());
    ch_odom_ = create_publisher<nav_msgs::msg::Odometry>("/chaser/odom", 10);
    tg_odom_ = create_publisher<nav_msgs::msg::Odometry>("/target/odom", 10);
    ch_imu_  = create_publisher<sensor_msgs::msg::Imu>("/chaser/imu", 10);
    arm_state_pub_ = create_publisher<sensor_msgs::msg::JointState>("/chaser/arm/joint_states", 10);
    // Ground truth, for scoring an estimator against what actually happened.
    // /sim_only/ names it for what it is: no hardware driver will ever publish
    // this, so nothing that has to run on the Jetson may subscribe to it. Off
    // by default -- it costs a publish per cycle and exists for experiments.
    publish_truth_ = declare_parameter<bool>("publish_ground_truth", false);
    if (publish_truth_) {
        truth_pub_ = create_publisher<sensor_msgs::msg::JointState>(
            "/sim_only/ground_truth", 10);
        RCLCPP_INFO(get_logger(),
                    "publishing /sim_only/ground_truth (constraint forces + "
                    "applied wrench) -- simulation diagnostics only");
    }
    tf_br_   = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // Subscriber
    thr_sub_ = create_subscription<pat_msgs::msg::ThrusterCommand>(
        "/chaser/thruster_command", 10,
        std::bind(&SimulationNode::thrusterSubscriberCallback, this, std::placeholders::_1));
    arm_torque_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        "/chaser/arm/torque_command", 10,
        std::bind(&SimulationNode::armTorqueSubscriberCallback, this, std::placeholders::_1));

    // Cosmetic ee_setpoint markers -- gracefully absent if the loaded MJCF
    // doesn't have the marker bodies (mocapId() returns -1, resolved once
    // here rather than by name on every message).
    const int marker_mocap_id_L = sim_->mocapId("ee_setpoint_marker_L");
    const int marker_mocap_id_R = sim_->mocapId("ee_setpoint_marker_R");
    // Draw the markers at the chaser's true world-frame height rather than a
    // guessed constant -- the arms rotate about world Z only, so every EE
    // site sits at this same height regardless of joint configuration.
    // Falls back to 0.0 if the model has no "chaser" body (already covered
    // by mocapId()'s -1 no-op path in that case anyway).
    {
        const int chaser_body_id = mj_name2id(sim_->model(), mjOBJ_BODY, "chaser");
        if (chaser_body_id >= 0) {
            marker_height_ = sim_->data()->xpos[3 * chaser_body_id + 2];
        }
    }
    // Lambdas, not std::bind -- std::bind's result has no fixed call
    // signature rclcpp's callback-type detection can cleanly introspect
    // once a non-placeholder argument (mocap_id) precedes the placeholder;
    // a lambda's operator() is unambiguous.
    ee_setpoint_marker_sub_L_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/chaser/arm/left/ee_setpoint", 10,
        [this, marker_mocap_id_L](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
            eeSetpointMarkerCallback(marker_mocap_id_L, msg);
        });
    ee_setpoint_marker_sub_R_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/chaser/arm/right/ee_setpoint", 10,
        [this, marker_mocap_id_R](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
            eeSetpointMarkerCallback(marker_mocap_id_R, msg);
        });

    sim_tmr_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / sim_hz),
        std::bind(&SimulationNode::simTimerCallback, this)); 
    pub_tmr_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / pub_hz),
        std::bind(&SimulationNode::publish, this));

}

// === Private Methods ============================================================
int SimulationNode::declareChannel(const std::string& name, double default_sigma) {
    const std::string p = "sensors." + name + ".";
    pat_simulation::ChannelConfig c;
    c.noise_std     = declare_parameter<double>(p + "noise_std",     default_sigma);
    c.bias          = declare_parameter<double>(p + "bias",          0.0);
    c.bias_walk_std = declare_parameter<double>(p + "bias_walk_std", 0.0);
    c.scale_error   = declare_parameter<double>(p + "scale_error",   0.0);
    c.quantisation  = declare_parameter<double>(p + "quantisation",  0.0);
    c.delay_s       = declare_parameter<double>(p + "delay_s",       0.0);
    c.dropout_prob  = declare_parameter<double>(p + "dropout_prob",  0.0);
    c.rate_hz       = declare_parameter<double>(p + "rate_hz",       0.0);
    return sensors_.addChannel(name, c);
}

pat_simulation::WrenchProfile SimulationNode::declareWrench(const std::string& name) {
    const std::string p = "disturbance." + name + ".";
    pat_simulation::WrenchProfile w;
    w.type       = declare_parameter<std::string>(p + "type", "none");
    w.fx         = declare_parameter<double>(p + "fx",         0.0);
    w.fy         = declare_parameter<double>(p + "fy",         0.0);
    w.tz         = declare_parameter<double>(p + "tz",         0.0);
    w.start_s    = declare_parameter<double>(p + "start_s",    0.0);
    w.duration_s = declare_parameter<double>(p + "duration_s", 0.0);
    w.freq_hz    = declare_parameter<double>(p + "freq_hz",    0.0);
    w.walk_std   = declare_parameter<double>(p + "walk_std",   0.0);
    return w;
}

void SimulationNode::publishTruth(const rclcpp::Time& t) {
    // Carried on a JointState because it is already a dependency and is a
    // name/value list, so the vector describes itself instead of being a bare
    // array whose meaning lives in a comment elsewhere.
    sensor_msgs::msg::JointState m;
    m.header.stamp = t;
    const double* qfrc = sim_->constraintForce();
    const int nv = sim_->nv();
    m.name.reserve(static_cast<std::size_t>(nv) + 3);
    m.effort.reserve(static_cast<std::size_t>(nv) + 3);
    for (int i = 0; i < nv; ++i) {
        m.name.push_back("qfrc_constraint." + std::to_string(i));
        m.effort.push_back(qfrc[i]);
    }
    // The wrench most recently applied to the chaser, so one log holds both
    // the disturbance that was commanded and the contact force that resulted.
    m.name.push_back("wrench.chaser.fx");
    m.effort.push_back(last_wrench_chaser_[0]);
    m.name.push_back("wrench.chaser.fy");
    m.effort.push_back(last_wrench_chaser_[1]);
    m.name.push_back("wrench.chaser.tz");
    m.effort.push_back(last_wrench_chaser_[2]);
    // The contact count, because qfrc_constraint alone cannot distinguish a
    // contact from a joint limit -- MuJoCo puts both in it. A nonzero
    // qfrc_constraint with ncon == 0 is a joint hitting its range, which looks
    // exactly like contact to anything scoring the vector and means something
    // completely different.
    m.name.push_back("ncon");
    m.effort.push_back(static_cast<double>(sim_->contactCount()));
    truth_pub_->publish(m);

    // Which geoms are touching, throttled. A JointState cannot carry names
    // that change every step, and scenario authoring needs them: the DOF rows
    // alone cannot distinguish an arm against the target from the two arms
    // against each other. Only while ground truth is on, so it costs nothing
    // in a normal run.
    const int ncon = sim_->contactCount();
    if (ncon > 0) {
        RCLCPP_INFO_THROTTLE(get_logger(), log_clock_, 1000,
                             "contact (%d): %s", ncon,
                             sim_->contactPairs().c_str());
    }
}

void SimulationNode::publish() {
    // Every message on this tick is stamped with MuJoCo's simulated time, not
    // with this node's ROS clock: that clock only learns the time from the
    // /clock published here, so it would lag the physics by up to a tick.
    // Stamping from the physics directly makes a stamp say exactly when the
    // simulated measurement was taken, which is what an estimator that
    // integrates IMU intervals between stamps needs.
    const rclcpp::Time now(static_cast<std::int64_t>(std::llround(sim_->time() * 1e9)),
                           RCL_ROS_TIME);

    // /clock first, so no subscriber ever receives a message stamped ahead of
    // its own clock.
    rosgraph_msgs::msg::Clock clk;
    clk.clock = now;
    clock_pub_->publish(clk);

    // mj_step evaluates the dynamics before integrating, so mjData's derived
    // quantities -- the <sensor> outputs among them -- describe the state the
    // last step started from, and under RK4 an intermediate stage of it. One
    // forward evaluation here puts the sensors on the same instant as the
    // qpos/qvel that odom is built from; without it the IMU and the odometry
    // disagree by a few percent and anything estimating from both reads the
    // mismatch as a real disturbance. Once per publish (100 Hz), not once per
    // step (500 Hz).
    sim_->refreshSensors();

    // Relevant satellites
    const auto ch  = sim_->getChaserState();
    const auto tg  = sim_->getTargetState();

    // Publish data for each satellite. The corrupted copies are what go out;
    // the TF broadcast below deliberately keeps ground truth, because TF is a
    // visualisation and debugging aid here, not a sensor the algorithms read.
    publishOdom(*ch_odom_, "map", "chaser", ch, now, h_nav_, h_chaser_vel_);
    publishOdom(*tg_odom_, "map", "target", tg, now, h_target_, h_target_vel_);
    publishImu(now);
    publishArmState(now);
    if (publish_truth_) publishTruth(now);

    // Broadcast the transforms
    broadcastTF("map", "chaser", ch, now);
    broadcastTF("map", "target", tg, now);
}

void SimulationNode::publishOdom(rclcpp::Publisher<nav_msgs::msg::Odometry>& pub,
                     const std::string& frame, const std::string& child,
                     const pat_simulation::PlanarState& s, const rclcpp::Time& t,
                     int h_pose, int h_vel) {

    // Init odom msg
    nav_msgs::msg::Odometry m;

    // Populate info
    m.header.stamp = t;
    m.header.frame_id = frame;
    m.child_frame_id = child;

    // Measured pose and twist. With no sensor configuration both calls are
    // exact pass-throughs and these are the same values the node has always
    // published, bit for bit.
    const double pose_truth[3] = {s.x, s.y, s.theta};
    const double vel_truth[3]  = {s.xdot, s.ydot, s.thetadot};
    double pose[3] = {s.x, s.y, s.theta};
    double vel[3]  = {s.xdot, s.ydot, s.thetadot};
    if (h_pose >= 0) sensors_.applyVector(h_pose, pose_truth, pose, 3);
    if (h_vel  >= 0) sensors_.applyVector(h_vel,  vel_truth,  vel,  3);

    // Populate pose
    m.pose.pose.position.x = pose[0];
    m.pose.pose.position.y = pose[1];

    // Orientation
    tf2::Quaternion q;
    q.setRPY(0, 0, pose[2]);
    m.pose.pose.orientation.w = q.w();
    m.pose.pose.orientation.x = q.x();
    m.pose.pose.orientation.y = q.y();
    m.pose.pose.orientation.z = q.z();

    // Velocities
    m.twist.twist.linear.x = vel[0];
    m.twist.twist.linear.y = vel[1];
    m.twist.twist.angular.z = vel[2];
    pub.publish(m);
}

void SimulationNode::publishImu(const rclcpp::Time& t) {
    // Init imu msg
    sensor_msgs::msg::Imu m;

    // Populate header info
    m.header.stamp = t;
    m.header.frame_id = "chaser_imu";

    // Straight off the MJCF's accelerometer and gyro, in the imu site's own
    // (body) frame, which is the frame the message's frame_id names. The
    // accelerometer previously published pure noise with no signal under it at
    // all; it now carries the real proper acceleration, and with gravity
    // disabled there is no 1 g term to subtract -- the same thing a real IMU
    // reads in free fall.
    const double* acc  = sim_->sensorData(sid_accel_);
    const double* gyro = sim_->sensorData(sid_gyro_);

    const double acc_truth[2] = {acc[0], acc[1]};
    double acc_out[2] = {acc[0], acc[1]};
    sensors_.applyVector(h_imu_accel_, acc_truth, acc_out, 2);

    m.linear_acceleration.x = acc_out[0];
    m.linear_acceleration.y = acc_out[1];
    // Only z is meaningful on a planar table; x and y stay at the message's
    // zero rather than being filled with noise that means nothing.
    m.angular_velocity.z = sensors_.apply(h_imu_gyro_, gyro[2]);

    // Publish
    ch_imu_->publish(m);
}

void SimulationNode::broadcastTF(const std::string& parent, const std::string& child,
                     const pat_simulation::PlanarState& s, const rclcpp::Time& t) {

    // Init transform msg 
    geometry_msgs::msg::TransformStamped tf;

    // Header info 
    tf.header.stamp = t; 
    tf.header.frame_id = parent; 
    tf.child_frame_id = child;

    // Position 
    tf.transform.translation.x = s.x; 
    tf.transform.translation.y = s.y;

    // Pose 
    tf2::Quaternion q; 
    q.setRPY(0, 0, s.theta);
    tf.transform.rotation.w = q.w(); 
    tf.transform.rotation.x = q.x();
    tf.transform.rotation.y = q.y(); 
    tf.transform.rotation.z = q.z();

    // Broadcast
    tf_br_->sendTransform(tf);
}

void SimulationNode::publishArmState(const rclcpp::Time& t) {
    if (arm_joint_names_.empty()) return;
    const auto s = sim_->getJointStates(arm_joint_names_);
    sensor_msgs::msg::JointState m;
    m.header.stamp = t;
    m.name     = s.name;
    m.position = s.pos;
    m.velocity = s.vel;
    // Arm encoders. One channel per joint, in arm_joint_names_ order, which is
    // the order getJointStates returned them in -- so a single joint can be
    // given quantisation or a dead encoder without touching the others. Exact
    // by default, so m.position/m.velocity are MuJoCo's own values.
    if (h_joint_pos_ >= 0) {
        sensors_.applyVector(h_joint_pos_, s.pos.data(), m.position.data(),
                             static_cast<int>(s.pos.size()));
    }
    if (h_joint_vel_ >= 0) {
        sensors_.applyVector(h_joint_vel_, s.vel.data(), m.velocity.data(),
                             static_cast<int>(s.vel.size()));
    }
    arm_state_pub_->publish(m);
}

// === Callbacks ==================================================================
void SimulationNode::thrusterSubscriberCallback(
    const pat_msgs::msg::ThrusterCommand::SharedPtr msg) {
    // Block
    std::lock_guard<std::mutex> lk(mu_);

    // Update control forces — bound is a literal 4, matching
    // ThrusterCommand.force[4], NOT ctrl_.size() (which now covers all
    // actuators including the arm; msg->force only ever has 4 elements).
    for (size_t i = 0; i < 4 && i < ctrl_.size(); ++i)
        ctrl_[i] = msg->force[i];
}

void SimulationNode::armTorqueSubscriberCallback(
    const sensor_msgs::msg::JointState::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(mu_);
    // Match by name against our configured arm joints — a message may
    // legitimately cover only a subset (e.g. one of several independent arm
    // controllers), so unmatched names/entries are simply ignored.
    for (size_t k = 0; k < msg->name.size() && k < msg->effort.size(); ++k) {
        for (size_t i = 0; i < arm_joint_names_.size(); ++i) {
            if (msg->name[k] == arm_joint_names_[i]) {
                ctrl_[arm_actuator_ids_[i]] = msg->effort[k];
                break;
            }
        }
    }
    arm_torque_silent_steps_ = 0;
    arm_torque_seen_ = true;
}

void SimulationNode::eeSetpointMarkerCallback(
    int mocap_id, const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    if (mocap_id < 0) return;  // model has no marker body for this side -- skip
    // marker_height_ (the chaser's true world Z, read once at startup), NOT
    // msg->pose.position.z -- ee_setpoint's z is conventionally unused/zero
    // (the planar task-space law never reads it, see arm_nmpc.cpp's
    // controlLaw), which would otherwise bury the marker at table level.
    std::lock_guard<std::mutex> lk(mu_);
    sim_->setMocapPose(mocap_id, msg->pose.position.x, msg->pose.position.y, marker_height_,
                        msg->pose.orientation.w, msg->pose.orientation.x,
                        msg->pose.orientation.y, msg->pose.orientation.z);
}

void SimulationNode::simTimerCallback() {
    std::lock_guard<std::mutex> lk(mu_);

    // Drop the arm torques if their controller has gone quiet. Only the arm
    // actuators are zeroed: the thrusters are a separate topic, and GNC is never
    // run alongside the arm NMPC.
    if (arm_torque_timeout_steps_ > 0 && arm_torque_seen_ &&
        ++arm_torque_silent_steps_ > arm_torque_timeout_steps_) {
        bool was_commanding = false;
        for (int id : arm_actuator_ids_) {
            if (ctrl_[static_cast<size_t>(id)] != 0.0) was_commanding = true;
            ctrl_[static_cast<size_t>(id)] = 0.0;
        }
        if (was_commanding) {
            RCLCPP_ERROR(get_logger(),
                "pat_simulation: no arm torque_command for %.2f s — zeroing arm "
                "actuators (a held torque would keep accelerating the joints)",
                arm_torque_timeout_);
        }
    }

    // Commanded external disturbances. xfrc_applied persists across steps, so
    // a time-varying profile has to be rewritten every step; when nothing is
    // configured the field is never touched at all, which is what keeps a
    // default run bit-identical to one from before this existed.
    if (any_disturbance_) {
        const double t = sim_->time();
        if (dist_chaser_.active()) {
            dist_chaser_.wrenchAt(t, last_wrench_chaser_);
            sim_->setExternalWrench(body_chaser_, last_wrench_chaser_[0],
                                    last_wrench_chaser_[1], last_wrench_chaser_[2]);
        }
        if (dist_target_.active()) {
            double w[3];
            dist_target_.wrenchAt(t, w);
            sim_->setExternalWrench(body_target_, w[0], w[1], w[2]);
        }
    }

    sim_->step(ctrl_);
}

}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<simulation_node::SimulationNode>());
    rclcpp::shutdown();
}

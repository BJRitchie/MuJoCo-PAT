#include "pat_simulation/mujoco_sim.hpp"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace pat_simulation {

MuJoCoSim::MuJoCoSim(const std::string& model_path, bool visualise) {

    // Read in model 
    char err[1024] = {};
    model_ = mj_loadXML(model_path.c_str(), nullptr, err, sizeof(err));
    if (!model_) throw std::runtime_error("MuJoCoSim: " + std::string(err));
    
    // Init Mujoco data handle 
    data_ = mj_makeData(model_);

    mj_resetData(model_, data_);
    mj_forward(model_, data_);
    jid_cx_   = jointId("chaser_x");   jid_cy_   = jointId("chaser_y");
    jid_cyaw_ = jointId("chaser_yaw"); jid_tx_   = jointId("target_x");
    jid_ty_   = jointId("target_y");   jid_tyaw_ = jointId("target_yaw");

    // Initialise the visualiser
    if (visualise) {
        vis_ = std::make_unique<mujoco_vis::MuJoCoVisualiser>(*model_, *data_);
        if (vis_->init(5, -20, 900, 1200, "PAT Simulation") == FAIL) {
            std::cerr << "[ERROR]: Visualiser failed to initialise" << std::endl; 
        } else {
            std::cout << "[INFO]: Visualiser initialised" << std::endl; 
        }
    }
}

MuJoCoSim::~MuJoCoSim() {
    // Clean up 
    if (data_)  { mj_deleteData(data_);   data_  = nullptr; }
    if (model_) { mj_deleteModel(model_); model_ = nullptr; }
}

void MuJoCoSim::reset() {
    mj_resetData(model_, data_);
    mj_forward(model_, data_);
}

void MuJoCoSim::setJointPositions(const std::vector<std::string>& joint_names,
                                   const std::vector<double>& positions) {
    if (joint_names.size() != positions.size())
        throw std::runtime_error(
            "MuJoCoSim::setJointPositions: joint_names/positions size mismatch ("
            + std::to_string(joint_names.size()) + " vs " + std::to_string(positions.size()) + ")");
    for (size_t i = 0; i < joint_names.size(); ++i) {
        const int id = jointId(joint_names[i].c_str());
        data_->qpos[model_->jnt_qposadr[id]] = positions[i];
    }
    mj_forward(model_, data_);
}

void MuJoCoSim::setJointDrag(const std::vector<std::string>& joint_names,
                             const std::vector<double>& damping,
                             const std::vector<double>& frictionloss) {
    if (joint_names.size() != damping.size() || joint_names.size() != frictionloss.size())
        throw std::runtime_error(
            "MuJoCoSim::setJointDrag: joint_names/damping/frictionloss size mismatch");
    for (size_t i = 0; i < joint_names.size(); ++i) {
        if (damping[i] < 0.0 || frictionloss[i] < 0.0)
            throw std::runtime_error(
                "MuJoCoSim::setJointDrag: negative value for joint '" + joint_names[i] + "'");
        const int dof = model_->jnt_dofadr[jointId(joint_names[i].c_str())];
        model_->dof_damping[dof]      = damping[i];
        model_->dof_frictionloss[dof] = frictionloss[i];
    }
}

void MuJoCoSim::step(const std::vector<double>& ctrl) {
    // Update the control input
    for (int i = 0; i < model_->nu && static_cast<size_t>(i) < ctrl.size(); ++i)
        data_->ctrl[i] = ctrl[i];

    // Use mujoco step 
    mj_step(model_, data_);

    // Update visualiser (throttled — see kRenderEveryNSteps)
    if (vis_ && ++render_counter_ >= kRenderEveryNSteps) {
        render_counter_ = 0;
        if (vis_->updateWindow() == FAIL)
            std::cerr << "[WARNING] Visualiser window closed or failed to update." << std::endl;
    }
}

double MuJoCoSim::time() const noexcept { return data_->time; }
double MuJoCoSim::dt()   const noexcept { return model_->opt.timestep; }

int MuJoCoSim::jointId(const char* name) const {
    int id = mj_name2id(model_, mjOBJ_JOINT, name);
    if (id < 0) throw std::runtime_error(std::string("joint '") + name + "' not found");
    return id;
}
double MuJoCoSim::jointPos(int id) const noexcept { 
    return data_->qpos[model_->jnt_qposadr[id]]; 
}

double MuJoCoSim::jointVel(int id) const noexcept { 
    return data_->qvel[model_->jnt_dofadr[id]]; 
}

PlanarState MuJoCoSim::getChaserState() const {
    return { jointPos(jid_cx_),  jointPos(jid_cy_),  jointPos(jid_cyaw_),
             jointVel(jid_cx_),  jointVel(jid_cy_),  jointVel(jid_cyaw_) };
}
PlanarState MuJoCoSim::getTargetState() const {
    return { jointPos(jid_tx_),  jointPos(jid_ty_),  jointPos(jid_tyaw_),
             jointVel(jid_tx_),  jointVel(jid_ty_),  jointVel(jid_tyaw_) };
}

NamedJointState MuJoCoSim::getJointStates(const std::vector<std::string>& joint_names) const {
    NamedJointState s;
    s.name = joint_names;
    for (const auto& n : joint_names) {
        const int id = jointId(n.c_str());
        s.pos.push_back(jointPos(id));
        s.vel.push_back(jointVel(id));
    }
    return s;
}

int MuJoCoSim::actuatorId(const std::string& name) const {
    const int id = mj_name2id(model_, mjOBJ_ACTUATOR, name.c_str());
    if (id < 0) throw std::runtime_error("actuator '" + name + "' not found");
    return id;
}

int MuJoCoSim::mocapId(const std::string& name) const {
    const int body_id = mj_name2id(model_, mjOBJ_BODY, name.c_str());
    if (body_id < 0) return -1;
    return model_->body_mocapid[body_id];
}

void MuJoCoSim::setMocapPose(int mocap_id, double x, double y, double z,
                              double qw, double qx, double qy, double qz) {
    data_->mocap_pos[3 * mocap_id + 0] = x;
    data_->mocap_pos[3 * mocap_id + 1] = y;
    data_->mocap_pos[3 * mocap_id + 2] = z;
    data_->mocap_quat[4 * mocap_id + 0] = qw;
    data_->mocap_quat[4 * mocap_id + 1] = qx;
    data_->mocap_quat[4 * mocap_id + 2] = qy;
    data_->mocap_quat[4 * mocap_id + 3] = qz;
}

} // namespace pat_simulation

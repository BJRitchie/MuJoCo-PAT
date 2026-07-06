#ifndef MUJOCO_VIS_HPP_
#define MUJOCO_VIS_HPP_

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <string> 
#include <iostream>

#include "pat_simulation/header.h"

namespace mujoco_vis 
{

class MuJoCoVisualiser 
{
public: 
    MuJoCoVisualiser(mjModel& g_m, mjData& d); 
    ~MuJoCoVisualiser(); 

    int init(int cam_dist, int cam_elevation, int window_height, int window_width, std::string title); 
    int updateWindow(); 

private:
    void keyCallback(GLFWwindow* w, int key, int /*scancode*/, int act, int /*mods*/);
    void mouseButtonCallback(GLFWwindow* /*w*/, int button, int act, int /*mods*/);
    void cursorPosCallback(GLFWwindow* /*w*/, double x, double y);
    void scrollCallback(GLFWwindow* /*w*/, double /*dx*/, double dy);

    // GLFW callbacks are plain C function pointers and cannot bind a `this`;
    // these static trampolines recover the instance via the window's user pointer.
    static void keyCallbackTrampoline(GLFWwindow* w, int key, int scancode, int act, int mods);
    static void mouseButtonCallbackTrampoline(GLFWwindow* w, int button, int act, int mods);
    static void cursorPosCallbackTrampoline(GLFWwindow* w, double x, double y);
    static void scrollCallbackTrampoline(GLFWwindow* w, double dx, double dy);

    // Visualiser window
    GLFWwindow* window;

    // Pointers to MuJoCo model classes
    mjModel* g_m;
    mjData* d;

    // MuJoCo render context
    mjvOption opt;
    mjrContext con;
    mjrRect vp           = {0, 0, 0, 0};

    // Rendering variables 
    mjvScene g_scn;
    mjvCamera g_cam;
    bool g_paused        = false;
    bool g_button_left   = false;
    bool g_button_right  = false;
    double g_last_x      = 0;
    double g_last_y      = 0;
}; 


}

#endif 
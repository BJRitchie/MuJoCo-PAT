#include "pat_simulation/mujoco_vis.hpp"

namespace mujoco_vis
{

// === Public Interface =====================================================
MuJoCoVisualiser::MuJoCoVisualiser(mjModel& g_m, mjData& d) 
    : g_m(&g_m), d(&d) {}

MuJoCoVisualiser::~MuJoCoVisualiser() {
    // Cleanup 
    mjr_freeContext(&con);
    mjv_freeScene(&g_scn);
    glfwTerminate();
} 

int MuJoCoVisualiser::init(
    int cam_dist, int cam_elevation, int window_height, int window_width, 
    std::string title) {

    // GLFW init 
    if (!glfwInit()) { std::cerr << "GLFW init failed\n"; return FAIL; }
    window = glfwCreateWindow(window_width, window_height, title.c_str(), nullptr, nullptr);

    // Check window 
    if (!window) { std::cerr << "GLFW window failed\n"; glfwTerminate(); return FAIL; }
    glfwMakeContextCurrent(window);
    // Vsync (interval 1) blocks glfwSwapBuffers until the next monitor
    // refresh. updateWindow() is called from the ~500 Hz physics step, and
    // blocking there stalls the single-threaded executor, throttling
    // everything else (odom/imu publishing, thruster command handling).
    glfwSwapInterval(0);

    // Set callbacks — GLFW needs plain C function pointers, so route through
    // static trampolines that recover `this` from the window's user pointer.
    glfwSetWindowUserPointer(window, this);
    glfwSetKeyCallback(window, &MuJoCoVisualiser::keyCallbackTrampoline);
    glfwSetMouseButtonCallback(window, &MuJoCoVisualiser::mouseButtonCallbackTrampoline);
    glfwSetCursorPosCallback(window, &MuJoCoVisualiser::cursorPosCallbackTrampoline);
    glfwSetScrollCallback(window, &MuJoCoVisualiser::scrollCallbackTrampoline);

    // Render context 
    mjv_defaultCamera(&g_cam);
    mjv_defaultOption(&opt);
    mjv_makeScene(g_m, &g_scn, 2000);
    mjr_defaultContext(&con);
    mjr_makeContext(g_m, &con, mjFONTSCALE_150);

    g_cam.distance  = cam_dist;
    g_cam.elevation = cam_elevation;

    return SUCCESS; 
}

int MuJoCoVisualiser::updateWindow() {

    // Always poll events, even while paused, so the window stays responsive
    // (otherwise there's no way to receive the key-press that unpauses it).
    glfwPollEvents();
    if (glfwWindowShouldClose(window)) return FAIL;

    // Pausing is a valid user action, not a failure — skip the render but
    // still report success.
    if (g_paused) return SUCCESS;

    glfwGetFramebufferSize(window, &vp.width, &vp.height);
    mjv_updateScene(g_m, d, &opt, nullptr, &g_cam, mjCAT_ALL, &g_scn);
    mjr_render(vp, &g_scn, &con);
    glfwSwapBuffers(window);
    return SUCCESS;
}


// === Private Methods ======================================================

void MuJoCoVisualiser::keyCallback(GLFWwindow* w, int key, int /*scancode*/, int act, int /*mods*/)
{
    if (act != GLFW_PRESS) return;
    if (key == GLFW_KEY_ESCAPE || key == GLFW_KEY_Q)
        glfwSetWindowShouldClose(w, GLFW_TRUE);
    if (key == GLFW_KEY_SPACE)
        g_paused = !g_paused;
}

void MuJoCoVisualiser::mouseButtonCallback(GLFWwindow* /*w*/, int button, int act, int /*mods*/)
{
    g_button_left  = (button == GLFW_MOUSE_BUTTON_LEFT  && act == GLFW_PRESS);
    g_button_right = (button == GLFW_MOUSE_BUTTON_RIGHT && act == GLFW_PRESS);
}

void MuJoCoVisualiser::cursorPosCallback(GLFWwindow* /*w*/, double x, double y)
{
    double dx = x - g_last_x;
    double dy = y - g_last_y;
    g_last_x = x; g_last_y = y;
    if (!g_m) return;

    if (g_button_left)
        mjv_moveCamera(g_m, mjMOUSE_ROTATE_V, dx / 500.0, dy / 500.0, &g_scn, &g_cam);
    if (g_button_right)
        mjv_moveCamera(g_m, mjMOUSE_ZOOM, 0, dy / 200.0, &g_scn, &g_cam);
}

void MuJoCoVisualiser::scrollCallback(GLFWwindow* /*w*/, double /*dx*/, double dy)
{
    if (g_m) mjv_moveCamera(g_m, mjMOUSE_ZOOM, 0, -0.05 * dy, &g_scn, &g_cam);
}

void MuJoCoVisualiser::keyCallbackTrampoline(GLFWwindow* w, int key, int scancode, int act, int mods)
{
    static_cast<MuJoCoVisualiser*>(glfwGetWindowUserPointer(w))->keyCallback(w, key, scancode, act, mods);
}

void MuJoCoVisualiser::mouseButtonCallbackTrampoline(GLFWwindow* w, int button, int act, int mods)
{
    static_cast<MuJoCoVisualiser*>(glfwGetWindowUserPointer(w))->mouseButtonCallback(w, button, act, mods);
}

void MuJoCoVisualiser::cursorPosCallbackTrampoline(GLFWwindow* w, double x, double y)
{
    static_cast<MuJoCoVisualiser*>(glfwGetWindowUserPointer(w))->cursorPosCallback(w, x, y);
}

void MuJoCoVisualiser::scrollCallbackTrampoline(GLFWwindow* w, double dx, double dy)
{
    static_cast<MuJoCoVisualiser*>(glfwGetWindowUserPointer(w))->scrollCallback(w, dx, dy);
}

}
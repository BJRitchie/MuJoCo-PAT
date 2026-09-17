#!/usr/bin/env python3
"""
The first rclpy node in this repo (every other .py file here is a launch
description). One instance per PINN trajectory episode, launched fresh by
pinn_datagen.launch.py -- not an installed ament executable, so it's run as a
loose script via `python3 pinn_recorder_node.py --ros-args -p ...`, exactly
like any C++ node's --ros-args CLI convention.

Buffers one full row EVERY TIME /chaser/arm/joint_states arrives (100 Hz
nominal), using whatever /chaser/odom and per-joint /chaser/arm/
torque_command values are cached at that instant -- the simplest
self-consistent scheme given this stack's fully-asynchronous ROS 2 message
timing (no shared /clock, three independently-timed publishers).
joint_states IS the row trigger (not odom or torque) because
publishArmState() always emits ALL configured arm joints together in one
message (unlike torque_command, which is legitimately partial/name-matched
-- an omitted joint's commanded torque is HELD at its last value, see
simulation_node.cpp's armTorqueSubscriberCallback) -- so triggering on it
guarantees q/qdot are always fully populated for every recorded row.

Buffers are pre-allocated numpy arrays (sized from run_time_s * an expected
row rate, with a safety margin), written by index -- not Python lists
appended-to and converted at the end. /chaser/arm/joint_states fires at
~100 Hz for the whole episode, so this keeps the hot-path callback down to
plain fixed-size numpy writes with no per-call allocation or list-growth
cost. /chaser/odom and /chaser/arm/torque_command (published even more
often, combined) similarly update small fixed-size numpy caches in place
rather than a dict rebuilt per message.

Self-terminating duration control: once `run_time_s` wall-clock seconds have
elapsed since the FIRST recorded sample, dumps everything (trimmed to the
actual row count) to a .npz at `out_path` and calls rclpy.shutdown() -- no
external SIGINT race to get right. pinn_datagen.launch.py wires this node's
process exit to shut down the whole launch tree
(RegisterEventHandler(OnProcessExit(...))), so a Python orchestrator's
`subprocess.run(["ros2","launch",...])` call simply blocks for exactly one
episode.
"""

import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState

DEFAULT_ARM_JOINT_NAMES = ["joint2_L", "joint3_L", "joint5_L", "joint2_R", "joint3_R", "joint5_R"]


def _yaw_from_quat(q: Quaternion) -> float:
    # Matches simulation_node.cpp's own convention exactly: it builds this
    # quaternion via tf2::Quaternion::setRPY(0, 0, theta), so it is always a
    # pure Z-rotation -- the same 2*atan2(qz,qw) identity used in helpers.py.
    return 2.0 * float(np.arctan2(q.z, q.w))


class PinnRecorderNode(Node):
    def __init__(self):
        super().__init__("pinn_recorder")

        self.run_time_s = float(self.declare_parameter("run_time_s", 15.0).value)
        self.out_path = self.declare_parameter("out_path", "").value
        if not self.out_path:
            raise RuntimeError("pinn_recorder: 'out_path' parameter is required")
        self.arm_joint_names = list(self.declare_parameter(
            "arm_joint_names", DEFAULT_ARM_JOINT_NAMES).value)
        expected_hz = float(self.declare_parameter("expected_rate_hz", 100.0).value)

        n_j = len(self.arm_joint_names)
        self._name_to_idx = {n: i for i, n in enumerate(self.arm_joint_names)}

        # Small fixed-size numpy caches, updated in place by the
        # higher-frequency odom/torque callbacks -- no per-message
        # allocation, unlike a dict/list rebuilt from scratch each time.
        self._base_pose = np.zeros(3)   # x, y, yaw
        self._base_vel = np.zeros(3)    # vx, vy, omega
        self._tau_cache = np.zeros(n_j)

        # Pre-allocated row buffers -- avoids Python list.append + a
        # list-to-array conversion at the end on the hot path
        # (_on_joint_states fires at ~100 Hz for the whole episode). Sized
        # with a safety margin over the nominal rate; _grow() (rare) handles
        # the wall-clock rate ever running faster than expected without
        # dropping samples or indexing out of bounds.
        self._cap = max(1, int(round(self.run_time_s * expected_hz * 1.5)))
        self._t = np.zeros(self._cap)
        self._base_pose_hist = np.zeros((self._cap, 3))
        self._base_vel_hist = np.zeros((self._cap, 3))
        self._q_hist = np.zeros((self._cap, n_j))
        self._qdot_hist = np.zeros((self._cap, n_j))
        self._tau_hist = np.zeros((self._cap, n_j))
        self._n = 0
        self._t0 = None
        self._saved = False
        self.done = False   # main()'s spin loop watches this; NOT torn down from in here

        self.create_subscription(Odometry, "/chaser/odom", self._on_odom, 10)
        self.create_subscription(JointState, "/chaser/arm/torque_command", self._on_torque, 10)
        self.create_subscription(JointState, "/chaser/arm/joint_states", self._on_joint_states, 10)

        self.get_logger().info(
            f"pinn_recorder: run_time_s={self.run_time_s} out_path={self.out_path} "
            f"capacity={self._cap} rows")

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        self._base_pose[0] = p.x
        self._base_pose[1] = p.y
        self._base_pose[2] = _yaw_from_quat(msg.pose.pose.orientation)
        t = msg.twist.twist
        self._base_vel[0] = t.linear.x
        self._base_vel[1] = t.linear.y
        self._base_vel[2] = t.angular.z

    def _on_torque(self, msg: JointState):
        for name, effort in zip(msg.name, msg.effort):
            idx = self._name_to_idx.get(name)
            if idx is not None:
                self._tau_cache[idx] = effort

    def _on_joint_states(self, msg: JointState):
        if self._saved:
            return
        if self._n >= self._cap:
            self._grow()

        try:
            for i, name in enumerate(self.arm_joint_names):
                j = msg.name.index(name)
                self._q_hist[self._n, i] = msg.position[j]
                self._qdot_hist[self._n, i] = msg.velocity[j]
        except ValueError:
            # joint_states hasn't yet carried all configured joints (e.g. a
            # startup race) -- skip this row rather than record a partial one.
            return

        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        t_rel = now - self._t0

        self._t[self._n] = t_rel
        self._base_pose_hist[self._n] = self._base_pose
        self._base_vel_hist[self._n] = self._base_vel
        self._tau_hist[self._n] = self._tau_cache
        self._n += 1

        if t_rel >= self.run_time_s:
            self._save()

    def _grow(self):
        """Doubles every buffer's capacity, preserving existing rows. Plain
        np.resize() is NOT used here -- it tiles/repeats old data to fill
        the new (larger) size rather than zero-padding, which would corrupt
        already-recorded rows; allocating fresh zero arrays and copying the
        old content in is the correct way to grow in place.
        """
        new_cap = self._cap * 2

        def grow(arr):
            new = np.zeros((new_cap,) + arr.shape[1:])
            new[: self._cap] = arr
            return new

        self._t = grow(self._t)
        self._base_pose_hist = grow(self._base_pose_hist)
        self._base_vel_hist = grow(self._base_vel_hist)
        self._q_hist = grow(self._q_hist)
        self._qdot_hist = grow(self._qdot_hist)
        self._tau_hist = grow(self._tau_hist)
        self._cap = new_cap

    def _save(self):
        """Writes the .npz and flips `done` -- does NOT touch rclpy/executor
        lifecycle itself. Calling rclpy.shutdown() from inside a callback
        that a blocking rclpy.spin(node) is currently running (and then
        destroy_node() afterwards, in a `finally`) was tried first and left
        the process hung indefinitely despite this method completing and
        logging successfully -- shutdown-from-within-a-callback appears to
        leave rclpy.spin()'s executor in a state where it neither returns
        nor raises. main()'s own manual spin_once() loop owns shutdown now,
        in the correct order (destroy_node() before rclpy.shutdown(), not
        after), which does not hang.
        """
        self._saved = True
        n = self._n
        np.savez(
            self.out_path,
            time=self._t[:n],
            base_pose=self._base_pose_hist[:n],
            base_vel=self._base_vel_hist[:n],
            q=self._q_hist[:n],
            qdot=self._qdot_hist[:n],
            tau=self._tau_hist[:n],
            arm_joint_names=np.array(self.arm_joint_names),
        )
        self.get_logger().info(f"pinn_recorder: saved {n} rows to {self.out_path}")
        self.done = True


def main():
    rclpy.init(args=sys.argv)
    node = PinnRecorderNode()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

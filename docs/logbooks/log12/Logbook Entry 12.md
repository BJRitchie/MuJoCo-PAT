# Logbook Entry 12

First meeting: October 14, 2026
Organization: USYD
Supervisor: Xiaofeng Wu

## Objectives

<aside>
💡

Bullet-point summary of goals and tasks for the current period, based on the previous discussion with the supervisor and the scheduled activities. (max 50 words)

</aside>

- Disturbance observer and sensor models
- Base state estimator
- Trajectory following and teleoperation
- Baseline controllers (OMPC, SMC) and a thesis evaluation pipeline

## Activities Performed

<aside>
💡

Describe what was done, including coding, testing, designing, researching, meetings,
etc.) (max 50 words)

</aside>

- Sensor suite with configurable noise and disturbance models
- Momentum disturbance observer (real-time compatible); drag recovery 95-106%, model error costs 2.14 mm of which feedforward recovers 1.96 mm
- Extracted `pat_rt` realtime helpers; IMU-driven error-state EKF with delayed-measurement replay, run by `nav_node` on its own thread, with almost no cost to arm accuracy
- Trajectory following with a preview window; teleoperation node
- Ported baseline controllers (OMPC, SMC) and refactored for a common structure
- Thesis evaluation pipeline, flight-sensing state sources, joint Kalman filter, `pat_sensors`
- AVGS-type relative navigation sensor; replotted figures

## Meeting Minutes

## Data / Results / Plots / Diagrams / Equations

<aside>
💡

Plots, pictures, logs, diagrams or equations relevant to the work (max 100 words)

</aside>

- Thesis figures in `docs/assets/thesis/`

## Observations and Reflections

<aside>
💡

What worked, what didn’t, any insights gained, changes in direction, challenges, new literature etc. (max 100 words)

</aside>

- The AVGS-type sensor improved base position from 12.5 mm to 2.4 mm; the estimator is limited by the sensor, not its tuning
- Total momentum is conserved on the frictionless table; the base yaw drift is the reaction to the arms
- Feedforward from the observer is bounded by velocity-sensing noise

## Problems Encountered

<aside>
💡

Any bugs, hardware issues, theoretical blocks, etc. (max 50 words)

</aside>

- Cold-start EKF dominated errors until a warm-up phase was added
- A bench bug silently ignored zero base damping
- LMPC cannot do the 3x circle without entering safe mode

## Next Steps / To-Do List (max 50 words)

<aside>
💡

List specific tasks for the next session. Any required resources or support needed (e.g.
hardware, software, tech. support, etc.)

</aside>

- Draft thesis report (due 30 Oct)
- Deploy and benchmark on the Jetson
- Wire the joint Kalman filter into the live arm node

## Appendices / Reference Notes (optional)

<aside>
💡

(Include any resources, references, links, more results or literature notes here)

</aside>

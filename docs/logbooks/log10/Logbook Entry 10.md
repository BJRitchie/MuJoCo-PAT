# Logbook Entry 10

First meeting: September 16, 2026
Organization: USYD
Supervisor: Xiaofeng Wu

## Objectives

<aside>
💡

Bullet-point summary of goals and tasks for the current period, based on the previous discussion with the supervisor and the scheduled activities. (max 50 words)

</aside>

- Porting over to PAT simulator (more realistic setup)
- Optimal trajectory generator
- Tests on hardware

## Activities Performed

<aside>
💡

Describe what was done, including coding, testing, designing, researching, meetings,
etc.) (max 50 words)

</aside>

- PAT development/controller port
    - Iteratively linearised MPC working
    - SQP MPC is also working now - tracked down two bugs that were resulting in some instability
- PINN data can also now be generated from this sim - sent a batch to Saad to see if the PINN can be trained on that
- STLs can be loaded in the sim, sim also supports GPU rendering now (very smooth panning)
    - Helped with controller stability bc the stl rendering has been handed off from CPU to GPU
- Met with George to discuss access to the Jetson

## Meeting Minutes

### Week 7 Meeting

- Still working on the port over to PAT sim
    - NMPC is working, trying to set up a script to generate data for PINN training
        - Required non-trivial tuning to reduce jittering
        - Iterative has drift in the long term
        - Full nonlinear is still a bit jittery - long solve times are impeding on performance
    - Haven’t heard from Saad about how the PINN training is going based on the basilisk sim, will follow up when I have data from the PAT sim
- Platform:
    - Cleaner hasn’t replied
- George says he’ll leave 14 oct → 29 oct
- Xiaofeng leave 12th → 20th

### Week 8

- STLs now rendering in the PAT simulator
- Platform nearly ready
    - Just need to move the platform in
- Emphasis on deploying controllers with a proper real-time setup

### Meeting with George (Sep 17)

- Setup on the PAT system - can I be added to the github/ see what its like?
- Controlling the out-of-plane joints (PID controllers?)
    - PD (assume that it is locked)
    - Mostly just microvibration (op on 1000-2000Hz)
- XML file - are the arms mounted properly?
    - Arms are on the sides
    - George sent the zip
- Best way to get work on the Nvidia Jetson?
    - Controller level - on the torque level (dynamics)
        - Piper isnt very good
        - Kinematics level its alright
        - Xiaofeng arranging cleaners for the cleanroom (platform in the cleanroom in ~2 weeks)
        - Control frequency?
            - Can validate by mapping the position/velocity profile
            - HW test on kin level
            - Sim test
    - Timeline:
        - Hook the spare arm up to my laptop
            - Install SDK for arm
            - They have the CAN→USB
            - (not on the platform)
        - Put on the platform
            - Previously: had a script of the trajectory
            - ROSbag the joint information
        - Come in sometime next week to play on the robotic arm
            - Kin level trajectory ready
        - Testing Jetson?
            - George will let me know (he’ll look at it over the next week)
    - Pilot the Jetson over VPN
    - Use DGX to run the controller
        - Scale relative to the operating speed
    - Easiest way is to precompile simulation data to profile the speed
        - Feed data to the jetson and check exec speed
        - 
- Path planners - minimising base disturbances
    - Dual arm and single arm are very similar
    - Instead of one kinematics tree, you’re working with 2 kin trees
- 14th → 30th Oct
    - 5th Oct work

## Data / Results / Plots / Diagrams / Equations

<aside>
💡

Plots, pictures, logs, diagrams or equations relevant to the work (max 100 words)

</aside>

- Live plotjuggler plots (note these screenshots are all taken at different times);
    - Tab 1:
        
        ![image.png](Logbook%20Entry%2010/image.png)
        
    - Tab 2:
        
        ![image.png](Logbook%20Entry%2010/image%201.png)
        
    - Tab 3:
        
        ![image.png](Logbook%20Entry%2010/image%202.png)
        
- 

## Observations and Reflections

<aside>
💡

What worked, what didn’t, any insights gained, changes in direction, challenges, new literature etc. (max 100 words)

</aside>

- Migration went relatively smoothly, suggesting that developing in BSK first was a good idea because its single-threaded
- The more realistic setup of ROS with nodes running concurrently is revealing some issues in the controllers that weren’t showing up in BSK
- Piper arms supposedly aren’t good with torque-control so I can only really test trajectories on the hardware (pos/vel setpoints) which kind of makes hardware tests irrelevant
    - Will focus more effort on just having the controller running on the Jetson and link it into my simulators

## Problems Encountered

<aside>
💡

Any bugs, hardware issues, theoretical blocks, etc. (max 50 words)

</aside>

- Small bugs like specific tuning parameters not mapping properly, setpoint bugs, forgetting a Cv term in the dynamicsAt() function which was throwing off the SQP NMPC algorithm etc

## Next Steps / To-Do List (max 50 words)

<aside>
💡

List specific tasks for the next session. Any required resources or support needed (e.g.
hardware, software, tech. support, etc.)

</aside>

- Adaption to support real-time, then deployment in a real-time kernel
- Deployment on the Jetson
- Benchmarking on the Jetson
- Arm control via the Piper SDK

## Appendices / Reference Notes (optional)

<aside>
💡

(Include any resources, references, links, more results or literature notes here)

</aside>
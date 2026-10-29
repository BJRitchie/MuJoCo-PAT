# Logbook Entry 11

First meeting: September 30, 2026
Organization: USYD
Supervisor: Xiaofeng Wu

## Objectives

<aside>
💡

Bullet-point summary of goals and tasks for the current period, based on the previous discussion with the supervisor and the scheduled activities. (max 50 words)

</aside>

- Make the NMPC real-time safe (tick under the 10 ms budget)
- Benchmark timing, including on aarch64 (Orin proxy)
- Settle the constraint configuration

## Activities Performed

<aside>
💡

Describe what was done, including coding, testing, designing, researching, meetings,
etc.) (max 50 words)

</aside>

- Timing tools and investigations; NMPC tuned to 99.997% of ticks under 10 ms
- Cut per-tick heap allocations from 1369 to 351, then to 1
- Moved outputs and logging off the control tick via lock-free queues; SCHED_FIFO control thread halves p95 under load
- Added `nmpc_bench` for the Orin; all tests pass on the DGX
- Soft box with hard torque rows chosen after a sweep of the aarch64 soft/soft failure
- Renamed the package to `pat_arm_control`

## Meeting Minutes

### Week 9 Meeting

- Real-time tests conducted
    - 99.997% of ticks are under my 10ms budget
        - Without any of the full real-time tools
    - Is this sufficient to move forward
- Research suggests Orin will be 1.5-3.0x the cost per tick
- Granite bench is cleaned
    - Getting calibrated today
- Consider state estimator in the control loop?
    - Kalman filter should be sufficient
    - Consider pipelining the architecture
        - 1st stage - sensing stage - raw data processing
        - 2nd stage - sensor fusion state estimation
        - 3rd stage - control
    - Means I can test time budget for my own controller, and avoid the inclusion of sensor fusion
- Draft by 30th Oct for thesis report
- Can I get quantitative measurement of the reaction torque?
- Want to know
    - Have simulation
    - What is the reaction from the platform POV
    - Want to minimise reaction on the platform

## Data / Results / Plots / Diagrams / Equations

<aside>
💡

Plots, pictures, logs, diagrams or equations relevant to the work (max 100 words)

</aside>

- Timing Run Results
    
    ```bash
    dev@ben-XPS-15-9530:/ws$ ./tools/nmpc_timing/run_trial.sh -d 300 -o /tmp/base.log -T data/nmpc_timing/trace.csv
    tools/nmpc_timing/plot_trace.py --bins 14 --phases --top 3 data/nmpc_timing/trace_left.csv
    running 300s trial -> /tmp/base.log
      params: /ws/install/pat_arm_nmpc/share/pat_arm_nmpc/config/nmpc.yaml
      trace:  data/nmpc_timing/trace.csv (_left / _right)
    done: 618 report windows captured
    
    data/nmpc_timing/trace_left.csv
      arm=left  period_us=10000  N=40  Ts=0.025  warm_start=1  qp_max_iter=100
      30923 ticks over 309.3 s
      tick_us distribution, every tick
            1420-2696     ############################################   15588  50.4% (cum  50.4%)
            2696-3973     ####################################           12859  41.6% (cum  92.0%)
            3973-5249     ######                                          2025   6.5% (cum  98.5%)
            5249-6526     #                                                339   1.1% (cum  99.6%)
            6526-7802     .                                                 64   0.2% (cum  99.8%)
            7802-9079     .                                                 27   0.1% (cum  99.9%)
            9079-10355    .                                                  9   0.0% (cum 100.0%)  <-- deadline
           10355-11632    .                                                  5   0.0% (cum 100.0%)
           11632-12909    .                                                  3   0.0% (cum 100.0%)
           12909-14185    .                                                  3   0.0% (cum 100.0%)
           14185-15462                                                       0   0.0% (cum 100.0%)
           15462-16738                                                       0   0.0% (cum 100.0%)
           16738-18015                                                       0   0.0% (cum 100.0%)
           18015-19291    .                                                  1   0.0% (cum 100.0%)
      tick_us: min 1420  p50 2684  p90 3845  p99 5552  p99.9 8427  max 19291 us
      over the 10000 us period: 14 (0.05%)
      qp_iters: mean 8.23  max 40   retries 0   damped ticks 0
      mean tick by phase
                    lambda                                                   3.1 us   0.1%
                   qpsolve ###################################            2236.3 us  79.8%
          rollout_dynamics ####                                            251.8 us   9.0%
         rollout_integrate ##                                              154.6 us   5.5%
                      pinv #                                                36.2 us   1.3%
               build_model #                                                35.3 us   1.3%
                    defect                                                   7.0 us   0.2%
              trust_region                                                   1.6 us   0.1%
                  finalize                                                   0.7 us   0.0%
              unattributed                                                  75.6 us   2.7%
                     TOTAL                                                2802.4 us
      3 slowest ticks
        t= 212.54s  tick=   19291 us  iters=40 status=0 retried=0  qpsolve=18800
        t=  12.75s  tick=   13141 us  iters=13 status=0 retried=0  qpsolve=10649 rollout=1062 rollout=704
        t=  12.76s  tick=   12978 us  iters=13 status=0 retried=0  qpsolve=11031 rollout=857
    ```
    
- Timing run histogram (right arm)
    
    ```
    dev@ben-XPS-15-9530:/ws$ tools/nmpc_timing/plot_trace.py data/nmpc_timing/trace_right.csv 
    
    data/nmpc_timing/trace_right.csv
      arm=right  period_us=10000  N=40  Ts=0.025  warm_start=1  qp_max_iter=100
      30927 ticks over 309.3 s
      tick_us distribution, every tick
        range [us]                                                      count  share   CDF
            1482-2041     ##########################################      7694  24.9%   24.9%
            2041-2600     ############################################    8128  26.3%   51.2%
            2600-3159     ###########################################     8006  25.9%   77.0%
            3159-3718     ########################                        4489  14.5%   91.6%
            3718-4277     ########                                        1433   4.6%   96.2%
            4277-4836     ###                                              571   1.8%   98.0%
            4836-5395     #                                                237   0.8%   98.8%
            5395-5954     #                                                123   0.4%   99.2%
            5954-6513     .                                                 85   0.3%   99.5%
            6513-7072     .                                                 56   0.2%   99.7%
            7072-7631     .                                                 35   0.1%   99.8%
            7631-8189     .                                                 18   0.1%   99.8%
            8189-8748     .                                                 17   0.1%   99.9%
            8748-9307     .                                                  8   0.0%   99.9%
            9307-9866     .                                                 11   0.0%   99.9%
            9866-10425    .                                                 10   0.0%  100.0%  <-- deadline
           10425-10984    .                                                  5   0.0%  100.0%
           10984-11543                                                       0   0.0%  100.0%
           11543-12102                                                       0   0.0%  100.0%
           12102-12661    .                                                  1   0.0%  100.0%
      tick_us: min 1482  p50 2569  p90 3615  p99 5607  p99.9 8954  max 12661 us
      over the 10000 us period: 16 (0.05%)
      qp_iters: mean 7.94  max 17   retries 0   damped ticks 0
    ```
    
- Timing run histogram (left arm)
    
    ```
    dev@ben-XPS-15-9530:/ws$ tools/nmpc_timing/plot_trace.py data/nmpc_timing/trace_left.csv
    
    data/nmpc_timing/trace_left.csv
      arm=left  period_us=10000  N=40  Ts=0.025  warm_start=1  qp_max_iter=100
      30923 ticks over 309.3 s
      tick_us distribution, every tick
        range [us]                                                      count  share   CDF
            1420-2313     ############################################   11818  38.2%   38.2%
            2313-3207     #####################################           9871  31.9%   70.1%
            3207-4100     ###########################                     7185  23.2%   93.4%
            4100-4994     #####                                           1422   4.6%   98.0%
            4994-5888     ##                                               420   1.4%   99.3%
            5888-6781     .                                                110   0.4%   99.7%
            6781-7675     .                                                 43   0.1%   99.8%
            7675-8568     .                                                 27   0.1%   99.9%
            8568-9462     .                                                 11   0.0%   99.9%
            9462-10355    .                                                  4   0.0%  100.0%  <-- deadline
           10355-11249    .                                                  5   0.0%  100.0%
           11249-12143    .                                                  1   0.0%  100.0%
           12143-13036    .                                                  4   0.0%  100.0%
           13036-13930    .                                                  1   0.0%  100.0%
           13930-14823                                                       0   0.0%  100.0%
           14823-15717                                                       0   0.0%  100.0%
           15717-16611                                                       0   0.0%  100.0%
           16611-17504                                                       0   0.0%  100.0%
           17504-18398                                                       0   0.0%  100.0%
           18398-19291    .                                                  1   0.0%  100.0%
      tick_us: min 1420  p50 2684  p90 3845  p99 5552  p99.9 8427  max 19291 us
      over the 10000 us period: 14 (0.05%)
      qp_iters: mean 8.23  max 40   retries 0   damped ticks 0
    ```
    

After successfully running on “performance” mode; 

- Timing run histogram
    
    ```
    dev@ben-XPS-15-9530:/ws$ tools/nmpc_timing/plot_trace.py /tmp/perf_trace_left.csv 
    
    /tmp/perf_trace_left.csv
      arm=left  period_us=10000  N=30  Ts=0.033  warm_start=1  qp_max_iter=60
      30932 ticks over 309.3 s
      tick_us distribution, every tick
        range [us]                                                        count   share       CDF
             773-944      ############################################   13778  44.543%   44.543%
             944-1114     ################################               10009  32.358%   76.901%
            1114-1284     ##########                                      3285  10.620%   87.521%
            1284-1455     ######                                          1748   5.651%   93.172%
            1455-1625     ##                                               747   2.415%   95.587%
            1625-1795     #                                                405   1.309%   96.896%
            1795-1966     #                                                365   1.180%   98.076%
            1966-2136     #                                                255   0.824%   98.901%
            2136-2306     .                                                136   0.440%   99.340%
            2306-2477     .                                                 84   0.272%   99.612%
            2477-2647     .                                                 50   0.162%   99.774%
            2647-2817     .                                                 18   0.058%   99.832%
            2817-2988     .                                                 30   0.097%   99.929%
            2988-3158     .                                                 10   0.032%   99.961%
            3158-3328     .                                                  3   0.010%   99.971%
            3328-3499     .                                                  3   0.010%   99.981%
            3499-3669     .                                                  2   0.006%   99.987%
            3669-3839                                                        0   0.000%   99.987%
            3839-4010     .                                                  2   0.006%   99.994%
            4010-4180     .                                                  2   0.006%  100.000%
      tick_us: min 773  p50 959  p90 1341  p99 2168  p99.9 2956  max 4180 us
      over the 10000 us period: 0 (0.00%)
      qp_iters: mean 8.12  max 25   retries 0   damped ticks 0
    dev@ben-XPS-15-9530:/ws$ tools/nmpc_timing/plot_trace.py /tmp/perf_trace_right.csv 
    
    /tmp/perf_trace_right.csv
      arm=right  period_us=10000  N=30  Ts=0.033  warm_start=1  qp_max_iter=60
      30932 ticks over 309.3 s
      tick_us distribution, every tick
        range [us]                                                        count   share       CDF
             791-1019     ############################################   20388  65.912%   65.912%
            1019-1248     ###############                                 7174  23.193%   89.105%
            1248-1477     ####                                            2022   6.537%   95.642%
            1477-1706     #                                                532   1.720%   97.362%
            1706-1934     #                                                356   1.151%   98.513%
            1934-2163     #                                                241   0.779%   99.292%
            2163-2392     .                                                113   0.365%   99.657%
            2392-2621     .                                                 61   0.197%   99.855%
            2621-2849     .                                                 33   0.107%   99.961%
            2849-3078     .                                                  7   0.023%   99.984%
            3078-3307     .                                                  1   0.003%   99.987%
            3307-3536     .                                                  2   0.006%   99.994%
            3536-3764                                                        0   0.000%   99.994%
            3764-3993     .                                                  1   0.003%   99.997%
            3993-4222                                                        0   0.000%   99.997%
            4222-4450                                                        0   0.000%   99.997%
            4450-4679                                                        0   0.000%   99.997%
            4679-4908                                                        0   0.000%   99.997%
            4908-5137                                                        0   0.000%   99.997%
            5137-5365     .                                                  1   0.003%  100.000%
      tick_us: min 791  p50 966  p90 1265  p99 2034  p99.9 2686  max 5365 us
      over the 10000 us period: 0 (0.00%)
      qp_iters: mean 7.87  max 16   retries 0   damped ticks 0
    dev@ben-XPS-15-9530:/ws$ 
    ```
    

With dedicated RT thread (`std::thread`): 

- TIming run:
    
    ```
    dev@ben-XPS-15-9530:/ws$ ./tools/nmpc_timing/run_trial.sh -g -d 300 -o /tmp/base.log -T data/nmpc_timing/trace.csv
    running 300s trial -> /tmp/base.log
      params: /ws/install/pat_arm_nmpc/share/pat_arm_nmpc/config/nmpc.yaml
      trace:  data/nmpc_timing/trace.csv (_left / _right)
      governor: performance
    done: 618 report windows captured
    dev@ben-XPS-15-9530:/ws$ tools/nmpc_timing/plot_trace.py data/nmpc_timing/trace_left.csv 
    
    data/nmpc_timing/trace_left.csv
      arm=left  period_us=10000  N=30  Ts=0.033  warm_start=1  qp_max_iter=60
      30937 ticks over 309.4 s
      tick_us distribution, every tick
        range [us]                                                        count   share       CDF
             776-1109     ############################################   24455  79.048%   79.048%
            1109-1441     ##########                                      5532  17.882%   96.929%
            1441-1774     #                                                779   2.518%   99.447%
            1774-2107     .                                                134   0.433%   99.880%
            2107-2440     .                                                 23   0.074%   99.955%
            2440-2772     .                                                  7   0.023%   99.977%
            2772-3105                                                        0   0.000%   99.977%
            3105-3438     .                                                  3   0.010%   99.987%
            3438-3771                                                        0   0.000%   99.987%
            3771-4103     .                                                  2   0.006%   99.994%
            4103-4436                                                        0   0.000%   99.994%
            4436-4769                                                        0   0.000%   99.994%
            4769-5101                                                        0   0.000%   99.994%
            5101-5434     .                                                  1   0.003%   99.997%
            5434-5767                                                        0   0.000%   99.997%
            5767-6100                                                        0   0.000%   99.997%
            6100-6432                                                        0   0.000%   99.997%
            6432-6765                                                        0   0.000%   99.997%
            6765-7098                                                        0   0.000%   99.997%
            7098-7431     .                                                  1   0.003%  100.000%
      tick_us: min 776  p50 934  p90 1252  p99 1647  p99.9 2205  max 7431 us
      over the 10000 us period: 0 (0.00%)
      qp_iters: mean 8.12  max 45   retries 0   damped ticks 0
    dev@ben-XPS-15-9530:/ws$ tools/nmpc_timing/plot_trace.py data/nmpc_timing/trace_right.csv 
    
    data/nmpc_timing/trace_right.csv
      arm=right  period_us=10000  N=30  Ts=0.033  warm_start=1  qp_max_iter=60
      30937 ticks over 309.4 s
      tick_us distribution, every tick
        range [us]                                                        count   share       CDF
             776-957      ############################################   17799  57.533%   57.533%
             957-1139     ####################                            8011  25.895%   83.428%
            1139-1321     #########                                       3499  11.310%   94.738%
            1321-1502     ###                                             1115   3.604%   98.342%
            1502-1684     #                                                334   1.080%   99.421%
            1684-1866     .                                                 82   0.265%   99.686%
            1866-2047     .                                                 50   0.162%   99.848%
            2047-2229     .                                                 20   0.065%   99.913%
            2229-2411     .                                                  9   0.029%   99.942%
            2411-2592     .                                                  6   0.019%   99.961%
            2592-2774     .                                                  4   0.013%   99.974%
            2774-2956     .                                                  3   0.010%   99.984%
            2956-3137     .                                                  3   0.010%   99.994%
            3137-3319                                                        0   0.000%   99.994%
            3319-3501     .                                                  1   0.003%   99.997%
            3501-3682                                                        0   0.000%   99.997%
            3682-3864                                                        0   0.000%   99.997%
            3864-4046                                                        0   0.000%   99.997%
            4046-4227                                                        0   0.000%   99.997%
            4227-4409     .                                                  1   0.003%  100.000%
      tick_us: min 776  p50 937  p90 1227  p99 1588  p99.9 2179  max 4409 us
      over the 10000 us period: 0 (0.00%)
      qp_iters: mean 7.87  max 16   retries 0   damped ticks 0
    ```
    

## Observations and Reflections

<aside>
💡

What worked, what didn’t, any insights gained, changes in direction, challenges, new literature etc. (max 100 words)

</aside>

- Allocations and logging on the control thread were the main source of tail latency; removing them mattered more than solver tuning
- Running the control loop on a SCHED_FIFO thread greatly reduces timing variance under load
- Soft torque rows are fragile near infeasibility on aarch64, so the default is a soft box with hard torque rows

## Problems Encountered

<aside>
💡

Any bugs, hardware issues, theoretical blocks, etc. (max 50 words)

</aside>

- Soft/soft constraints collapsed on aarch64 (21.7% availability against 96.7%)
- Occasional solve spikes over 10 ms (0.05% of ticks before tuning)
- A stalled subscriber could starve the control loop; measured and the torque QoS made explicit

## Next Steps / To-Do List (max 50 words)

<aside>
💡

List specific tasks for the next session. Any required resources or support needed (e.g.
hardware, software, tech. support, etc.)

</aside>

- Disturbance observer for reaction/contact forces
- Sensor models and a state estimator for the base
- Trajectory following; deploy on the Jetson

## Appendices / Reference Notes (optional)

<aside>
💡

(Include any resources, references, links, more results or literature notes here)

</aside>
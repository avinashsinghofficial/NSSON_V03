Dataset: qos_all_runs_balanced_complete.csv
Design: baseline_sdn_iot vs nsson_full
Loads: light=2 Mbit/s, moderate=12 Mbit/s, heavy=19 Mbit/s
Bottleneck: 20 Mbit/s, delay=3 ms, max_queue_size=30
Repetitions per condition: 10
Total complete observations: 60
Inference flow: qc (10.20.0.1) -> qs (10.20.0.2), TCP/8090
Background flow: qb1 (10.20.0.11) -> qb2 (10.20.0.12), UDP/5001
Controller: Ryu 4.34, OpenFlow 1.3, port 6633
Eventlet: 0.39.1; dnspython: 2.8.0; EVENTLET_NO_GREENDNS=yes

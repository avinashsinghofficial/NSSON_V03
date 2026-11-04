Validated multi-slice SLA experiment
Mode: slicing_sla
Load: light (2 Mbit/s UDP background)
Repetition: 1
Topology: qc + qb1 -- s1 -- s2 -- qs + qb2
Queues: 0 background, 1 autonomous, 2 healthcare, 3 industrial
Controller: nsson_multislice_sla_controller.py

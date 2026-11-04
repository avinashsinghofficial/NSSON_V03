#!/usr/bin/env bash
set -euo pipefail

echo "========== PAIR 01 =========="
/usr/bin/python3 "/home/avi/NSSON/buffer_qos_replicates_jjvvwp6u/pair_01/pilot.py" 2>&1 | tee "/home/avi/NSSON/buffer_qos_replicates_jjvvwp6u/pair_01/console.log"
sleep 3

echo "========== PAIR 02 =========="
/usr/bin/python3 "/home/avi/NSSON/buffer_qos_replicates_jjvvwp6u/pair_02/pilot.py" 2>&1 | tee "/home/avi/NSSON/buffer_qos_replicates_jjvvwp6u/pair_02/console.log"
sleep 3

echo "========== PAIR 03 =========="
/usr/bin/python3 "/home/avi/NSSON/buffer_qos_replicates_jjvvwp6u/pair_03/pilot.py" 2>&1 | tee "/home/avi/NSSON/buffer_qos_replicates_jjvvwp6u/pair_03/console.log"
sleep 3

echo "========== PAIR 04 =========="
/usr/bin/python3 "/home/avi/NSSON/buffer_qos_replicates_jjvvwp6u/pair_04/pilot.py" 2>&1 | tee "/home/avi/NSSON/buffer_qos_replicates_jjvvwp6u/pair_04/console.log"
sleep 3

echo "========== PAIR 05 =========="
/usr/bin/python3 "/home/avi/NSSON/buffer_qos_replicates_jjvvwp6u/pair_05/pilot.py" 2>&1 | tee "/home/avi/NSSON/buffer_qos_replicates_jjvvwp6u/pair_05/console.log"
sleep 3

echo "========== PAIR 06 =========="
/usr/bin/python3 "/home/avi/NSSON/buffer_qos_replicates_jjvvwp6u/pair_06/pilot.py" 2>&1 | tee "/home/avi/NSSON/buffer_qos_replicates_jjvvwp6u/pair_06/console.log"
sleep 3

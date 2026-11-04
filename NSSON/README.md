# NSSON Best Upgraded Structure

SDN-first neuromorphic surveillance pipeline for workstation + Jetson Nano DK + remote verifier under Ryu controller orchestration.

## Run order
1. Start Ryu controller.
2. Start edge worker or Jetson worker.
3. Start Jetson node agent.
4. Optionally start remote worker.
5. Start workstation demo.

## Core flow
- Webcam frame -> proposal detector -> hand ROI generation -> SNN gate + local verifier.
- ROI metadata -> Ryu controller decision API.
- Controller chooses local / Jetson / remote + QoS class.
- ROI sent to selected node.
- Result fused, masked, logged.

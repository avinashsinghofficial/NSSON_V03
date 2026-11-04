#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$HOME/NSSON}"
cd "$ROOT"

cat > camnsson/sdn/decision_client.py <<'EOF'
import requests

class ControllerDecisionClient:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip('/')
    def decide(self, payload):
        try:
            r = requests.post(self.base_url + '/camnsson/decision', json=payload, timeout=2)
            r.raise_for_status()
            return r.json()
        except Exception:
            local_score = float(payload.get('local_score', 0.0))
            spike_score = float(payload.get('spike_score', 0.0))
            roi_bytes = int(payload.get('roi_bytes', 0))
            urgency = max(local_score, spike_score)
            if urgency > 0.80:
                target = 'remote'
            elif roi_bytes < 120000:
                target = 'jetson'
            else:
                target = 'local'
            qos_class = 'critical' if urgency > 0.80 else 'priority' if urgency > 0.60 else 'best_effort'
            return {'target': target, 'qos_class': qos_class, 'route_id': f'{target}-{qos_class}', 'telemetry': {'queue_util': 0.5, 'link_delay_ms': 20}, 'jetson_url': '', 'remote_url': ''}
EOF

cat > apps/jetson_node_agent.py <<'EOF'
import time, socket, requests, psutil
from camnsson.utils.config import load_config

class NodeAgent:
    def __init__(self, controller_url, node_id, role, service_url, heartbeat_sec=2):
        self.controller_url = controller_url.rstrip('/')
        self.node_id = node_id
        self.role = role
        self.service_url = service_url
        self.heartbeat_sec = heartbeat_sec
    def payload(self):
        return {'node_id': self.node_id, 'role': self.role, 'url': self.service_url, 'hostname': socket.gethostname(), 'status': 'up', 'load': psutil.cpu_percent(interval=0.1)/100.0, 'mem_used': psutil.virtual_memory().percent/100.0, 'rtt_ms': 5}
    def run(self):
        try:
            requests.post(self.controller_url + '/camnsson/register', json=self.payload(), timeout=2)
        except Exception:
            pass
        while True:
            try:
                requests.post(self.controller_url + '/camnsson/heartbeat', json=self.payload(), timeout=2)
            except Exception:
                pass
            time.sleep(self.heartbeat_sec)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); args = p.parse_args(); cfg = load_config(args.config)
    NodeAgent(cfg['controller_url'], cfg['node_id'], cfg['role'], cfg['service_url'], cfg.get('heartbeat_sec',2)).run()
EOF

cat > apps/run_demo.py <<'EOF'
import time, cv2
from camnsson.utils.config import load_config
from camnsson.utils.io import append_csv
from camnsson.utils.system import system_load
from camnsson.capture.webcam import WebcamStream
from camnsson.models.proposal_detector import ProposalDetector
from camnsson.models.snn_gate import SNNGate
from camnsson.models.verifier import ROIModel
from camnsson.roi.hand_roi import propose_hand_rois
from camnsson.roi.roi_selector import select_best_rois
from camnsson.policy.decision_features import build_features
from camnsson.policy.offload import decide_target
from camnsson.net.client import EdgeClient
from camnsson.sdn.decision_client import ControllerDecisionClient
from camnsson.fusion.fuser import fuse_scores
from camnsson.utils.image import crop_box, draw_box, blur_region, encode_jpg

FIELDS = ['ts','roi_index','local_score','spike_score','decision_target','qos_class','edge_score','fused_score','danger','edge_latency_ms']

def main(cfg):
    cam = WebcamStream(cfg['camera_index'], cfg['frame']['width'], cfg['frame']['height'])
    proposal = ProposalDetector(cfg['models']['proposal_model'])
    local_gate = SNNGate(cfg['models'].get('snn_checkpoint'), device='cpu')
    local_verifier = ROIModel(cfg['models']['verifier_model'], cfg['models'].get('verifier_task','classify'))
    controller = ControllerDecisionClient(cfg['controller_url'])
    jetson = EdgeClient(cfg['edge_url'])
    remote = EdgeClient(cfg['remote_url'])
    while True:
        ok, frame = cam.read()
        if not ok:
            break
        vis = frame.copy()
        dets = proposal.infer(frame)
        people = [d for d in dets if d['label'].lower() == 'person']
        for det in people:
            draw_box(vis, det['box'], (0,255,0), f"person {det['score']:.2f}")
        rois = select_best_rois(propose_hand_rois(frame, people, cfg['roi']['expand_ratio'], cfg['roi']['min_size'], cfg['roi']['max_size']))
        for idx, roi in enumerate(rois):
            crop = crop_box(frame, roi['box'])
            if crop.size == 0:
                continue
            gate = local_gate.infer(crop)
            local = local_verifier.infer(crop)
            local_score = float(local['danger_score'])
            feats = build_features(local_score, gate['spike_score'], len(encode_jpg(crop)), system_load())
            feats.update({'roi_index': idx, 'side': roi['side']})
            decision = controller.decide(feats)
            target = decide_target(decision)
            edge_score = None
            edge_latency = 0.0
            if target == 'jetson':
                try:
                    out = jetson.infer(crop, {'roi_index': idx, 'side': roi['side'], 'route_id': decision['route_id']})
                    edge_score = float(out['cnn']['danger_score'])
                    edge_latency = float(out.get('edge_latency_ms', 0.0))
                except Exception:
                    target = 'local'
            elif target == 'remote':
                try:
                    out = remote.infer(crop, {'roi_index': idx, 'side': roi['side'], 'route_id': decision['route_id']})
                    edge_score = float(out['cnn']['danger_score'])
                    edge_latency = float(out.get('edge_latency_ms', 0.0))
                except Exception:
                    target = 'local'
            fused = fuse_scores(local_score, edge_score, gate['spike_score'])
            danger = fused >= cfg['policy']['danger_alert_threshold']
            draw_box(vis, roi['box'], (0,0,255) if danger else (0,255,255), f"{roi['side']} {target} {decision.get('qos_class','na')} fs={fused:.2f}")
            if danger:
                vis = blur_region(vis, roi['box'])
            append_csv(cfg['logging']['metrics_csv'], {'ts': time.time(), 'roi_index': idx, 'local_score': local_score, 'spike_score': gate['spike_score'], 'decision_target': target, 'qos_class': decision.get('qos_class','na'), 'edge_score': edge_score if edge_score is not None else '', 'fused_score': fused, 'danger': int(danger), 'edge_latency_ms': edge_latency}, FIELDS)
        cv2.imshow('CamNSSON SDN Demo', vis)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); args = p.parse_args(); main(load_config(args.config))
EOF

echo 'patched controller fallback and resilient agent/demo'

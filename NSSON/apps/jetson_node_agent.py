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

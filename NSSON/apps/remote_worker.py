from apps.edge_worker import app
from camnsson.utils.config import load_config
from camnsson.jetson.worker import JetsonWorker
import apps.edge_worker as ew

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); args = p.parse_args()
    cfg = load_config(args.config)
    ew.worker = JetsonWorker(cfg['models']['verifier_model'], cfg['models'].get('verifier_task','classify'), cfg['models'].get('snn_checkpoint'), cfg.get('runtime',{}).get('device','cpu'))
    app.run(host=cfg['host'], port=cfg['port'])

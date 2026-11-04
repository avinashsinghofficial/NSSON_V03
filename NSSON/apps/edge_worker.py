import base64
from flask import Flask, request, jsonify
from camnsson.utils.config import load_config
from camnsson.utils.image import decode_jpg_bytes
from camnsson.jetson.worker import JetsonWorker

app = Flask(__name__)
worker = None

@app.route('/infer', methods=['POST'])
def infer():
    data = request.get_json(force=True)
    img = decode_jpg_bytes(base64.b64decode(data['image_b64']))
    out = worker.infer(img)
    out['meta'] = data.get('meta', {})
    return jsonify(out)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); args = p.parse_args()
    cfg = load_config(args.config)
    worker = JetsonWorker(cfg['models']['verifier_model'], cfg['models'].get('verifier_task','classify'), cfg['models'].get('snn_checkpoint'), cfg.get('runtime',{}).get('device','cpu'))
    app.run(host=cfg['host'], port=cfg['port'])

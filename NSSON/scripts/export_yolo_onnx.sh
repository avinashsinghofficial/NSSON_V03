#!/usr/bin/env bash
python - <<'PY'
from ultralytics import YOLO
model = YOLO('models/verifier_yolo_cls.pt')
model.export(format='onnx', imgsz=224)
PY

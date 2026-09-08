"""Exercise the exact Linux runtime with a synthetic JPEG and local CPU inference."""
import argparse
import base64
import ctypes.util
import importlib.metadata as metadata
import json
import re
import sys
from io import BytesIO
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    installed = {d.metadata['Name'].lower().replace('_', '-') for d in metadata.distributions()}
    assert 'ultralytics-opencv-headless' in installed
    assert 'opencv-python-headless' in installed
    assert not installed.intersection({'ultralytics', 'opencv-python', 'opencv-contrib-python', 'opencv-contrib-python-headless'})
    for library in ['GL', 'glib-2.0', 'gio-2.0', 'xml2']:
        assert ctypes.util.find_library(library) is None, library
    assert not Path('/usr/bin/perl').exists()
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    import app.main
    from app.vision.detector import YoloDetector
    from app.vision.frame import validate_frame
    assert torch.version.cuda is None
    torch.set_num_threads(1)
    assert re.search(r'GUI:\s+NONE', cv2.getBuildInformation())
    frame = BytesIO()
    Image.new('RGB', (320, 320), (32, 64, 96)).save(frame, format='JPEG')
    encoded = base64.b64encode(frame.getvalue()).decode('ascii')
    validate_frame(encoded)
    decoded = cv2.imdecode(np.frombuffer(frame.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (320, 320, 3)
    detector = YoloDetector(args.model, 0.5)
    detections = detector.detect(encoded)
    assert isinstance(detections, list)
    print(json.dumps({'status': 'passed', 'synthetic_jpeg': True, 'local_cpu_inference': True,
                      'cv2': cv2.__version__,
                      'ultralytics_headless': metadata.version('ultralytics-opencv-headless')}))


if __name__ == '__main__':
    main()

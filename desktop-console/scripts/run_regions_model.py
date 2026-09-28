"""Run plate capture, vehicle categories and annotated video locally (no training).

Uses the existing green YOLOv8 plate checkpoint and the pretrained IISc UVH-26
Indian-traffic vehicle model. OCR and crop angle correction are on by default;
OCR can be switched off for faster region capture.
"""
import argparse
from hashlib import sha256
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PLATE = ROOT / 'platevision/models/runs/combined_green_v3/weights/best.onnx'
VEHICLE = ROOT / 'platevision/models/indian-vehicles/UVH-26-MV-YOLOv11-S.onnx'
PLATE_SHA = '658d763e2a603cde6269bb45272e87795d46f206aa18c06655de7f54d41437a5'


def configure(*, plate=None, plate_sha=None,
              detector_name='YOLOv8n green — plate capture',
              video_output_dir='video-results-regions'):
    """Configure this process; callers must supply an independently verified hash."""
    plate = PLATE if plate is None else Path(plate)
    plate_sha = PLATE_SHA if plate_sha is None else plate_sha
    if not plate.is_file() or sha256(plate.read_bytes()).hexdigest() != plate_sha:
        raise ValueError('Plate checkpoint missing or changed. Run python scripts/setup_demo_models.py from the repository root.')
    if not VEHICLE.is_file():
        raise ValueError('Vehicle model missing. Run python scripts/setup_demo_models.py from the repository root.')
    # Select both models explicitly; a local .env must not silently disable
    # vehicle analysis or select a different plate checkpoint.
    os.environ.update(
        MODEL_PATH=str(plate), DETECTOR_MODE='onnx',
        DETECTOR_NAME=detector_name, DETECTOR_RETRAINED='true',
        BASELINE_MODEL='true', REGION_CONFIDENCE_THRESHOLD='.25',
        REGION_SUPPLEMENT_PATH='', VEHICLE_CONTEXT_ENABLED='true',
        VEHICLE_PLATE_SEARCH_ENABLED='true', VEHICLE_PLATE_MAX_CROPS='8',
        VEHICLE_MODEL_PATH=str(VEHICLE), VEHICLE_TAXONOMY='uvh',
        VEHICLE_CONFIDENCE_THRESHOLD='.30', VIDEO_OUTPUT_DIR=video_output_dir,
    )


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8003)
    args=parser.parse_args()
    try:
        configure()
    except ValueError as error:
        parser.error(str(error))
    backend=ROOT/'platevision/backend'
    os.chdir(backend)
    sys.path.insert(0,str(backend))
    import uvicorn
    uvicorn.run('app.main:app',host='127.0.0.1',port=args.port,access_log=False)


if __name__=='__main__':
    main()

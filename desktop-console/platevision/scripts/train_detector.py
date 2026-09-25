"""Fine-tune a small YOLO detector and export ONNX weights for PlateVision."""
from pathlib import Path
from ultralytics import YOLO

root = Path(__file__).parents[1]
data = root / "data" / "yolo-indian" / "data.yaml"
output = root / "models" / "runs"
model = YOLO(str(root / "models" / "base" / "yolo11n.pt"))
model.train(data=str(data), epochs=80, imgsz=640, batch=8, patience=20, project=str(output), name="indian_plate", seed=42, device="cpu")
best = output / "indian_plate" / "weights" / "best.pt"
YOLO(str(best)).export(format="onnx", imgsz=640, simplify=True)

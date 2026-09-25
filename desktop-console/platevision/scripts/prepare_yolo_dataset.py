"""Create a leakage-resistant YOLO split from the downloaded Indian plate dataset."""
from pathlib import Path
import shutil

root = Path(__file__).parents[1]
source = root / "data" / "indian-plates"
target = root / "data" / "yolo-indian"
splits = {"train": ["vid-1", "vid-2"], "val": ["vid-3"]}

for split, videos in splits.items():
    for kind in ("images", "labels"):
        (target / split / kind).mkdir(parents=True, exist_ok=True)
    for video in videos:
        for image in (source / video).glob("*.*"):
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            label = image.with_suffix(".txt")
            if not label.exists():
                continue
            name = f"{video}_{image.name}"
            shutil.copy2(image, target / split / "images" / name)
            shutil.copy2(label, target / split / "labels" / Path(name).with_suffix(".txt"))

(target / "data.yaml").write_text(f"path: {target.as_posix()}\ntrain: train/images\nval: val/images\nnames:\n  0: number_plate\n")
print(f"Prepared {target}")

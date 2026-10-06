import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor


def main():
    parser = argparse.ArgumentParser(description="Apply SAM 3 concepts to image crops")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    jobs = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained(
        "facebook/sam3", dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    results = []
    for job in jobs:
        image = np.array(Image.open(job["image"]).convert("RGB"))
        inputs = processor(
            images=Image.fromarray(image),
            text=job["concept"],
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        found = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]

        masks = found["masks"].detach().cpu().numpy().astype(bool)
        scores = found["scores"].detach().float().cpu().tolist()
        overlay = image.astype(np.float32)
        labels = []
        colors = np.random.default_rng(1234)
        for number, mask in enumerate(masks, 1):
            color = colors.integers(40, 256, size=3)
            overlay[mask] = overlay[mask] * 0.35 + color * 0.65
            ys, xs = np.where(mask)
            if len(xs):
                labels.append((number, int(np.median(xs)), int(np.median(ys))))

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        for number, x, y in labels:
            cv2.putText(overlay, str(number), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(overlay, str(number), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 0, 0), 1, cv2.LINE_AA)

        output = args.output / f"{job['id']}_sam3.jpg"
        Image.fromarray(overlay).save(output, quality=96)
        results.append({
            "id": job["id"],
            "concept": job["concept"],
            "count": len(masks),
            "scores": [round(score, 4) for score in scores],
            "image": str(output),
        })

    print(json.dumps(results))


if __name__ == "__main__":
    main()

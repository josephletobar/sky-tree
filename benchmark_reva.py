import base64
import json
import os
import random
import re
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np

from choose_nodes import MODEL
from orchestrate_search import run_search
from video_utils import sample_frames, split_to_n, uniform_samples


DATASET = Path("/data/ReVA")
OUTPUT = Path(__file__).parent / "summaries" / "reva_11_video_row_eval"
SAMPLE_SIZE = 11
SEED = 42
CAPTION_BATCH_SIZE = 12


def choose_sample():
    by_video = {}
    for split in ("train", "val"):
        for qa in json.loads((DATASET / f"{split}.json").read_text())["QA"]:
            by_video.setdefault((split, qa["video_path"]), []).append(qa)

    rng = random.Random(SEED)
    videos = list(by_video)
    rng.shuffle(videos)
    chosen = []
    covered_tasks = set()
    for key in videos:
        qa = rng.choice(by_video[key])
        if qa["task"] not in covered_tasks:
            chosen.append((key[0], qa))
            covered_tasks.add(qa["task"])
    for key in videos:
        if len(chosen) == SAMPLE_SIZE:
            break
        if any(qa["video_path"] == key[1] for _, qa in chosen):
            continue
        chosen.append((key[0], rng.choice(by_video[key])))
    return chosen


def make_tree(total_frames):
    def node(start, end, node_id):
        item = {"start": start, "end": end, "children": [], "summary": ""}
        if end - start > 15:
            item["children"] = [node(a, b, f"{node_id}/{i}")
                                for i, (a, b) in enumerate(split_to_n(start, end, 2))]
        return item

    tree = {"start": 0, "end": total_frames, "summary": "Whole video",
            "children": []}
    tree["children"] = [node(a, b, f"root/{i}")
                        for i, (a, b) in enumerate(split_to_n(0, total_frames, 8))]
    return tree


def indexed_nodes(tree):
    found = []

    def visit(node, node_id):
        if node_id != "root":
            found.append((node_id, node))
        for i, child in enumerate(node.get("children", [])):
            visit(child, f"{node_id}/{i}")
    visit(tree, "root")
    return found


def node_panel(video, node_id, node):
    indices = uniform_samples(node["start"], node["end"], 3)
    encoded = sample_frames(video, node["start"], node["end"], 3)
    frames = []
    for image in encoded:
        frame = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        frame = cv2.resize(frame, (256, 144))
        frames.append(frame)
    panel = np.hstack(frames)
    cv2.rectangle(panel, (0, 0), (768, 28), (0, 0, 0), -1)
    cv2.putText(panel, f"{node_id}: frames {indices}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    ok, jpeg = cv2.imencode(".jpg", panel, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError(f"Could not encode {node_id}")
    return jpeg.tobytes()


def caption_batch(items, panels):
    ids = [node_id for node_id, _ in items]
    schema = {
        "type": "object",
        "properties": {"summaries": {
            "type": "array", "minItems": len(ids), "maxItems": len(ids),
            "items": {"type": "object", "properties": {
                "id": {"type": "string", "enum": ids},
                "summary": {"type": "string"}},
                "required": ["id", "summary"], "additionalProperties": False}}},
        "required": ["summaries"], "additionalProperties": False,
    }
    prompt = (
        "Each image is a labeled three-frame panel from one video segment. "
        "Return one dense factual summary per panel, using its exact node ID. "
        "Describe visible terrain, objects, buildings, roads, spatial relations, "
        "and supported changes. Treat each panel independently. Do not invent "
        "details. Keep each summary around 40-100 words. Panels appear in this "
        f"order: {ids}"
    )
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt,
                      "images": [base64.b64encode(x).decode() for x in panels]}],
        "format": schema, "stream": False, "think": False,
        "keep_alive": os.getenv("SKYTREE_KEEP_ALIVE", "30m"),
        "options": {"num_ctx": 16384, "num_predict": 3000, "temperature": 0},
    }
    request = Request("http://localhost:11434/api/chat",
                      data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=900) as response:
        result = json.loads(json.load(response)["message"]["content"])
    summaries = result["summaries"]
    mapped = {item["id"]: item["summary"] for item in summaries}
    if set(mapped) != set(ids):
        raise ValueError("Caption batch returned missing or duplicate node IDs")
    return mapped


def build_tree(video_path):
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise ValueError(f"Could not open {video_path}")
    tree = make_tree(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
    items = indexed_nodes(tree)
    for start in range(0, len(items), CAPTION_BATCH_SIZE):
        batch = items[start:start + CAPTION_BATCH_SIZE]
        panels = [node_panel(video, node_id, node) for node_id, node in batch]
        for node_id, summary in caption_batch(batch, panels).items():
            next(node for wanted, node in batch if wanted == node_id)["summary"] = summary
    video.release()
    return tree


def main():
    os.environ.setdefault("SKYTREE_KEEP_ALIVE", "30m")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    finished = {item["qa_id"] for item in results}

    sample = choose_sample()
    (OUTPUT / "sample.json").write_text(json.dumps([
        {"split": split, **qa} for split, qa in sample
    ], indent=2))

    for number, (split, qa) in enumerate(sample, 1):
        if qa["qa_id"] in finished:
            continue
        print(f"[{number}/{len(sample)}] {qa['qa_id']} {qa['task']}", flush=True)
        case_dir = OUTPUT / qa["qa_id"]
        case_dir.mkdir(exist_ok=True)
        tree_path = case_dir / "tree.json"
        try:
            if tree_path.exists():
                tree = json.loads(tree_path.read_text())
            else:
                tree = build_tree(DATASET / qa["video_path"])
                tree_path.write_text(json.dumps(tree, indent=2))
            trace_path = case_dir / "trace.json"
            state = run_search(tree, qa["question"], output_path=trace_path,
                               video_path=DATASET / qa["video_path"],
                               answer_path=case_dir / "answer.json",
                               options=qa["options"])
            raw_prediction = state["visual_answer"]["answer"].strip().upper()
            match = re.search(r"\b([A-D])\b", raw_prediction)
            predicted = match.group(1) if match else raw_prediction
            row = {"qa_id": qa["qa_id"], "split": split,
                   "video_path": qa["video_path"], "task": qa["task"],
                   "question": qa["question"], "options": qa["options"],
                   "correct_answer": qa["correct_answer"],
                   "predicted_answer": predicted,
                   "correct": predicted == qa["correct_answer"],
                   "error": None}
        except Exception as error:
            row = {"qa_id": qa["qa_id"], "split": split,
                   "video_path": qa["video_path"], "task": qa["task"],
                   "question": qa["question"], "options": qa["options"],
                   "correct_answer": qa["correct_answer"],
                   "predicted_answer": None, "correct": False,
                   "error": f"{type(error).__name__}: {error}"}
        results.append(row)
        results_path.write_text(json.dumps(results, indent=2))
        scored = [x for x in results if not x["error"]]
        correct = sum(x["correct"] for x in scored)
        print(f"result={row['predicted_answer']} expected={row['correct_answer']} "
              f"running={correct}/{len(scored)} errors={len(results)-len(scored)}",
              flush=True)


if __name__ == "__main__":
    main()

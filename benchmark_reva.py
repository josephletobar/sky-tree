import argparse
import base64
import json
import os
import random
import re
from pathlib import Path
from urllib.request import Request, urlopen

import cv2

from interval_search import MODEL
from orchestrate_search import run_search
from video_utils import sample_frames, split_to_n, uniform_samples


DATASET = Path("/data/ReVA")
OUTPUT = Path(__file__).parent / "summaries" / "reva_11_video_row_eval"
SAMPLE_SIZE = 11
SEED = 42


def choose_sample(sample_size=SAMPLE_SIZE, splits=("train", "val"), seed=SEED,
                  excluded_videos=(), excluded_qa_ids=(), excluded_questions=()):
    by_video = {}
    excluded_videos = set(excluded_videos)
    excluded_qa_ids = set(excluded_qa_ids)
    excluded_questions = set(excluded_questions)
    for split in splits:
        for qa in json.loads((DATASET / f"{split}.json").read_text())["QA"]:
            if (qa["video_path"] in excluded_videos
                    or qa["qa_id"] in excluded_qa_ids
                    or qa["question"] in excluded_questions):
                continue
            by_video.setdefault((split, qa["video_path"]), []).append(qa)

    rng = random.Random(seed)
    videos = list(by_video)
    rng.shuffle(videos)
    chosen = []
    chosen_questions = set()
    covered_tasks = set()
    for key in videos:
        if len(chosen) == sample_size:
            break
        candidates = [qa for qa in by_video[key]
                      if qa["question"] not in chosen_questions]
        if not candidates:
            continue
        qa = rng.choice(candidates)
        if qa["task"] not in covered_tasks:
            chosen.append((key[0], qa))
            covered_tasks.add(qa["task"])
            chosen_questions.add(qa["question"])
    for key in videos:
        if len(chosen) == sample_size:
            break
        if any(qa["video_path"] == key[1] for _, qa in chosen):
            continue
        candidates = [qa for qa in by_video[key]
                      if qa["question"] not in chosen_questions]
        if not candidates:
            continue
        qa = rng.choice(candidates)
        chosen.append((key[0], qa))
        chosen_questions.add(qa["question"])
    if len(chosen) != sample_size:
        raise ValueError(f"Could only find {len(chosen)} eligible unique videos/questions")
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


def caption_segment(video, node_id, node, question):
    indices = uniform_samples(node["start"], node["end"], 3)
    images = sample_frames(video, node["start"], node["end"], 3)
    schema = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"], "additionalProperties": False,
    }
    prompt = (
        "These are three separate full-resolution frames sampled in chronological "
        f"order from video segment {node_id}, at frame indices {indices}. "
        f"The video question is: {question}\n"
        "Describe all visible evidence that could help answer that question, "
        "including uncertainty or evidence that the relevant subject is absent. "
        "Describe visible terrain, objects, buildings, roads, spatial relations, "
        "and supported changes between frames. Inspect small and distant objects "
        "carefully. Do not invent details. Return one dense factual summary of "
        "roughly 150-200 words."
    )
    if "qwen" in MODEL.lower():
        prompt = "/no_think\n" + prompt
    for attempt in range(2):
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt,
                          "images": [base64.b64encode(x).decode() for x in images]}],
            "format": schema, "stream": False, "think": False,
            "keep_alive": os.getenv("SKYTREE_KEEP_ALIVE", "30m"),
            "options": {"num_ctx": 8192, "num_predict": 1600, "temperature": 0},
        }
        request = Request("http://localhost:11434/api/chat",
                          data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=900) as response:
            message = json.load(response)["message"]
        text = message.get("content", "").strip()
        if not text and "{" in message.get("thinking", ""):
            text = message["thinking"].strip()
        try:
            start = text.find("{")
            result, _ = json.JSONDecoder().raw_decode(text[start:])
            return result["summary"]
        except (ValueError, KeyError, json.JSONDecodeError):
            if attempt:
                raise ValueError(f"{MODEL} returned invalid caption JSON")


def build_tree(video_path, question):
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise ValueError(f"Could not open {video_path}")
    tree = make_tree(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
    tree["question"] = question
    items = [(f"root/{i}", node) for i, node in enumerate(tree["children"])]
    for node_id, node in items:
        node["summary"] = caption_segment(video, node_id, node, question)
    video.release()
    return tree


def main():
    parser = argparse.ArgumentParser(description="Run a resumable ReVA subset evaluation")
    parser.add_argument("--size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--splits", nargs="+", choices=("train", "val"),
                        default=["train", "val"])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--exclude-sample", type=Path)
    parser.add_argument("--exclude-results", type=Path, nargs="*", default=[])
    parser.add_argument("--exclude-video", action="append", default=[])
    args = parser.parse_args()

    os.environ.setdefault("SKYTREE_KEEP_ALIVE", "30m")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    finished = {item["qa_id"] for item in results}

    excluded_videos = set(args.exclude_video)
    excluded_qa_ids = set()
    excluded_questions = set()
    if args.exclude_sample:
        for item in json.loads(args.exclude_sample.read_text()):
            excluded_videos.add(item["video_path"])
            excluded_qa_ids.add(item["qa_id"])
            excluded_questions.add(item["question"])
    for path in args.exclude_results:
        data = json.loads(path.read_text())
        rows = data if isinstance(data, list) else data.get("results", [])
        for item in rows:
            if item.get("video_path"):
                excluded_videos.add(item["video_path"])
            if item.get("qa_id"):
                excluded_qa_ids.add(item["qa_id"])
            if item.get("question"):
                excluded_questions.add(item["question"])
    sample = choose_sample(args.size, args.splits, args.seed, excluded_videos,
                           excluded_qa_ids, excluded_questions)
    (output / "sample.json").write_text(json.dumps([
        {"split": split, **qa} for split, qa in sample
    ], indent=2))

    for number, (split, qa) in enumerate(sample, 1):
        if qa["qa_id"] in finished:
            continue
        print(f"[{number}/{len(sample)}] {qa['qa_id']} {qa['task']}", flush=True)
        case_dir = output / qa["qa_id"]
        case_dir.mkdir(exist_ok=True)
        tree_path = case_dir / "tree.json"
        try:
            if tree_path.exists():
                tree = json.loads(tree_path.read_text())
            else:
                tree = None
            if tree is None or tree.get("question") != qa["question"]:
                tree = build_tree(DATASET / qa["video_path"], qa["question"])
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

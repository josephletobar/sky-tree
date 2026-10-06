"""Run SkyTree or an upstream video-QA method on the ReVA benchmark."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from copy import deepcopy
from pathlib import Path


REVA_ROOT = Path("/data/ReVA")
BASELINES = Path("/data/aerial_benchmark_audit/baselines")
LETTERS = {"A", "B", "C", "D"}


def question_with_choices(item):
    choices = "\n".join(f"{key}. {value}" for key, value in item["options"].items())
    return f'{item["question"]}\n{choices}\nAnswer with only A, B, C, or D.'


def answer_letter(text):
    text = str(text or "").strip()
    patterns = [
        r"<answer>\s*([A-D])\s*</answer>",
        r"(?:answer|choice|option)\s*(?:is|:)?\s*\(?([A-D])\)?",
        r"^\s*\(?([A-D])\)?(?:[.\s]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def run_skytree(video_path, item, case_dir, _args):
    from benchmark_reva import build_tree
    from orchestrate_search import run_search

    tree_path = case_dir / "tree.json"
    tree = json.loads(tree_path.read_text()) if tree_path.exists() else None
    if tree is None or tree.get("question") != item["question"]:
        tree = build_tree(video_path, item["question"])
        tree_path.write_text(json.dumps(tree, indent=2))
    state = run_search(
        tree,
        item["question"],
        output_path=case_dir / "trace.json",
        video_path=video_path,
        answer_path=case_dir / "answer.json",
        options=item["options"],
    )
    return state["visual_answer"]["answer"], state


def run_longvideor1(video_path, item, case_dir, args):
    repo = BASELINES / "LongVideo-R1"
    python = repo / ".venv/bin/python"
    command = [
        str(python), str(repo / "cli.py"),
        "--video_path", str(video_path),
        "--question", question_with_choices(item),
        "--cache_dir", str(case_dir / "caption_cache"),
        "--reasoning_base_url", args.reasoning_url,
        "--caption_base_url", args.visual_url,
        "--videoqa_base_url", args.visual_url,
        "--reasoning_model", args.reasoning_model,
        "--caption_model", args.visual_model,
        "--videoqa_model", args.visual_model,
    ]
    completed = subprocess.run(command, cwd=repo, text=True, capture_output=True)
    (case_dir / "method.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"LongVideo-R1 failed; see {case_dir / 'method.log'}")
    matches = re.findall(r"^\[Answer\]\s*(.*)$", completed.stdout, re.MULTILINE)
    if not matches:
        raise ValueError("LongVideo-R1 did not print an answer")
    return matches[-1], {"log": "method.log"}


def run_vts(video_path, item, case_dir, args):
    """Use VTS's released single-sample worker with a ReVA-shaped sample."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    cap.release()
    worker_input = case_dir / "vts_input.json"
    worker_output = case_dir / "vts_worker.json"
    worker_input.write_text(json.dumps({
        "video_id": item["qa_id"],
        "qid": item["qa_id"],
        "video_path": str(video_path),
        "question": item["question"],
        "choices": list(item["options"].values()),
        "duration": duration,
        # VTS requires these fields for evaluation bookkeeping. They are blank
        # and are never included in the model prompt.
        "gt_timestamps": [], "answer": "", "right_answer": "",
    }))
    command = [
        str(BASELINES / "VTS/infer/.venv/bin/python"), str(Path(__file__).resolve()),
        "--vts-worker", str(worker_input), str(worker_output), str(case_dir),
        "--visual-url", args.visual_url, "--visual-model", args.visual_model,
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    (case_dir / "method.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"VTS failed; see {case_dir / 'method.log'}")
    result = json.loads(worker_output.read_text())
    return result["answer"], result


def run_videotree(_video_path, _item, _case_dir, _args):
    raise RuntimeError(
        "VideoTree's released pipeline requires precomputed frame features and "
        "captions; it has no raw-video inference entry point. Run its preprocessing "
        "first, then use its main_qa.py stage."
    )


METHODS = {
    "skytree": run_skytree,
    "longvideor1": run_longvideor1,
    "vts": run_vts,
    "videotree": run_videotree,
}


def read_results(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_outputs(dataset, rows, output_dir, split):
    latest = {row["qa_id"]: row for row in rows}
    completed = {qa_id: row for qa_id, row in latest.items()
                 if row.get("predicted_answer") in LETTERS}
    labeled = [row for row in completed.values() if row.get("correct_answer") in LETTERS]
    summary = {
        "split": split,
        "completed": len(completed),
        "total": len(dataset["QA"]),
        "errors": sum(bool(row.get("error")) for row in latest.values()),
        "accuracy": (sum(row["correct"] for row in labeled) / len(labeled)) if labeled else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    predictions = deepcopy(dataset)
    for item in predictions["QA"]:
        if item["qa_id"] in completed:
            item["correct_answer"] = completed[item["qa_id"]]["predicted_answer"]
    predictions_path = output_dir / "predictions.json"
    predictions_path.write_text(json.dumps(predictions, indent=2))

    if len(completed) == len(dataset["QA"]):
        with zipfile.ZipFile(output_dir / "submission.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(predictions_path, "predictions.json")
    return summary


def evaluate(args):
    split_path = args.reva_root / f"{args.split}.json"
    dataset = json.loads(split_path.read_text())
    output_dir = args.output or Path("results") / f"{args.method}_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    rows = read_results(results_path)
    done = {row["qa_id"] for row in rows if not row.get("error")}

    items = dataset["QA"][args.start:]
    if args.limit is not None:
        items = items[:args.limit]
    for number, item in enumerate(items, 1):
        if item["qa_id"] in done:
            continue
        case_dir = output_dir / item["qa_id"]
        case_dir.mkdir(exist_ok=True)
        started = time.time()
        try:
            raw, trace = METHODS[args.method](
                args.reva_root / item["video_path"], item, case_dir, args
            )
            predicted = answer_letter(raw)
            if predicted is None:
                raise ValueError(f"Could not parse A-D from answer: {raw!r}")
            (case_dir / "adapter_trace.json").write_text(json.dumps(trace, indent=2))
            error = None
        except Exception as exc:
            raw, predicted = None, None
            error = f"{type(exc).__name__}: {exc}"
        truth = item.get("correct_answer", "")
        row = {
            "qa_id": item["qa_id"], "video_path": item["video_path"],
            "question": item["question"], "method": args.method,
            "predicted_answer": predicted, "correct_answer": truth,
            "correct": predicted == truth if truth in LETTERS else None,
            "runtime_seconds": round(time.time() - started, 3), "error": error,
        }
        with results_path.open("a") as file:
            file.write(json.dumps(row) + "\n")
        rows.append(row)
        print(f"[{number}/{len(items)}] {item['qa_id']}: {predicted or error}", flush=True)

    summary = write_outputs(dataset, rows, output_dir, args.split)
    print(json.dumps(summary, indent=2))


def vts_worker(input_path, output_path, case_dir, visual_url, visual_model):
    repo = BASELINES / "VTS/infer"
    sys.path.insert(0, str(repo))
    from example_inference import _process_sample

    item = json.loads(Path(input_path).read_text())
    result = _process_sample(
        item, 0, 1, Path(case_dir), base_url=visual_url,
        model_name=visual_model, dataset_type="reva",
        min_segment_duration=1.0, max_turns=20,
    )
    files = list(Path(case_dir).glob("*_inference.json"))
    if not files:
        files = list((Path(case_dir) / "failed").glob("*_inference.json"))
    if not files:
        raise RuntimeError(result.get("error") or "VTS produced no inference result")
    trace = json.loads(files[-1].read_text())
    Path(output_path).write_text(json.dumps({"answer": trace["predicted_answer"], "trace": trace}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="skytree")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--reva-root", type=Path, default=REVA_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reasoning-url", default="http://127.0.0.1:25600/v1")
    parser.add_argument("--visual-url", default="http://127.0.0.1:9081/v1")
    parser.add_argument("--reasoning-model", default="longvideor1")
    parser.add_argument("--visual-model", default="Qwen3-VL-32B")
    parser.add_argument("--vts-worker", nargs=3, metavar=("INPUT", "OUTPUT", "CASE_DIR"),
                        help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.vts_worker:
        vts_worker(*parsed.vts_worker, parsed.visual_url, parsed.visual_model)
    else:
        evaluate(parsed)

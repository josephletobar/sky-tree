import base64
import json
import os
import subprocess
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np

from video_utils import uniform_samples


MODEL = os.getenv("SKYTREE_MODEL",
                  "hf.co/google/gemma-4-31B-it-qat-q4_0-gguf:latest")
FALLBACK_MODEL = "qwen3-vl:30b-a3b-thinking-q4_K_M"
SAM_PYTHON = "/data/skytree-sam3-venv/bin/python"
SAM_SCRIPT = Path(__file__).with_name("sam3_tool.py")
GRID_SIZE = 10
TIMESTAMP_RULE = """The supplied images are excerpts from the original video, not a new
sequence beginning at zero. Each image's time_seconds is its absolute timestamp
in the original video. Use those timestamps directly when choosing a temporal
answer; never replace them with an answer option's timestamp. When the exact
timestamp is not an option, compare the absolute numerical differences and
choose the nearest option. Never choose an option farther from the evidence."""


def _ask(prompt, images, schema):
    encoded = [base64.b64encode(Path(image).read_bytes()).decode()
               for image in images]
    models = [MODEL, MODEL, FALLBACK_MODEL]
    for attempt, model in enumerate(models):
        instruction = prompt if attempt == 0 else (
            prompt + "\nReturn complete valid JSON. Keep the response concise.")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "/no_think\n" + instruction,
                          "images": encoded}],
            "format": schema,
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "options": {"num_ctx": 16384, "num_predict": 1600,
                        "temperature": 0},
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
            return result
        except (ValueError, json.JSONDecodeError):
            if attempt == len(models) - 1:
                raise ValueError("Qwen returned invalid structured JSON")


def _index_tree(tree):
    nodes = {}

    def visit(node, node_id):
        nodes[node_id] = node
        for index, child in enumerate(node.get("children", [])):
            visit(child, f"{node_id}/{index}")

    visit(tree if isinstance(tree, dict) else {"children": tree}, "root")
    return nodes


def _leaf_ids(entries):
    for node_id, entry in entries.items():
        if entry.get("status") == "leaf":
            yield node_id
        yield from _leaf_ids(entry.get("children", {}))


def _safe_id(frame_id):
    return frame_id.replace("/", "_").replace(":", "_")


def _extract_frames(video_path, tree, state, directory):
    nodes = _index_tree(tree)
    leaf_ids = list(_leaf_ids(state.get("progress", {})))
    if not leaf_ids:
        raise ValueError("The search produced no leaf nodes to inspect")

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = video.get(cv2.CAP_PROP_FPS) or 0
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        video.release()
        raise ValueError(f"Could not determine video FPS: {video_path}")

    frames = []
    try:
        for node_id in leaf_ids:
            node = nodes[node_id]
            for index in uniform_samples(int(node["start"]), int(node["end"]), 3):
                video.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = video.read()
                if not ok:
                    raise ValueError(f"Could not read frame {index}")
                frame_id = f"{node_id}:{index}"
                path = directory / f"{_safe_id(frame_id)}.jpg"
                cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 96])
                frames.append({
                    "id": frame_id,
                    "node_id": node_id,
                    "frame": index,
                    "time_seconds": round(index / fps, 3),
                    "path": path,
                })
    finally:
        video.release()
    return frames, fps, total_frames / fps


def prepare_uniform_evidence(video_path, count=8):
    """Load evenly spaced full-resolution frames for direct visual answering."""
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = video.get(cv2.CAP_PROP_FPS) or 0
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        video.release()
        raise ValueError(f"Could not determine video FPS: {video_path}")
    frames = []
    try:
        for index in uniform_samples(0, total_frames, count):
            video.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = video.read()
            ok, encoded = cv2.imencode(".jpg", image,
                                       [cv2.IMWRITE_JPEG_QUALITY, 96]) if ok else (False, None)
            if not ok:
                raise ValueError(f"Could not read frame {index}")
            frames.append({"id": f"uniform:{index}", "frame": index,
                           "time_seconds": round(index / fps, 3),
                           "image": encoded.tobytes()})
    finally:
        video.release()
    return {"frames": frames, "fps": fps, "duration": total_frames / fps}


def prepare_node_evidence(video_path, tree, node_ids):
    """Load final evidence from a mixed-resolution antichain of tree nodes."""
    nodes = _index_tree(tree)
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = video.get(cv2.CAP_PROP_FPS) or 0
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        video.release()
        raise ValueError(f"Could not determine video FPS: {video_path}")
    frames = []
    try:
        for node_id in node_ids:
            node = nodes[node_id]
            count = 1 if node_id.count("/") == 1 else 3
            for index in uniform_samples(int(node["start"]), int(node["end"]), count):
                video.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = video.read()
                ok, encoded = cv2.imencode(".jpg", image,
                                           [cv2.IMWRITE_JPEG_QUALITY, 96]) if ok else (False, None)
                if not ok:
                    raise ValueError(f"Could not read frame {index}")
                frames.append({"id": f"{node_id}:{index}", "node_id": node_id,
                               "frame": index,
                               "time_seconds": round(index / fps, 3),
                               "image": encoded.tobytes()})
    finally:
        video.release()
    return {"frames": frames, "fps": fps, "duration": total_frames / fps}


def _materialize_evidence(evidence, directory):
    frames = []
    for item in evidence["frames"]:
        path = directory / f"{_safe_id(item['id'])}.jpg"
        path.write_bytes(item["image"])
        frames.append({key: value for key, value in item.items() if key != "image"}
                      | {"path": path})
    return frames, evidence["fps"], evidence["duration"]


def _make_grid(image_path, output_path):
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]
    gap, top, left = 4, 36, 52
    xs = [round(i * width / GRID_SIZE) for i in range(GRID_SIZE + 1)]
    ys = [round(i * height / GRID_SIZE) for i in range(GRID_SIZE + 1)]
    canvas = np.full((top + height + gap * 9, left + width + gap * 9, 3),
                     24, dtype=np.uint8)
    x_positions, cursor = [], left
    for x in range(GRID_SIZE):
        x_positions.append(cursor)
        cursor += xs[x + 1] - xs[x] + (gap if x < 9 else 0)
    y_positions, cursor = [], top
    for y in range(GRID_SIZE):
        y_positions.append(cursor)
        cursor += ys[y + 1] - ys[y] + (gap if y < 9 else 0)
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            tile = image[ys[y]:ys[y + 1], xs[x]:xs[x + 1]]
            py, px = y_positions[y], x_positions[x]
            canvas[py:py + tile.shape[0], px:px + tile.shape[1]] = tile
    for x in range(GRID_SIZE):
        center = x_positions[x] + (xs[x + 1] - xs[x]) // 2
        cv2.putText(canvas, f"x{x}", (center - 12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1,
                    cv2.LINE_AA)
    for y in range(GRID_SIZE):
        center = y_positions[y] + (ys[y + 1] - ys[y]) // 2
        cv2.putText(canvas, f"y{y}", (12, center + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1,
                    cv2.LINE_AA)
    cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 96])


def _crop(image_path, box, output_path):
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    crop = image[round(y1 * height / 10):round(y2 * height / 10),
                 round(x1 * width / 10):round(x2 * width / 10)]
    cv2.imwrite(str(output_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 96])


def _run_sam(jobs, directory):
    manifest = directory / "sam3_jobs.json"
    manifest.write_text(json.dumps(jobs, indent=2))
    subprocess.run(["ollama", "stop", MODEL], capture_output=True)
    env = os.environ.copy()
    env["HF_HOME"] = "/data/huggingface-cache"
    result = subprocess.run(
        [SAM_PYTHON, str(SAM_SCRIPT), str(manifest), str(directory)],
        env=env, capture_output=True, text=True, timeout=900, check=True,
    )
    return {item["id"]: item for item in json.loads(result.stdout.strip().splitlines()[-1])}


def answer_question(video_path, tree, state, artifact_dir=None, evidence=None):
    """Answer from selected frames, optionally using per-frame crop and SAM tools."""
    temporary = None
    if artifact_dir is None:
        temporary = tempfile.TemporaryDirectory(dir="/data/skytree-tmp")
        directory = Path(temporary.name)
    else:
        directory = Path(artifact_dir)
        directory.mkdir(parents=True, exist_ok=True)

    if evidence is None:
        frames, fps, duration = _extract_frames(video_path, tree, state, directory)
    else:
        frames, fps, duration = _materialize_evidence(evidence, directory)
    frame_context = [{key: frame[key] for key in ("id", "time_seconds")}
                     for frame in frames]
    question = state["question"]
    options = state.get("options")
    rounds = state.get("rounds", [])
    search_conclusion = rounds[-1].get("reasoning", "") if rounds else ""
    branch_conclusions = state.get("evidence_nodes", {})
    normalized_question = question.strip().lower()
    measurement_question = (normalized_question.startswith("how many")
                            or "number of" in normalized_question
                            or any(term in normalized_question for term in
                                   ("density", "occupancy")))
    timeline = [0, round(duration, 3)]

    decision_schema = {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "action": {"type": "string", "enum": ["answer", "tools", "insufficient"]},
            "answer": {"type": "string"},
        },
        "required": ["reason", "action", "answer"],
        "additionalProperties": False,
    }
    counting_rule = ("This question requires measurement across the evidence. "
                     "You must choose action 'tools'; unaided visual estimates "
                     "are not reliable." if measurement_question else "")
    choice_rule = ("This is a multiple-choice task. Ambiguity among visible "
                   "candidate objects, regions, or boundaries is not missing "
                   "evidence. In that situation you must identify the most "
                   "plausible visible referent, compare it with available scale "
                   "or context cues, and choose the closest supported option. "
                   "State the ambiguity and low confidence in the reason. Use "
                   "insufficient only when the relevant subject itself is absent "
                   "from the supplied evidence, not merely because an estimate "
                   "is approximate." if options else "")
    decision_prompt = f"""Act as the final verifier for an evidence search. The search conclusion and branch conclusions below combine information gathered across separately explored parts of the video. Verify that conclusion against the supplied images instead of solving the task again from scratch. Preserve it when the images agree. Override it only if you can name a specific supplied frame that contradicts it and explain the contradiction.

Reason through the visual and temporal evidence before choosing an action. Treat your unaided visual estimates as uncertain. Use tools whenever cropping or segmentation could measure, localize, count, or clarify relevant evidence. Answer immediately when the supplied evidence establishes the answer or the multiple-choice ambiguity rule below requires the closest supported option. If the relevant subject is absent and the available tools cannot repair that missing evidence, choose 'insufficient' and leave answer empty. Never guess an option after stating that the evidence is insufficient.
{counting_rule}
{choice_rule}
If you answer now, choose action 'answer' and give the answer. If tools can help, choose 'tools'. Otherwise choose 'insufficient'. Leave answer empty for both non-answer actions.
{TIMESTAMP_RULE}
The video timeline {timeline} is a hard boundary.
Question: {question}
Options: {json.dumps(options)}
Search conclusion: {search_conclusion}
Branch conclusions: {json.dumps(branch_conclusions)}
Frames: {json.dumps(frame_context)}"""
    decision = _ask(decision_prompt, [frame["path"] for frame in frames],
                    decision_schema)
    if measurement_question:
        decision["action"] = "tools"
        decision["answer"] = ""

    if decision["action"] == "insufficient":
        result = {
            "question": question, "video": str(video_path), "fps": fps,
            "video_duration_seconds": round(duration, 3),
            "leaf_nodes": frame_context, "answer": "",
            "reasoning": decision["reason"], "evidence": [],
            "evidence_sufficient": False, "tool_decision": decision,
            "tool_plan": [],
        }
        if temporary:
            temporary.cleanup()
        return result

    if decision["action"] == "answer":
        result = {
            "question": question, "video": str(video_path), "fps": fps,
            "video_duration_seconds": round(duration, 3),
            "leaf_nodes": frame_context, "answer": decision["answer"],
            "reasoning": decision["reason"],
            "evidence": [decision["reason"]], "tool_decision": decision,
            "tool_plan": [],
        }
        if temporary:
            temporary.cleanup()
        return result

    grid_paths = []
    for frame in frames:
        path = directory / f"{_safe_id(frame['id'])}_grid.jpg"
        _make_grid(frame["path"], path)
        grid_paths.append(path)

    plan_schema = {
        "type": "object",
        "properties": {
            "frames": {
                "type": "array",
                "minItems": len(frames) if measurement_question else 0,
                "maxItems": len(frames),
                "items": {
                    "type": "object",
                    "properties": {
                        "frame_id": {"type": "string", "enum": [f["id"] for f in frames]},
                        "crop": {
                            "type": "object",
                            "properties": {
                                "x1": {"type": "integer", "minimum": 0, "maximum": 10},
                                "y1": {"type": "integer", "minimum": 0, "maximum": 10},
                                "x2": {"type": "integer", "minimum": 0, "maximum": 10},
                                "y2": {"type": "integer", "minimum": 0, "maximum": 10},
                            },
                            "required": ["x1", "y1", "x2", "y2"],
                            "additionalProperties": False,
                        },
                        "segment": ({"type": "string", "minLength": 1}
                                    if measurement_question else
                                    {"type": ["string", "null"]}),
                    },
                    "required": ["frame_id", "crop", "segment"],
                    "additionalProperties": False,
                },
            },
            "reason": {"type": "string"},
        },
        "required": ["frames", "reason"],
        "additionalProperties": False,
    }
    segment_rule = ("Because this question requires measurement over the evidence, "
                    "include every supplied frame and provide a non-empty SAM "
                    "segmentation concept for each one." if measurement_question else "")
    plan_prompt = f"""Plan visual preprocessing for the question below. Each image has a separated 10x10 grid; x increases left-to-right and y top-to-bottom. For each frame you want to process, choose its own crop object with named x1, y1, x2, y2 coordinates; starts are inclusive and ends are exclusive. Drone motion means crops may differ by frame. A crop is required for every listed frame. Optionally set segment to the short, countable object concept SAM 3 should find inside the crop, or null when the crop alone is better. A segmentation concept names the target objects, not their location, container, or relationship. Unlisted frames remain unchanged. After execution you will automatically inspect all resulting evidence and answer.
{segment_rule}
Question: {question}
Options: {json.dumps(options)}
Frames in image order: {json.dumps(frame_context)}"""
    plan = _ask(plan_prompt, grid_paths, plan_schema)

    by_id = {frame["id"]: frame for frame in frames}
    seen = set()
    processed = {frame["id"]: frame["path"] for frame in frames}
    sam_jobs = []
    valid_plan = []
    for item in plan["frames"]:
        frame_id = item["frame_id"]
        if frame_id in seen or frame_id not in by_id:
            continue
        seen.add(frame_id)
        crop = item["crop"]
        x1, y1, x2, y2 = crop["x1"], crop["y1"], crop["x2"], crop["y2"]
        if not (0 <= x1 < x2 <= 10 and 0 <= y1 < y2 <= 10):
            continue
        executed_crop = [x1, y1, x2, y2]
        item["executed_crop"] = {
            "x1": executed_crop[0], "y1": executed_crop[1],
            "x2": executed_crop[2], "y2": executed_crop[3],
        }
        valid_plan.append(item)
        crop_path = directory / f"{_safe_id(frame_id)}_crop.jpg"
        _crop(by_id[frame_id]["path"], executed_crop, crop_path)
        processed[frame_id] = crop_path
        if item["segment"]:
            sam_jobs.append({
                "id": _safe_id(frame_id), "image": str(crop_path),
                "concept": item["segment"], "frame_id": frame_id,
            })
    plan["frames"] = valid_plan

    sam_results = {}
    if sam_jobs:
        raw_sam_results = _run_sam(sam_jobs, directory)
        for job in sam_jobs:
            result = raw_sam_results[job["id"]]
            processed[job["frame_id"]] = Path(result["image"])
            sam_results[job["frame_id"]] = result

    final_schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "answer": {"type": "string"},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "evidence_sufficient": {"type": "boolean"},
        },
        "required": ["reasoning", "answer", "evidence", "evidence_sufficient"],
        "additionalProperties": False,
    }
    final_context = []
    for frame in frames:
        entry = {key: frame[key] for key in ("id", "time_seconds")}
        planned = next((item for item in plan["frames"]
                        if item["frame_id"] == frame["id"]), None)
        if planned:
            entry.update({"requested_crop": planned["crop"],
                          "executed_crop": planned["executed_crop"],
                          "segment": planned["segment"]})
        if frame["id"] in sam_results:
            entry["sam_count"] = sam_results[frame["id"]]["count"]
        final_context.append(entry)
    final_prompt = f"""Act as the final verifier for an evidence search. The search conclusion and branch conclusions below combine information gathered across separately explored parts of the video. Verify that conclusion against the supplied images instead of solving the task again from scratch. Preserve it when the images agree. Override it only if you can name a specific supplied frame that contradicts it and explain the contradiction.

Reason through the evidence explicitly before giving the answer. Answer the original question using the processed images in chronological order. Some images may be untouched, cropped, or colored and numbered by SAM 3. Use both the pixels and supplied sam_count as evidence. SAM counts are fallible; verify the masks and requested spatial scope, but do not silently replace a consistent SAM count with an unsupported manual recount. For multiple choice, compare the evidence against every plausible option before choosing. Set evidence_sufficient=false and leave answer empty if the supplied evidence does not establish an answer. Never guess an option after explaining that evidence is missing or insufficient. The timeline {timeline} is a hard boundary. If choices are supplied and evidence is sufficient, return the chosen letter and answer.
{TIMESTAMP_RULE}
Question: {question}
Options: {json.dumps(options)}
Search conclusion: {search_conclusion}
Branch conclusions: {json.dumps(branch_conclusions)}
Images: {json.dumps(final_context)}"""
    final = _ask(final_prompt, [processed[frame["id"]] for frame in frames],
                 final_schema)
    if not final["evidence_sufficient"]:
        final["answer"] = ""
    result = {
        "question": question, "video": str(video_path), "fps": fps,
        "video_duration_seconds": round(duration, 3),
        "leaf_nodes": frame_context, **final,
        "tool_decision": decision, "tool_plan": plan,
        "sam3_results": sam_results,
    }
    if temporary:
        temporary.cleanup()
    return result

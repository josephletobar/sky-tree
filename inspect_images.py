import argparse
import json
from pathlib import Path

import cv2

from summarize_images import summarize_images
from video_utils import sample_frames, uniform_samples

PROMPT = """Answer the question using the supplied video frames.
The images are grouped by leaf node and listed in chronological order. Each
group has a node ID and an exclusive frame range; its images are the start,
middle, and end frames of that range. Use the actual pixels as the evidence.
Use goal_summary as the task definition while preserving the original question.
If answer options are supplied, use the images to choose the best option and
include the selected option in the answer.
Use conservative visual grounding: count or identify only objects that clearly
match the requested category. Do not relabel similar or related objects as the
target. If the pixels are ambiguous, report that uncertainty instead of
asserting a precise count. Do not invent observations. Give a direct answer
and briefly cite the node IDs or frame ranges that support it. Return JSON with
exactly these fields:
{\"answer\": \"...\", \"evidence\": [\"...\"]}
"""


def _index_tree(tree):
    if isinstance(tree, list):
        tree = {"children": tree}
    nodes = {}

    def visit(node, node_id):
        nodes[node_id] = node
        for index, child in enumerate(node.get("children", [])):
            visit(child, f"{node_id}/{index}")

    visit(tree, "root")
    return nodes


def _leaf_ids(entries):
    for node_id, entry in entries.items():
        if entry.get("status") == "leaf":
            yield node_id
        yield from _leaf_ids(entry.get("children", {}))


def inspect_video(video_path, tree, state):
    """Inspect selected leaf-node frames and return Gemma's final answer."""
    nodes = _index_tree(tree)
    leaf_ids = list(_leaf_ids(state.get("progress", {})))
    if not leaf_ids:
        raise ValueError("The search produced no leaf nodes to inspect")

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = video.get(cv2.CAP_PROP_FPS) or 0
    images = []
    groups = []
    try:
        for node_id in leaf_ids:
            node = nodes[node_id]
            start = int(node["start"])
            end = int(node["end"])
            indices = uniform_samples(start, end, 3)
            groups.append({"id": node_id, "start": start, "end": end,
                           "frames": indices})
            images.extend(sample_frames(video, start, end, 3))
    finally:
        video.release()

    context = {
        "question": state["question"],
        "options": state.get("options"),
        "goal_summary": state.get("goal_summary", state["question"]),
        "fps": fps,
        "groups": groups,
        "image_order": "For each group, images appear as start, middle, end; duplicate frames are included once.",
    }
    raw_answer = summarize_images(images, PROMPT + "\n" + json.dumps(context))
    try:
        text = raw_answer.strip()
        if text.startswith("```"):
            text = "\n".join(text.splitlines()[1:]).strip()
        start = text.find("{")
        answer, _ = json.JSONDecoder().raw_decode(text[start:])
        answer = {key.lower(): value for key, value in answer.items()}
        if "answer" not in answer or "evidence" not in answer:
            raise ValueError("Missing answer fields")
    except (json.JSONDecodeError, ValueError):
        answer = {"answer": raw_answer, "evidence": []}
    return {"question": state["question"], "video": str(video_path),
            "fps": fps, "leaf_nodes": groups, **answer}


def main():
    parser = argparse.ArgumentParser(description="Inspect selected leaf-node frames")
    parser.add_argument("tree", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = inspect_video(args.video, json.loads(args.tree.read_text()),
                           json.loads(args.trace.read_text()))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

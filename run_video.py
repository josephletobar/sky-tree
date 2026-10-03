import cv2
import json
import tomllib
from pathlib import Path
from pprint import pprint

from summarize_images import summarize_images
from video_utils import sample_frames, split_to_n

VIDEO_DIR = Path("/home/joseph/reva_hard_examples/videos")


def save_json(data, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def play_segments(cap, segments, pause_seconds=2):
    frame_delay = max(1, round(1000 / (cap.get(cv2.CAP_PROP_FPS) or 30)))
    try:
        for segment_index, (start, end) in enumerate(segments):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for _ in range(start, end):
                ok, frame = cap.read()
                if not ok:
                    return
                cv2.imshow("Video Segment", frame)
                if cv2.waitKey(frame_delay) & 0xFF == ord("q"):
                    return
            if segment_index < len(segments) - 1:
                if cv2.waitKey(pause_seconds * 1000) & 0xFF == ord("q"):
                    return
    finally:
        cv2.destroyAllWindows()


def recursive_split(cap, start, end, samples_per_segment, output_dir):
    images = sample_frames(cap, start, end, samples_per_segment)
    print(f"Summarizing frames {start}–{end}...", flush=True)
    node = {"start": start, "end": end, "children": [], "summary": summarize_images(images)}
    save_json(node, output_dir / f"{start}-{end}.json")
    if end - start > 15:
        node["children"] = [
            recursive_split(cap, a, b, samples_per_segment, output_dir)
            for a, b in split_to_n(start, end, 2)
        ]
        save_json(node, output_dir / f"{start}-{end}.json")
    return node


def main():
    with Path(__file__).with_name("videos.toml").open("rb") as file:
        config = tomllib.load(file)

    video_path = VIDEO_DIR / f"{config['selected']}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    segments = split_to_n(0, total_frames, config["top_split"])
    output_dir = Path(__file__).parent / "summaries" / config["selected"]
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        play_segments(cap, segments)
        tree = {
            "start": 0,
            "end": total_frames,
            "children": [
                recursive_split(cap, start, end, config["samples_per_segment"], output_dir)
                for start, end in segments
            ],
            "summary": "TODO: load the video-level consolidated_caption",
        }
    finally:
        cap.release()

    save_json(tree, output_dir / "tree.json")
    pprint(tree, sort_dicts=False)


if __name__ == "__main__":
    main()

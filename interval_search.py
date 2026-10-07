import argparse
import base64
from copy import deepcopy
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np

from video_utils import uniform_samples


MODEL = os.getenv(
    "SKYTREE_MODEL", "hf.co/google/gemma-4-31B-it-qat-q4_0-gguf:latest"
)
SAMPLES_PER_WINDOW = 8
MAX_ROUNDS = 6

PROMPT = """You control a temporal zoom lens over a short video. The eight
section summaries are a high-recall index: use them to find promising times,
but treat their claims as uncertain until verified in original frames.

At each round choose exactly one action:
- answer: the selected or current frames are sufficient for the final visual agent.
- zoom_in: inspect a strictly smaller time interval inside the current window.
- zoom_out: return to the parent window when the current crop was too narrow or
  you need to investigate a different period.

selected_frames is the complete evidence set you want to preserve across
rounds. It may contain frames from the current window or frames preserved from
an earlier window. Replace this list each round; omit irrelevant frames. For a
localized question, preserve only frames that directly contribute. For a
boundary or comparison question, preserve the distinct endpoints or states.
For a whole-video question, answer from the broad current view without zooming.
Zoom only to resolve a named visual uncertainty, never merely because more
detail might help. Frame IDs, frame indices, and timestamps are visibly labeled.
Reason concisely and never claim to have observed an unsupplied frame.
"""


class IntervalSearch:
    def __init__(self, tree, video_path, output_path=None,
                 samples_per_window=SAMPLES_PER_WINDOW, max_rounds=MAX_ROUNDS):
        self.tree = deepcopy(tree if isinstance(tree, dict) else {"children": tree})
        self.video_path = Path(video_path)
        self.output_path = Path(output_path) if output_path else None
        self.samples_per_window = samples_per_window
        self.max_rounds = max_rounds
        self.question = None
        self.options = None
        self.fps = None
        self.total_frames = None
        self.duration = None
        self.sections = []
        self.window = None
        self.parents = []
        self.selected_frames = []
        self.observed = {}
        self.rounds = []
        self.answered = False

    def state(self):
        return deepcopy({
            "question": self.question,
            "options": self.options,
            "sections": self.sections,
            "window": self.window,
            "parent_windows": self.parents,
            "selected_frames": self.selected_frames,
            "rounds": self.rounds,
            "answered": self.answered,
            "fps": self.fps,
            "duration": self.duration,
        })

    def _save(self):
        state = self.state()
        if self.output_path:
            temporary = self.output_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
            temporary.replace(self.output_path)
        return state

    def _video_info(self):
        video = cv2.VideoCapture(str(self.video_path))
        if not video.isOpened():
            raise ValueError(f"Could not open {self.video_path}")
        self.fps = video.get(cv2.CAP_PROP_FPS) or 0
        self.total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()
        if self.fps <= 0 or self.total_frames <= 0:
            raise ValueError(f"Could not read video metadata from {self.video_path}")
        self.duration = self.total_frames / self.fps

    def _section_index(self):
        top = self.tree.get("children", [])
        if not top:
            raise ValueError("The summary tree has no top-level sections")
        self.sections = []
        for index, section in enumerate(top[:8]):
            start = int(section.get("start", 0))
            end = int(section.get("end", start))
            self.sections.append({
                "section": index,
                "start_seconds": round(start / self.fps, 3),
                "end_seconds": round(end / self.fps, 3),
                "summary": section.get("summary", ""),
            })

    def _sample_window(self, start_seconds, end_seconds):
        start_frame = max(0, min(self.total_frames - 1,
                                 round(start_seconds * self.fps)))
        end_frame = max(start_frame + 1, min(self.total_frames,
                                             round(end_seconds * self.fps)))
        indices = uniform_samples(start_frame, end_frame, self.samples_per_window)
        video = cv2.VideoCapture(str(self.video_path))
        if not video.isOpened():
            raise ValueError(f"Could not open {self.video_path}")
        frames = []
        try:
            for frame_index in indices:
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = video.read()
                if not ok:
                    raise ValueError(f"Could not read frame {frame_index}")
                ok, raw = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 96])
                if not ok:
                    raise ValueError(f"Could not encode frame {frame_index}")
                frame_id = f"frame_{frame_index}"
                seconds = frame_index / self.fps
                header = np.full((50, frame.shape[1], 3), 245, dtype=np.uint8)
                cv2.putText(header, f"{frame_id}  {seconds:.2f}s", (12, 33),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2,
                            cv2.LINE_AA)
                ok, labeled = cv2.imencode(".jpg", np.vstack((header, frame)))
                if not ok:
                    raise ValueError(f"Could not label frame {frame_index}")
                record = {
                    "id": frame_id,
                    "frame": frame_index,
                    "time_seconds": round(seconds, 3),
                    "image": raw.tobytes(),
                    "labeled_image": labeled.tobytes(),
                }
                self.observed[frame_id] = record
                frames.append(frame_id)
        finally:
            video.release()
        return {
            "start_seconds": round(start_frame / self.fps, 3),
            "end_seconds": round(end_frame / self.fps, 3),
            "frames": frames,
        }

    def initialize(self, question, options=None):
        self._video_info()
        self._section_index()
        self.question = question
        self.options = options
        self.parents = []
        self.selected_frames = []
        self.observed = {}
        self.rounds = []
        self.answered = False
        self.window = self._sample_window(0, self.duration)
        return self._save()

    def _visible_frame_ids(self):
        ids = list(self.window["frames"])
        for frame_id in self.selected_frames:
            if frame_id not in ids:
                ids.append(frame_id)
        return ids

    def _ask(self):
        visible_ids = self._visible_frame_ids()
        frame_context = [
            {key: self.observed[frame_id][key]
             for key in ("id", "frame", "time_seconds")}
            for frame_id in visible_ids
        ]
        schema = {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "maxLength": 600},
                "action": {"type": "string",
                           "enum": ["answer", "zoom_in", "zoom_out"]},
                "start_seconds": {"type": "number", "minimum": 0,
                                  "maximum": self.duration},
                "end_seconds": {"type": "number", "minimum": 0,
                                "maximum": self.duration},
                "selected_frames": {
                    "type": "array", "minItems": 0,
                    "maxItems": len(visible_ids),
                    "items": {"type": "string", "enum": visible_ids},
                },
            },
            "required": ["reasoning", "action", "start_seconds",
                         "end_seconds", "selected_frames"],
            "additionalProperties": False,
        }
        context = {
            "question": self.question,
            "options": self.options,
            "video_duration_seconds": round(self.duration, 3),
            "sections": self.sections,
            "current_window": self.window,
            "can_zoom_out": bool(self.parents),
            "selected_frames": self.selected_frames,
            "frames_in_image_order": frame_context,
            "round": len(self.rounds) + 1,
            "rounds_remaining": self.max_rounds - len(self.rounds),
        }
        images = [base64.b64encode(self.observed[frame_id]["labeled_image"]).decode()
                  for frame_id in visible_ids]
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": json.dumps(context), "images": images},
            ],
            "format": schema,
            "stream": False,
            "think": False,
            "keep_alive": os.getenv("SKYTREE_KEEP_ALIVE", "30m"),
            "options": {"num_ctx": 16384, "num_predict": 1000,
                        "temperature": 0},
        }
        request = Request("http://localhost:11434/api/chat",
                          data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=900) as response:
            message = json.load(response)["message"]
        text = message.get("content", "").strip() or message.get("thinking", "").strip()
        start = text.find("{")
        result, _ = json.JSONDecoder().raw_decode(text[start:])
        return result

    def step(self):
        if self.answered:
            raise ValueError("Search has already answered")
        result = self._ask()
        action = result["action"]
        selected = result["selected_frames"]
        if len(set(selected)) != len(selected):
            raise ValueError("selected_frames contains duplicates")
        visible = set(self._visible_frame_ids())
        if any(frame_id not in visible for frame_id in selected):
            raise ValueError("A selected frame was not supplied to the agent")

        before = deepcopy(self.window)
        self.selected_frames = selected
        if action == "zoom_in":
            start = float(result["start_seconds"])
            end = float(result["end_seconds"])
            current_start = self.window["start_seconds"]
            current_end = self.window["end_seconds"]
            if not current_start <= start < end <= current_end:
                raise ValueError("zoom_in interval must be inside the current window")
            if end - start >= current_end - current_start - (1 / self.fps):
                raise ValueError("zoom_in must request a smaller interval")
            self.parents.append(deepcopy(self.window))
            self.window = self._sample_window(start, end)
        elif action == "zoom_out":
            if not self.parents:
                raise ValueError("Cannot zoom out from the full-video window")
            self.window = self.parents.pop()
        else:
            self.answered = True

        self.rounds.append({
            "round": len(self.rounds) + 1,
            "reasoning": result["reasoning"],
            "action": action,
            "requested_interval": [result["start_seconds"], result["end_seconds"]],
            "window_before": before,
            "window_after": deepcopy(self.window),
            "selected_frames": list(self.selected_frames),
        })
        if len(self.rounds) >= self.max_rounds and not self.answered:
            self.answered = True
            self.rounds.append({
                "round": len(self.rounds) + 1,
                "reasoning": "Search-round limit reached; use the best current evidence.",
                "action": "answer",
                "requested_interval": [self.window["start_seconds"],
                                       self.window["end_seconds"]],
                "window_before": deepcopy(self.window),
                "window_after": deepcopy(self.window),
                "selected_frames": list(self.selected_frames),
            })
        return self._save()

    def evidence(self):
        ids = self.selected_frames or self.window["frames"]
        records = [self.observed[frame_id] for frame_id in ids]
        records.sort(key=lambda item: item["frame"])
        return {
            "frames": [
                {key: item[key] for key in ("id", "frame", "time_seconds", "image")}
                for item in records
            ],
            "fps": self.fps,
            "duration": self.duration,
        }


def main():
    parser = argparse.ArgumentParser(description="Search a video with temporal zoom")
    parser.add_argument("tree", type=Path,
                        help="Summary tree; only its eight top-level sections are used")
    parser.add_argument("video", type=Path)
    parser.add_argument("question")
    parser.add_argument("--options", type=Path)
    args = parser.parse_args()

    tree = json.loads(args.tree.read_text())
    options = json.loads(args.options.read_text()) if args.options else None
    search = IntervalSearch(tree, args.video,
                            output_path=args.tree.with_name("interval_trace.json"))
    search.initialize(args.question, options)
    while not search.answered:
        search.step()
    print(json.dumps(search.state(), indent=2))


if __name__ == "__main__":
    main()

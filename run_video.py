import cv2
import json
import time
import tomllib
from pathlib import Path
from pprint import pprint
from summarize_images import summarize_images

with Path(__file__).with_name("videos.toml").open("rb") as file:
    config = tomllib.load(file)

video = f"/home/joseph/reva_hard_examples/videos/{config['selected']}.mp4"
cap = cv2.VideoCapture(video)
delay = 2
frame_delay = max(1, round(1000 / (cap.get(cv2.CAP_PROP_FPS) or 30)))

def uniform_samples(start, end, n):
    n = min(n, end - start)
    if n == 1:
        return [start]
    return [round(start + i * (end - start - 1) / (n - 1)) for i in range(n)]

def split_to_n(start, end, n):
    frames_per_segment = (end - start) // n
    segments = []
    for i in range(n):
        start_frame = start + i * frames_per_segment
        end_frame = start + (i + 1) * frames_per_segment if i < n - 1 else end
        segments.append((start_frame, end_frame))
    return segments

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
segments = split_to_n(0, total_frames, config['top_split'])

# debug show top level segments
for segment in segments:
    start_frame, end_frame = segment
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_num in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("Video Segment", frame)
        if cv2.waitKey(frame_delay) & 0xFF == ord("q"):
            break
    else:
        if segment != segments[-1] and cv2.waitKey(delay * 1000) & 0xFF == ord("q"):
            break
        continue
    break


# START RECURSIVE SPLIT

output_dir = Path(__file__).parent / "summaries" / config['selected']
output_dir.mkdir(parents=True, exist_ok=True)

def save_json(data, name):
    path = output_dir / name
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)

def recursive_split(start, end):
    images = []
    for index in uniform_samples(start, end, config['samples_per_segment']):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Could not read frame {index}")
        ok, image = cv2.imencode(".jpg", frame)
        if not ok:
            raise ValueError(f"Could not encode frame {index}")
        images.append(image.tobytes())
    print(f"Summarizing frames {start}–{end}...", flush=True)
    node = {"start": start, "end": end, "children": [], "summary": summarize_images(images)}
    save_json(node, f"{start}-{end}.json")
    if end - start > 15:
        node["children"] = [recursive_split(a, b) for a, b in split_to_n(start, end, 2)]
        save_json(node, f"{start}-{end}.json")
    return node

tree = {
    "start": 0,
    "end": total_frames,
    "children": [recursive_split(start, end) for start, end in segments],
    "summary": "TODO: load the video-level consolidated_caption",
}

save_json(tree, "tree.json")
pprint(tree, sort_dicts=False)


        
# while True:
#     ok, frame = cap.read()
#     if not ok:
#         break
#     cv2.imshow("Video", frame)
#     if cv2.waitKey(delay) & 0xFF == ord("q"):
#         break

cap.release()
cv2.destroyAllWindows()

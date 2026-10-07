# sky-tree

Simple OpenCV video player. Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_video.py
```

Choose a video name in `videos.toml`. Videos are loaded from
`/home/joseph/reva_hard_examples/videos` (change the path in `run_video.py` if needed).
Press **Q** to quit.

Run temporal-window evidence search with Gemma through your local Ollama server:

```bash
.venv/bin/python interval_search.py summaries/DJI_0157_d4_01/pasted_tree.json /path/to/video.mp4 "Your question"
```

The eight top-level summaries form a high-recall index. Gemma sees eight labeled
frames across the complete video, then answers or zooms into an arbitrary time
interval. It can zoom back out, inspect another interval, and preserve selected
frames between rounds. Recursive tree children are ignored. The trace is saved
beside the summary tree as `interval_trace.json`.

To run search and the final visual tool pipeline together:

```bash
.venv/bin/python orchestrate_search.py summaries/DJI_0157_d4_01/pasted_tree.json "Your question" --video /path/to/video.mp4
```

The runner sends only the selected frames to the shared Crop/SAM/final-answer
pipeline.

Use the search directly from Python:

```python
import json
from pathlib import Path
from interval_search import IntervalSearch

tree = json.loads(Path("summaries/DJI_0157_d4_01/pasted_tree.json").read_text())
search = IntervalSearch(tree, "/path/to/video.mp4",
                        output_path="interval_trace.json")
search.initialize("Your question")
while not search.answered:
    search.step()
print(search.state())
```

Run any supported method through the same ReVA interface:

```bash
python3 method_adapter.py --method skytree --split val --limit 20
python3 method_adapter.py --method longvideor1 --split test
python3 method_adapter.py --method vts --split test
```

Each run resumes from `results.jsonl`, saves its per-question traces, and writes
`predictions.json`. A Codabench-ready `submission.zip` is created once every
question in the split has a valid prediction. The available method names are
`skytree`, `longvideor1`, `vts`, and `videotree`; the released VideoTree code
must be preprocessed before its QA stage can run.

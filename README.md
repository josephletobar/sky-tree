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

Run adaptive evidence search with Gemma through your local Ollama server:

```bash
.venv/bin/python choose_nodes.py summaries/DJI_0157_d4_01/pasted_tree.json "Your question" --video /path/to/video.mp4
```

The search begins with uniform top-level coverage. Each round independently
retains, expands, or prunes every evidence node. Expansion replaces its parent.
The full action trace is saved beside the tree as `search_trace.json`.

To run the full search one level at a time:

```bash
.venv/bin/python orchestrate_search.py summaries/DJI_0157_d4_01/pasted_tree.json "Your question"
```

The runner uses the same adaptive search and then sends its mixed-resolution
evidence to the shared visual answer and tool pipeline.

Use the search directly from Python:

```python
import json
from pathlib import Path
from choose_nodes import AdaptiveEvidenceSearch

tree = json.loads(Path("summaries/DJI_0157_d4_01/pasted_tree.json").read_text())
search = AdaptiveEvidenceSearch(tree, output_path="search_trace.json",
                                video_path="/path/to/video.mp4")
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

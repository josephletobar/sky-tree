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

Choose nodes for a question with Gemma through your local Ollama server:

```bash
.venv/bin/python choose_nodes.py summaries/DJI_0157_d4_01/pasted_tree.json "Your question" --level 1
```

The root is level 0. Results and nested progress are saved beside the input
as `traversal_plan.json`. Four is the maximum number of active branches.

To run the full search one level at a time:

```bash
.venv/bin/python orchestrate_search.py summaries/DJI_0157_d4_01/pasted_tree.json "Your question"
```

The runner compares the complete candidate row in one call before moving to the
next depth. It stops when every remaining branch reaches a leaf.

For manual child selection:

```python
import json
from pathlib import Path
from choose_nodes import NodeChooser

tree = json.loads(Path("summaries/DJI_0157_d4_01/pasted_tree.json").read_text())
chooser = NodeChooser(tree, output_path="traversal_plan.json")
state = chooser.orchestrate("Your question", level=1)
node_id = next(iter(state["active_nodes"]))
state = chooser.choose_children(node_id)
print(chooser.progress)
```

Omit `output_path` to keep state in memory. Calling `orchestrate` again starts
a fresh search after its response passes validation. Selected IDs use tree
positions (`root/0/1`), rather than frame ranges. Existing saved plans use the
old format; running the new CLI replaces them with the new state format.

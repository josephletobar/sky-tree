import argparse
import json
from pathlib import Path

from choose_nodes import NodeChooser


def run_search(tree, question, level=1, output_path=None, video_path=None,
               answer_path=None, options=None):
    """Choose the initial nodes, then expand one complete tree level at a time."""
    chooser = NodeChooser(tree, output_path=output_path)
    chooser.orchestrate(question, level, options)
    while any(chooser.nodes[node_id].get("children")
              for node_id in chooser.active_nodes):
        chooser.choose_next_row()

    state = chooser.state()
    if video_path:
        from inspect_images import inspect_video
        state["visual_answer"] = inspect_video(video_path, tree, state)
        if output_path:
            output_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        if answer_path:
            answer_path.write_text(json.dumps(state["visual_answer"], indent=2),
                                   encoding="utf-8")
    return state


def main():
    parser = argparse.ArgumentParser(description="Run a level-order video tree search")
    parser.add_argument("tree", type=Path)
    parser.add_argument("question")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--video", type=Path,
                        help="Inspect selected leaf frames after traversal")
    parser.add_argument("--answer-output", type=Path)
    parser.add_argument("--options", type=Path,
                        help="JSON file containing answer choices")
    args = parser.parse_args()

    output_path = args.tree.with_name("traversal_plan.json")
    tree = json.loads(args.tree.read_text())
    options = json.loads(args.options.read_text()) if args.options else None
    state = run_search(tree, args.question, args.level, output_path,
                       args.video, args.answer_output, options)
    print(json.dumps(state, indent=2))
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

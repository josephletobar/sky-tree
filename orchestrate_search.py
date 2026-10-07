import argparse
import json
from pathlib import Path

from interval_search import IntervalSearch
from visual_agent import answer_question


def run_search(tree, question, output_path=None, video_path=None,
               answer_path=None, options=None, level=1):
    """Search temporal windows, then send compact evidence to the visual agent."""
    if not video_path:
        raise ValueError("Interval search requires a video path")

    search = IntervalSearch(tree, video_path, output_path=output_path)
    search.initialize(question, options)
    while not search.answered:
        search.step()

    state = search.state()
    answer_state = {
        **state,
        "evidence_nodes": {
            frame_id: "Selected by temporal-window search"
            for frame_id in state["selected_frames"]
        },
    }
    artifact_dir = Path(answer_path).parent / "visual_tools" if answer_path else None
    state["visual_answer"] = answer_question(
        video_path, tree, answer_state,
        artifact_dir=artifact_dir,
        evidence=search.evidence(),
    )
    if output_path:
        Path(output_path).write_text(json.dumps(state, indent=2), encoding="utf-8")
    if answer_path:
        Path(answer_path).write_text(json.dumps(state["visual_answer"], indent=2),
                                     encoding="utf-8")
    return state


def main():
    parser = argparse.ArgumentParser(description="Run temporal interval search")
    parser.add_argument("tree", type=Path)
    parser.add_argument("question")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--answer-output", type=Path)
    parser.add_argument("--options", type=Path)
    args = parser.parse_args()

    tree = json.loads(args.tree.read_text())
    options = json.loads(args.options.read_text()) if args.options else None
    output_path = args.tree.with_name("interval_trace.json")
    state = run_search(tree, args.question, output_path, args.video,
                       args.answer_output, options)
    print(json.dumps(state, indent=2))
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

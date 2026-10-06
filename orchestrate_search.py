import argparse
import json
from pathlib import Path

from choose_nodes import AdaptiveEvidenceSearch
from visual_agent import answer_question, prepare_node_evidence


def run_search(tree, question, output_path=None, video_path=None,
               answer_path=None, options=None):
    """Adaptively refine one evidence frontier, then run the visual-answer agent."""
    search = AdaptiveEvidenceSearch(tree, output_path=output_path,
                                    video_path=video_path)
    search.initialize(question, options)
    while not search.answered:
        search.step()

    state = search.state()
    if video_path:
        evidence = prepare_node_evidence(video_path, tree,
                                         state["evidence_nodes"])
        artifact_dir = Path(answer_path).parent / "visual_tools" if answer_path else None
        state["visual_answer"] = answer_question(
            video_path, tree, state, artifact_dir=artifact_dir, evidence=evidence
        )
        if output_path:
            Path(output_path).write_text(json.dumps(state, indent=2), encoding="utf-8")
        if answer_path:
            Path(answer_path).write_text(json.dumps(state["visual_answer"], indent=2),
                                         encoding="utf-8")
    return state


def main():
    parser = argparse.ArgumentParser(description="Run adaptive video evidence search")
    parser.add_argument("tree", type=Path)
    parser.add_argument("question")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--answer-output", type=Path)
    parser.add_argument("--options", type=Path,
                        help="JSON file containing answer choices")
    args = parser.parse_args()

    output_path = args.tree.with_name("search_trace.json")
    tree = json.loads(args.tree.read_text())
    options = json.loads(args.options.read_text()) if args.options else None
    state = run_search(tree, args.question, output_path,
                       args.video, args.answer_output, options)
    print(json.dumps(state, indent=2))
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

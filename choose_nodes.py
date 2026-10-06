import argparse
import base64
from copy import deepcopy
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

import cv2

from video_utils import sample_frames, uniform_samples


MODEL = os.getenv(
    "SKYTREE_MODEL", "hf.co/google/gemma-4-31B-it-qat-q4_0-gguf:latest"
)
FRONTIER_FRAME_BUDGET = 24
MAX_FRAMES_PER_NODE = 8
POLICY_PROMPT = """Manage one adaptive video-evidence set. For every supplied
node choose exactly one action: retain, expand, or prune. Retain keeps the node
at its current resolution. Expand replaces it with its immediate children.
Prune removes it. A node and its descendants can never coexist.
Never expand a node whose children list is empty.

Set answer=true only when the evidence remaining after these actions is
sufficient for the final visual-answer agent. Do not expand when answer=true.
Before acting, decide what evidence resolution the question requires. If the
answer depends on details that coarse evidence cannot directly establish,
top-level summaries are only hypotheses about where to look. This includes
fine temporal distinctions such as action transitions, order, repetition,
cadence, frequency, brief changes, identity continuity, and precise boundaries,
as well as fine visual or spatial distinctions such as small objects, exact
counts, close geometric estimates, and similar answer choices separated by
subtle visible details. Coarse evidence is never sufficient in these cases,
even when it suggests a plausible answer. Expand every coarse branch that could
contain the discriminating evidence, and set answer=true only after that detail
is directly represented by finer nodes. A terminal leaf cannot expand; retain
it when relevant so the final visual agent can inspect and crop its original
full-resolution frame. When the relevant fine-evidence nodes are terminal
leaves, set answer=true to hand them to that agent; do not continue without a
possible refinement.
Questions about change, consistency, trends, motion, or behavior across the
video require broad temporal coverage from multiple separated periods. For
these questions, observations from different times are corroborating temporal
evidence, not redundant evidence. Never reduce such a question to one period
and never answer it from a single node. Retain broad top-level coverage unless
a period is genuinely irrelevant, and refine only periods whose ambiguity
could change the answer.
Boundary or local questions should prune unrelated periods and expand only the
specific uncertain regions. Different branches may stop at different depths.
When an answer asserts that something appears, disappears, or remains present
through an interval, directly preserve evidence for every claimed temporal
endpoint. Evidence at one boundary cannot establish the other boundary, and
eliminating answer choices cannot substitute for observing a claimed start or
end. Do not prune later periods merely because an earlier observation makes one
option seem like the only remaining choice.

Do not expand merely because you are uncertain. For every expansion, name a
specific unresolved distinction that could change the answer and explain how
finer evidence should resolve it. Retained nodes must contribute distinct
information rather than being kept just in case. Never prune every node. Frame
indices and seconds are different units: use the supplied FPS when
relating them and never compare a raw frame index directly with seconds.
Reason before acting. Answer options are hypotheses, not evidence requirements.
"""


class AdaptiveEvidenceSearch:
    def __init__(self, tree, output_path=None, video_path=None):
        self.tree = deepcopy(tree)
        if isinstance(self.tree, list):
            self.tree = {"children": self.tree}
        self.output_path = Path(output_path) if output_path else None
        self.video_path = Path(video_path) if video_path else None
        self.question = None
        self.options = None
        self.required_resolution = None
        self.resolution_reason = None
        self.evidence_nodes = {}
        self.rounds = []
        self.answered = False
        self.nodes = {}

        def index(node, node_id):
            self.nodes[node_id] = node
            for i, child in enumerate(node.get("children", [])):
                index(child, f"{node_id}/{i}")

        index(self.tree, "root")

    def state(self):
        return deepcopy({
            "question": self.question,
            "options": self.options,
            "required_resolution": self.required_resolution,
            "resolution_reason": self.resolution_reason,
            "evidence_nodes": self.evidence_nodes,
            "rounds": self.rounds,
            "answered": self.answered,
        })

    def _save(self):
        state = self.state()
        if self.output_path:
            temporary = self.output_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
            temporary.replace(self.output_path)
        return state

    def initialize(self, question, options=None):
        ids = [f"root/{i}" for i in range(len(self.tree.get("children", [])))]
        if not ids:
            raise ValueError("The tree has no top-level evidence nodes")
        resolution = self._classify_resolution(question, options)
        self.question = question
        self.options = options
        self.required_resolution = resolution["required_resolution"]
        self.resolution_reason = resolution["reason"]
        self.evidence_nodes = {node_id: "Initial uniform coverage" for node_id in ids}
        self.rounds = []
        self.answered = False
        return self._save()

    def _classify_resolution(self, question, options):
        schema = {
            "type": "object",
            "properties": {
                "required_resolution": {
                    "type": "string", "enum": ["coarse", "fine"]
                },
                "reason": {"type": "string", "maxLength": 400},
            },
            "required": ["required_resolution", "reason"],
            "additionalProperties": False,
        }
        prompt = f"""Classify the evidence resolution required to answer the
question before seeing any video evidence. Fine means the decisive distinction
depends on detail that a broad snapshot or summary cannot directly establish.
This includes events or transitions within broad video segments, action order,
repetition, cadence, frequency, brief changes, identity continuity, precise
temporal boundaries, small visual targets, exact counts, close geometric or
capacity estimates, and answer choices separated by subtle visible details.
Coarse means broad snapshots can directly establish the answer. Broad temporal
coverage does not imply coarse resolution: periodicity and frequency require
broad coverage plus fine observation of events between snapshots. Conversely,
a gradual scene-level trend or broad change in appearance can require evidence
from separated times while remaining coarse when those snapshots directly show
the relevant states. Classify the resolution of the decisive visual distinction,
not merely whether the question mentions change or the whole video.
Question: {question}
Options: {json.dumps(options)}"""
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "format": schema,
            "stream": False,
            "think": False,
            "keep_alive": os.getenv("SKYTREE_KEEP_ALIVE", "30m"),
            "options": {"num_ctx": 4096, "num_predict": 500, "temperature": 0},
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

    def _evidence_images(self, ids):
        if not self.video_path:
            return None, None
        video = cv2.VideoCapture(str(self.video_path))
        if not video.isOpened():
            raise ValueError(f"Could not open {self.video_path}")
        count = min(MAX_FRAMES_PER_NODE,
                    max(1, FRONTIER_FRAME_BUDGET // len(ids)))
        images = []
        image_order = []
        try:
            for node_id in ids:
                node = self.nodes[node_id]
                node_count = min(count, node["end"] - node["start"])
                indices = uniform_samples(node["start"], node["end"], node_count)
                encoded_frames = sample_frames(
                    video, node["start"], node["end"], node_count
                )
                for index, encoded in zip(indices, encoded_frames):
                    images.append(encoded)
                    image_order.append({"node_id": node_id, "frame": index})
        finally:
            video.release()
        return images, image_order

    def _ask(self, ids, validation_feedback=None):
        action_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string", "enum": ids},
                "action": {"type": "string", "enum": ["retain", "expand", "prune"]},
                "reason": {"type": "string", "maxLength": 240},
                "unresolved_distinction": {"type": "string", "maxLength": 200},
            },
            "required": ["id", "action", "reason", "unresolved_distinction"],
            "additionalProperties": False,
        }
        schema = {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "maxLength": 600},
                "actions": {"type": "array", "minItems": len(ids),
                            "maxItems": len(ids), "items": action_schema},
                "answer": {"type": "boolean"},
            },
            "required": ["reasoning", "actions", "answer"],
            "additionalProperties": False,
        }
        nodes = []
        for node_id in ids:
            node = self.nodes[node_id]
            children = [{"id": f"{node_id}/{i}", "start": child.get("start"),
                         "end": child.get("end")}
                        for i, child in enumerate(node.get("children", []))]
            nodes.append({"id": node_id, "start": node.get("start"),
                          "end": node.get("end"), "summary": node.get("summary"),
                          "children": children})

        images = None
        image_order = None
        if self.video_path and any(not self.nodes[node_id].get("summary") for node_id in ids):
            images, image_order = self._evidence_images(ids)
        context = {
            "question": self.question,
            "options": self.options,
            "required_resolution": self.required_resolution,
            "resolution_reason": self.resolution_reason,
            "nodes": nodes,
            "image_order": image_order,
            "visual_evidence": (
                "Each supplied image is a separate full-resolution frame. "
                "Match images to nodes and frame indices using image_order."
                if images else None
            ),
            "frontier_frame_budget": FRONTIER_FRAME_BUDGET if images else None,
            "previous_rounds": self.rounds[-2:],
            "validation_feedback": validation_feedback,
        }
        if self.video_path:
            video = cv2.VideoCapture(str(self.video_path))
            context["fps"] = video.get(cv2.CAP_PROP_FPS) or None
            video.release()
        system = (POLICY_PROMPT + "\nThe required resolution was decided from the "
                  "question before viewing evidence and is locked; do not redefine it. "
                  "Keep the overall reasoning and each action reason concise.")
        for attempt in range(2):
            content = json.dumps(context)
            if "qwen" in MODEL.lower():
                content = "/no_think\n" + content
            if attempt:
                content += "\nReturn concise complete JSON."
            user = {"role": "user", "content": content}
            if images:
                user["images"] = [base64.b64encode(image).decode() for image in images]
            payload = {
                "model": MODEL,
                "messages": [{"role": "system", "content": system}, user],
                "format": schema,
                "stream": False,
                "think": False,
                "keep_alive": os.getenv("SKYTREE_KEEP_ALIVE", "30m"),
                "options": {"num_ctx": 16384 if images else 4096,
                            "num_predict": 2200, "temperature": 0},
            }
            request = Request("http://localhost:11434/api/chat",
                              data=json.dumps(payload).encode(),
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=900) as response:
                reply = json.load(response)["message"]
            text = reply.get("content", "").strip() or reply.get("thinking", "").strip()
            try:
                start = text.find("{")
                result, _ = json.JSONDecoder().raw_decode(text[start:])
                return result
            except (ValueError, json.JSONDecodeError):
                if attempt:
                    raise ValueError(f"{MODEL} returned invalid adaptive-search JSON")

    def _step_once(self, validation_feedback=None):
        if self.answered:
            raise ValueError("Search has already answered")
        ids = list(self.evidence_nodes)
        result = self._ask(ids, validation_feedback)
        actions = result.get("actions")
        if not isinstance(actions, list) or len(actions) != len(ids):
            raise ValueError("Every evidence node requires exactly one action")

        by_id = {}
        expected_fields = {"id", "action", "reason", "unresolved_distinction"}
        for item in actions:
            if (not isinstance(item, dict) or set(item) != expected_fields
                    or item["id"] not in ids or item["id"] in by_id
                    or item["action"] not in {"retain", "expand", "prune"}
                    or not isinstance(item["reason"], str)
                    or not isinstance(item["unresolved_distinction"], str)):
                raise ValueError("Invalid adaptive evidence action")
            if item["action"] == "expand":
                if not self.nodes[item["id"]].get("children"):
                    raise ValueError("A leaf node cannot expand")
                if not item["unresolved_distinction"].strip():
                    raise ValueError("Expansion requires an unresolved distinction")
            by_id[item["id"]] = item

        if (set(by_id) != set(ids)
                or not isinstance(result.get("reasoning"), str)
                or type(result.get("answer")) is not bool):
            raise ValueError("Invalid adaptive search response")
        if result["answer"] and any(item["action"] == "expand" for item in actions):
            raise ValueError("Cannot expand and answer in the same round")
        if (self.required_resolution == "fine" and result["answer"]
                and all(node_id.count("/") == 1 for node_id in ids)
                and any(self.nodes[node_id].get("children") for node_id in ids)):
            raise ValueError(
                "Fine-grained questions cannot be answered using only top-level nodes; "
                "expand the relevant coarse branches"
            )

        next_evidence = {}
        changed = False
        for node_id in ids:
            item = by_id[node_id]
            if item["action"] == "retain":
                next_evidence[node_id] = item["reason"]
            elif item["action"] == "expand":
                changed = True
                for i, _ in enumerate(self.nodes[node_id]["children"]):
                    next_evidence[f"{node_id}/{i}"] = item["reason"]
            else:
                changed = True

        if not next_evidence:
            raise ValueError("Search cannot remove all evidence")
        if not result["answer"] and not changed:
            raise ValueError("A continuing round must refine or prune evidence")

        self.rounds.append({
            "round": len(self.rounds) + 1,
            "required_resolution": self.required_resolution,
            "resolution_reason": self.resolution_reason,
            "reasoning": result["reasoning"],
            "before": ids,
            "actions": actions,
            "after": list(next_evidence),
            "answer": result["answer"],
        })
        self.evidence_nodes = next_evidence
        self.answered = result["answer"]
        return self._save()

    def step(self):
        feedback = None
        for attempt in range(2):
            try:
                return self._step_once(feedback)
            except ValueError as error:
                if attempt or self.answered:
                    raise
                feedback = str(error)


def main():
    parser = argparse.ArgumentParser(description="Run adaptive video evidence search")
    parser.add_argument("tree", type=Path)
    parser.add_argument("question")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--options", type=Path)
    args = parser.parse_args()
    tree = json.loads(args.tree.read_text())
    options = json.loads(args.options.read_text()) if args.options else None
    search = AdaptiveEvidenceSearch(tree,
                                    output_path=args.tree.with_name("search_trace.json"),
                                    video_path=args.video)
    search.initialize(args.question, options)
    while not search.answered:
        search.step()
    print(json.dumps(search.state(), indent=2))


if __name__ == "__main__":
    main()

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
from pprint import pprint
from urllib.request import Request, urlopen

MODEL = "hf.co/google/gemma-4-31B-it-qat-q4_0-gguf:latest"
TOP_K = 4

BASE_PROMPT = """Choose relevant video nodes to investigate the supplied question.
Rank selections by usefulness and give a reason for each. Select only supplied
IDs, without duplicates. The selection limit is a maximum, not a target.
Summaries are sparse: omission is not proof of absence. Plan the search, do not
answer the question or invent observations. start/end are frame indices, with
end exclusive. Return only the requested JSON fields; no search instructions.
If answer options are supplied, preserve their distinctions and gather evidence
that helps choose among them.
Keep the global_summary concise and current. Name what each active branch is
investigating, what is resolved, and which branches are redundant.
"""
ORCHESTRATOR_PROMPT = """You are the orchestrator initializing a fresh search.
Define a concise goal_summary and min_nodes, the hard minimum number of distinct
evidence nodes to retain, with minimum_reason. min_nodes must be at least one
and no larger than your selection count.
Always include the original question verbatim inside goal_summary, followed by
your concise interpretation of what evidence is needed to answer it.
Initialize global_summary with the role of every selected branch. Call out
redundant branches when they add no useful evidence.
Before selecting nodes, decide whether answering the question requires evidence
across multiple frames or parts of the video. If it does, enforce that need in
goal_summary and min_nodes, and choose broader, distributed seeds that together
cover the required evidence. Do not infer a global answer from one local node.
If it does not, choose the smallest useful local evidence.
"""
CHILD_PROMPT = """Choose immediate children of the supplied active parent.
Use the stored question and goal. Children replace this parent in the global
active set. Respect selection_limit and minimum_selection; zero children prunes
this branch when permitted. Do not redefine the goal or minimum evidence count.
Update global_summary so it describes every active branch, including which
branch is handling each unresolved part of the question and which branches are
now redundant.
Use prune_nodes to remove redundant active sibling branches when the global
summary shows they add no new evidence. Never prune below minimum_selection.
"""
ROW_PROMPT = """Choose the next active row from the complete supplied pool.
The pool contains the children of every active non-leaf node, plus any active
leaves. Compare all candidates together before discarding anything. Preserve
the stored goal and its local or cross-frame evidence needs. Respect
selection_limit and minimum_selection. Update global_summary to describe the
roles of the selected row. Do not redefine the goal or minimum evidence count.
"""


class NodeChooser:
    def __init__(self, tree, output_path=None):
        self.tree = deepcopy(tree)
        if isinstance(self.tree, list):
            self.tree = {"children": self.tree}
        self.output_path = Path(output_path) if output_path else None
        self.max_nodes = TOP_K
        self.question = None
        self.options = None
        self.goal_summary = None
        self.global_summary = ""
        self.min_nodes = None
        self.minimum_reason = None
        self.active_nodes = {}
        self.progress = {}
        self.nodes = {}

        def index(node, node_id):
            self.nodes[node_id] = node
            for i, child in enumerate(node.get("children", [])):
                index(child, f"{node_id}/{i}")
        index(self.tree, "root")

    def state(self):
        return deepcopy({"question": self.question, "options": self.options,
                         "goal_summary": self.goal_summary,
                         "global_summary": self.global_summary,
                         "max_nodes": self.max_nodes, "min_nodes": self.min_nodes,
                         "minimum_reason": self.minimum_reason,
                         "active_nodes": self.active_nodes, "progress": self.progress})

    def _commit(self, state):
        if self.output_path:
            temporary = self.output_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
            temporary.replace(self.output_path)
        for key, value in state.items():
            setattr(self, key, value)
        return self.state()

    def _entry(self, selection):
        leaf = not self.nodes[selection["id"]].get("children")
        return {"reason": selection["reason"], "status": "leaf" if leaf else "active",
                "children": {}}

    def _ask(self, supplement, context, ids, minimum, maximum,
             orchestrator=False, prunable_ids=None):
        selection_schema = {
            "type": "object", "properties": {
                "id": {"type": "string", "enum": ids},
                "reason": {"type": "string"}},
            "required": ["id", "reason"], "additionalProperties": False}
        properties = {"selected_nodes": {"type": "array", "minItems": minimum,
                      "maxItems": maximum, "items": selection_schema},
                      "global_summary": {"type": "string"},
                      "prune_nodes": {"type": "array", "minItems": 0,
                                      "maxItems": len(prunable_ids or []),
                                      "items": {"type": "string"}}}
        if prunable_ids:
            properties["prune_nodes"]["items"] = {"type": "string",
                                                    "enum": prunable_ids}
        if orchestrator:
            properties.update({"goal_summary": {"type": "string"},
                               "minimum_reason": {"type": "string"},
                               "min_nodes": {"type": "integer", "minimum": 1,
                                             "maximum": maximum}})
        context = {**context, "selection_limit": maximum, "minimum_selection": minimum,
                   "prunable_nodes": prunable_ids or [],
                   "nodes": [{"id": node_id, **{key: self.nodes[node_id].get(key)
                              for key in ("start", "end", "summary")}} for node_id in ids]}
        payload = {"model": MODEL, "messages": [
            {"role": "system", "content": BASE_PROMPT + supplement},
            {"role": "user", "content": json.dumps(context)}],
            "format": {"type": "object", "properties": properties,
                       "required": list(properties), "additionalProperties": False},
            "stream": False, "think": False,
            "keep_alive": os.getenv("SKYTREE_KEEP_ALIVE", "0"),
            "options": {"num_ctx": 4096, "num_predict": 800, "temperature": 0}}
        request = Request("http://localhost:11434/api/chat", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=600) as response:
            text = json.load(response)["message"]["content"].strip()
            if text.startswith("```"):
                text = "\n".join(text.splitlines()[1:-1]).strip()
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                text = text[start:end + 1]
            return json.loads(text)

    @staticmethod
    def _validate(result, ids, minimum, maximum, orchestrator=False,
                  prunable_ids=()):
        expected = {"selected_nodes", "global_summary", "prune_nodes"}
        if orchestrator:
            expected |= {"goal_summary", "min_nodes", "minimum_reason"}
        if not isinstance(result, dict) or set(result) != expected:
            raise ValueError("Unexpected model response fields")
        selected = result["selected_nodes"]
        if not isinstance(selected, list) or not minimum <= len(selected) <= maximum:
            raise ValueError("Selection violates node limits")
        seen = set()
        for item in selected:
            if (not isinstance(item, dict) or set(item) != {"id", "reason"}
                    or not isinstance(item["id"], str) or item["id"] not in ids
                    or item["id"] in seen or not isinstance(item["reason"], str)):
                raise ValueError("Invalid or duplicate selected node")
            seen.add(item["id"])
        pruned = result["prune_nodes"]
        if (not isinstance(pruned, list) or len(set(pruned)) != len(pruned)
                or any(node_id not in prunable_ids for node_id in pruned)
                or set(pruned) & seen):
            raise ValueError("Invalid or conflicting pruned node")
        if orchestrator and pruned:
            raise ValueError("The orchestrator cannot prune before selecting branches")
        if orchestrator:
            if (type(result["min_nodes"]) is not int
                    or not 1 <= result["min_nodes"] <= len(selected)
                    or not isinstance(result["goal_summary"], str)
                    or not isinstance(result["minimum_reason"], str)):
                raise ValueError("Invalid search goal or minimum")
        if not isinstance(result["global_summary"], str):
            raise ValueError("Invalid global summary")
        return selected, pruned

    def orchestrate(self, question, level=1, options=None):
        if type(level) is not int or level < 0:
            raise ValueError("level must be a nonnegative integer")
        ids = [node_id for node_id in self.nodes if node_id.count("/") == level]
        if not ids:
            raise ValueError(f"No nodes at level {level}")
        maximum = min(self.max_nodes, len(ids))
        result = self._ask(ORCHESTRATOR_PROMPT,
                           {"question": question, "options": options, "level": level},
                           ids, 1, maximum, orchestrator=True)
        selected, _ = self._validate(result, ids, 1, maximum, orchestrator=True)
        state = {"question": question, "options": options,
                 "goal_summary": result["goal_summary"],
                 "global_summary": result["global_summary"],
                 "max_nodes": self.max_nodes, "min_nodes": result["min_nodes"],
                 "minimum_reason": result["minimum_reason"],
                 "active_nodes": {item["id"]: item["reason"] for item in selected},
                 "progress": {item["id"]: self._entry(item) for item in selected}}
        return self._commit(state)

    def choose_children(self, node_id):
        if node_id not in self.active_nodes:
            raise ValueError("Choose an active node after orchestration")
        children = self.nodes[node_id].get("children", [])
        if not children:
            self._commit(self.state())
            return []
        ids = [f"{node_id}/{i}" for i in range(len(children))]
        remaining = len(self.active_nodes) - 1
        minimum = max(0, self.min_nodes - remaining)
        maximum = len(ids)
        prunable_ids = [active_id for active_id in self.active_nodes if active_id != node_id]
        result = self._ask(CHILD_PROMPT, {"question": self.question,
                           "options": self.options,
                           "goal_summary": self.goal_summary, "min_nodes": self.min_nodes,
                           "max_nodes": self.max_nodes, "active_nodes": self.active_nodes,
                           "global_summary": self.global_summary,
                           "parent_id": node_id,
                           "available_capacity": self.max_nodes - remaining},
                          ids, minimum, maximum, prunable_ids=prunable_ids)
        selected, pruned = self._validate(result, ids, minimum, maximum,
                                           prunable_ids=prunable_ids)
        final_count = remaining - len(pruned) + len(selected)
        if not self.min_nodes <= final_count <= self.max_nodes:
            raise ValueError("Pruning and selection violate the active-node limits")
        state = self.state()
        for pruned_id in pruned:
            del state["active_nodes"][pruned_id]
        state["global_summary"] = result["global_summary"]
        del state["active_nodes"][node_id]
        state["active_nodes"].update({item["id"]: item["reason"] for item in selected})

        def find(entries, wanted):
            if wanted in entries:
                return entries[wanted]
            for entry in entries.values():
                match = find(entry["children"], wanted)
                if match is not None:
                    return match
        for pruned_id in pruned:
            find(state["progress"], pruned_id)["status"] = "pruned"
        entry = find(state["progress"], node_id)
        entry["status"] = "expanded" if selected else "pruned"
        entry["children"] = {item["id"]: self._entry(item) for item in selected}
        self._commit(state)
        return [item["id"] for item in selected]

    def choose_next_row(self):
        """Replace the complete active row with one globally selected child row."""
        parents = list(self.active_nodes)
        ids = []
        for parent_id in parents:
            children = self.nodes[parent_id].get("children", [])
            if children:
                ids.extend(f"{parent_id}/{i}" for i in range(len(children)))
            else:
                ids.append(parent_id)

        if ids == parents:  # Every active node is already a leaf.
            return []

        maximum = min(self.max_nodes, len(ids))
        if len(ids) < self.min_nodes:
            raise ValueError("The next row cannot satisfy min_nodes")
        result = self._ask(ROW_PROMPT, {
            "question": self.question,
            "options": self.options,
            "goal_summary": self.goal_summary,
            "min_nodes": self.min_nodes,
            "global_summary": self.global_summary,
            "current_active_nodes": self.active_nodes,
        }, ids, self.min_nodes, maximum)
        selected, _ = self._validate(result, ids, self.min_nodes, maximum)
        selected_by_id = {item["id"]: item for item in selected}

        state = self.state()

        def find(entries, wanted):
            if wanted in entries:
                return entries[wanted]
            for entry in entries.values():
                match = find(entry["children"], wanted)
                if match is not None:
                    return match

        for parent_id in parents:
            entry = find(state["progress"], parent_id)
            children = self.nodes[parent_id].get("children", [])
            if not children:
                if parent_id not in selected_by_id:
                    entry["status"] = "pruned"
                continue
            child_ids = [f"{parent_id}/{i}" for i in range(len(children))]
            chosen_children = [child_id for child_id in child_ids
                               if child_id in selected_by_id]
            entry["status"] = "expanded" if chosen_children else "pruned"
            entry["children"] = {}
            for child_id in child_ids:
                if child_id in selected_by_id:
                    child_entry = self._entry(selected_by_id[child_id])
                else:
                    child_entry = {"reason": "Not selected in row decision",
                                   "status": "pruned", "children": {}}
                entry["children"][child_id] = child_entry

        state["active_nodes"] = {
            item["id"]: item["reason"] for item in selected
        }
        state["global_summary"] = result["global_summary"]
        self._commit(state)
        return list(state["active_nodes"])


def main():
    parser = argparse.ArgumentParser(description="Choose video tree nodes using Gemma")
    parser.add_argument("tree", type=Path)
    parser.add_argument("question")
    parser.add_argument("--level", type=int, default=1)
    args = parser.parse_args()
    chooser = NodeChooser(json.loads(args.tree.read_text()),
                          output_path=args.tree.with_name("traversal_plan.json"))
    pprint(chooser.orchestrate(args.question, args.level), sort_dicts=False)
    print(f"Saved to {chooser.output_path}")


if __name__ == "__main__":
    main()

"""Loader for PlanBench Blocksworld PDDL instances (Valmeekam et al., 2023).

Parses :init into a support map (block -> "TABLE" or the block below) and
:goal into (block, support) pairs for the Blocksworld grammar compiler.
PlanBench goals are partial: only some "on" relations are constrained.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def _strip_pddl_comments(text: str) -> str:
    return re.sub(r";[^\n]*", "", text)


def _extract_section(text: str, section: str) -> Optional[str]:
    """Find the (:init ...) or (:goal ...) section and return its body."""
    pattern = re.compile(rf"\(\s*:{section}\s")
    m = pattern.search(text)
    if not m:
        return None
    # Walk parens to find balanced close.
    depth = 0
    start = m.start()
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start + len(f"(:{section}"):i].strip()
    return None


def _parse_atoms(body: str) -> list[tuple[str, ...]]:
    """Parse atomic predicates from a section body."""
    atoms = []
    for m in re.finditer(r"\(([^()]+)\)", body):
        toks = m.group(1).split()
        if toks:
            atoms.append(tuple(t.strip() for t in toks))
    return atoms


# Mystery Blocksworld (Valmeekam et al. 2023) renames predicates and
# actions; the semantics are unchanged, so the grammar construction is
# unaffected.

MYSTERY_PREDICATE_MAP = {
    # Mystery name -> Blocksworld name
    "harmony": "handempty",
    "province": "clear",
    "planet": "ontable",
    "craves": "on",
    "pain": "holding",
}

MYSTERY_ACTION_NAMES = {
    # Blocksworld action -> Mystery action (for output grammar)
    "pickup": "attack",
    "putdown": "succumb",
    "stack": "overcome",
    "unstack": "feast",
}


def parse_pddl_instance(text: str, domain: str = "blocksworld") -> dict:
    """Parse a PlanBench Blocksworld PDDL instance.

    With domain='mystery' the obfuscated predicate names are mapped back
    to Blocksworld ones for grammar compilation; the grammar then emits
    Mystery action names, so the model sees the obfuscated task.

    Returns a dict with labels (block names), initial_support (block ->
    'TABLE' or the block below), goal_pairs (tuple of (block, support)
    constraints), held (the held block or None) and domain.
    """
    text = _strip_pddl_comments(text)
    # Object list: (:objects a b c d)
    obj_match = re.search(r"\(\s*:objects\s+([^)]+)\)", text)
    labels: list[str] = []
    if obj_match:
        labels = [t for t in obj_match.group(1).split() if t]

    init_body = _extract_section(text, "init") or ""
    goal_body = _extract_section(text, "goal") or ""

    # Translate mystery predicates back to Blocksworld semantics.
    def _translate(atoms: list[tuple]) -> list[tuple]:
        if domain != "mystery":
            return atoms
        out = []
        for a in atoms:
            new_name = MYSTERY_PREDICATE_MAP.get(a[0], a[0])
            out.append((new_name,) + a[1:])
        return out

    init_atoms = _translate(_parse_atoms(init_body))
    goal_atoms = _translate(_parse_atoms(goal_body))

    initial_support: dict[str, str] = {}
    held: Optional[str] = None
    for atom in init_atoms:
        if atom[0] == "ontable":
            initial_support[atom[1]] = "TABLE"
        elif atom[0] == "on":
            initial_support[atom[1]] = atom[2]
        elif atom[0] == "holding":
            held = atom[1]
        # clear, handempty are derived, not stored
    seen_blocks = set(initial_support.keys()) | {
        sup for sup in initial_support.values() if sup != "TABLE"
    }
    if held: seen_blocks.add(held)
    for b in seen_blocks:
        if b not in labels:
            labels.append(b)

    goal_pairs: list[tuple[str, str]] = []
    for atom in goal_atoms:
        if atom[0] == "on":
            goal_pairs.append((atom[1], atom[2]))
        elif atom[0] == "ontable":
            goal_pairs.append((atom[1], "TABLE"))

    return {
        "labels": labels,
        "initial_support": initial_support,
        "goal_pairs": tuple(goal_pairs),
        "held": held,
        "domain": domain,
    }


def load_planbench_instances(directory: Path, limit: Optional[int] = None,
                              domain: str = "blocksworld") -> list[dict]:
    """Load PlanBench Blocksworld (or Mystery) instances from a directory."""
    files = sorted(directory.glob("instance-*.pddl"),
                   key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    if limit is not None:
        files = files[:limit]
    out = []
    for f in files:
        text = f.read_text()
        parsed = parse_pddl_instance(text, domain=domain)
        parsed["instance_id"] = f.stem
        out.append(parsed)
    return out


__all__ = ["parse_pddl_instance", "load_planbench_instances"]

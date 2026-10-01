"""Strip `(not ...)` preconditions from a PDDL domain so pyperplan can parse it.

Drops `:negative-preconditions` from `:requirements` and removes `(not ...)`
inside `:precondition` blocks; negative effects are kept. The resulting
schemas are under-constrained, but a violated negative precondition yields
a no-op transition in rollouts, which induction skips (`sb == sa`).
"""
from __future__ import annotations

import argparse
import re


def strip_negative_pre(text: str) -> str:
    text = re.sub(r":negative-preconditions", "", text)

    out_lines = []
    i = 0
    tokens = text
    def _strip_not_in_block(block: str) -> str:
        """Remove every `(not ...)` form from a precondition block."""
        result = []
        depth = 0
        i = 0
        while i < len(block):
            if (block[i] == "(" and
                block[i:i+5] == "(not "):
                d = 1
                j = i + 1
                while j < len(block) and d > 0:
                    if block[j] == "(": d += 1
                    elif block[j] == ")": d -= 1
                    j += 1
                i = j
                continue
            result.append(block[i])
            i += 1
        return "".join(result)

    def process_pre_block(m):
        prefix = m.group(1)
        inner = m.group(2)
        return prefix + _strip_not_in_block(inner)

    # Match ":precondition" followed by a balanced parenthetical
    def find_and_strip(text):
        out = []
        i = 0
        while i < len(text):
            idx = text.find(":precondition", i)
            if idx < 0:
                out.append(text[i:])
                break
            out.append(text[i:idx])
            j = idx + len(":precondition")
            while j < len(text) and text[j].isspace():
                j += 1
            if j >= len(text) or text[j] != "(":
                out.append(text[idx:])
                break
            d = 1
            k = j + 1
            while k < len(text) and d > 0:
                if text[k] == "(": d += 1
                elif text[k] == ")": d -= 1
                k += 1
            inner_block = text[j:k]
            stripped = _strip_not_in_block(inner_block)
            out.append(":precondition " + stripped)
            i = k
        return "".join(out)

    return find_and_strip(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    text = open(args.input).read()
    stripped = strip_negative_pre(text)
    with open(args.output, "w") as f:
        f.write(stripped)
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()

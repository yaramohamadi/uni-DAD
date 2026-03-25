import matplotlib
matplotlib.use('Agg')
import re
from typing import List, Tuple, Union
from pathlib import Path
import ast, re

# This module keeps the pause-time prompt parsing logic independent from the
# trainer so generation/evaluation can reuse the same prompt-file heuristics.
# ---- paused-generation helpers (reuse generate.py) ----
from pathlib import Path
# Prefer the shared generate.py helpers when they are importable so offline
# scripts and in-training paused generation interpret prompt files the same way.
try:
    from scripts.generate import extract_prompt_blocks, read_instances, expand_prompt, LIVE_INSTANCES
except Exception:
    # Minimal fallbacks if generate.py isn't available
    LIVE_INSTANCES = {
        ("dog", "dog"), ("dog2", "dog"), ("dog3", "dog"), ("dog5", "dog"),
        ("dog6", "dog"), ("dog7", "dog"), ("dog8", "dog"),
        ("cat", "cat"), ("cat2", "cat"),
    }


# Prompt files in this project mix Python-ish list syntax, copied shell text,
# and plain prompt lines. The helpers below collapse those formats into a
# single template representation before the trainer expands tokens into text.
def _clean_prompt_token(s: str) -> str:
    """
    Make a single raw line like:
      "prompt_list = ["
      "'a {0} {1} in the jungle'.format(unique_token, class_token),"
    become a plain template:
      "a {0} {1} in the jungle"
    Also strips surrounding quotes and trailing commas/brackets.
    """
    s = s.strip()

    # Drop leading assignment tokens
    s = re.sub(r'^\s*prompt_list\s*=\s*\[?\s*$', '', s)

    # Drop trailing commas and solitary brackets
    s = s.rstrip(",").strip()
   

    # If it's "'...'.format(unique_token, class_token)" → extract the quoted part
    m = re.match(
        r"""^[\'"](?P<body>.+?)[\'"]\s*\.\s*format\s*\(\s*unique_token\s*,\s*class_token\s*\)\s*$""",
        s
    )
    if m:
        s = m.group("body").strip()

    # Strip surrounding quotes (if any remain)
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1]

    return s.strip()

def _clean_prompt_list(lines: List[str]) -> List[str]:
    out = []
    for ln in lines:
        t = _clean_prompt_token(ln)
        if t:
            out.append(t)
    return out

def extract_prompt_blocks(prompts_file: Union[str, Path]) -> Tuple[List[str], List[str]]:
    p = Path(prompts_file)
    lines = p.read_text(encoding="utf-8").splitlines()

    def clean_line(s: str) -> str:
        s = s.strip()
        if not s: return ""
        s = re.sub(r'^\s*(?:[-*•]\s+|\d+\.\s+)', '', s)  # drop bullets/numbering
        s = s.rstrip(',')
        if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
            s = s[1:-1]
        return s.strip()

    # Scan forward from a header marker and keep collecting lines until the
    # next header-like section starts. This makes the parser tolerant to human
    # edited prompt files rather than requiring a strict schema.
    def collect_after(header_keywords: List[str]) -> List[str]:
        out, take = [], False
        for raw in lines:
            line  = raw.strip()
            upper = line.upper()
            if all(k.upper() in upper for k in header_keywords):
                take = True
                continue
            if not take:
                continue
            # stop on next header-ish all-caps line
            if upper and upper.isupper() and len(line) < 80 and any(
                k in upper for k in ["PROMPTS", "OBJECT", "LIVE", "SUBJECT", "CLASSES"]
            ):
                break
            if not line or line.startswith("#"):
                continue
            if line.startswith("["):
                try:
                    arr = ast.literal_eval(line)
                    if isinstance(arr, list):
                        for it in arr:
                            s = clean_line(str(it))
                            if s: out.append(s)
                        continue
                except Exception:
                    pass
            s = clean_line(line)
            if s: out.append(s)
        return out

    obj  = collect_after(["OBJECT", "PROMPTS"])
    live = collect_after(["LIVE", "PROMPTS"]) or collect_after(["LIVE", "SUBJECT", "PROMPTS"])

    object_prompts = _clean_prompt_list(obj)
    live_prompts   = _clean_prompt_list(live)

    # If section-based parsing fails, salvage any obvious Python-style prompt
    # templates directly from the raw file so paused generation can continue.
    # fallback: quoted strings with .format(...unique..., ...class...)
    if not object_prompts and not live_prompts:
        raw = p.read_text(encoding="utf-8")
        cand = re.findall(
            r"""["']([^"']+)["']\s*\.\s*format\s*\(\s*[^,]*unique[^,]*\s*,\s*[^)]*class[^)]*\)""",
            raw, flags=re.IGNORECASE
        )
        object_prompts  = cand[:]
        live_prompts = cand[:]

    if object_prompts and not live_prompts: live_prompts = object_prompts[:]
    if live_prompts and not object_prompts: object_prompts = live_prompts[:]

    # dedup preserve order
    def dedup(xs: List[str]) -> List[str]:
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    object_prompts  = dedup(object_prompts)
    live_prompts = dedup(live_prompts)

    # The caller expects a fixed-size prompt bank for deterministic grids/tables,
    # so short prompt lists are repeated in order and long ones are truncated.
    # normalize to exactly 25
    def to27(xs: List[str]) -> List[str]:
        if not xs: return []
        if len(xs) >= 27: return xs[:27]
        out, i, n = [], 0, len(xs)
        while len(out) < 27:
            out.append(xs[i % n]); i += 1
        return out

    return to27(object_prompts), to27(live_prompts)
    
def expand_prompt(template: str, unique_token: str, class_name: str) -> str:
    s = template
    # If someone passed the literal "'.format(unique_token, class_token)" form, strip it:
    m = re.match(
        r"""^[\'"]?(?P<body>.+?)[\'"]?\s*\.\s*format\s*\(\s*unique_token\s*,\s*class_token\s*\)\s*$""",
        s
    )
    if m:
        s = m.group("body").strip()
    # Support both named placeholders and legacy positional placeholders because
    # existing prompt files use both conventions.
    s = s.replace("{unique_token}", unique_token).replace("{class_token}", class_name)
    s = s.replace("{0}", unique_token).replace("{1}", class_name)
    return s.replace("_", " ")

# Instance extraction is intentionally heuristic: the same text file may carry
# instance/class rows, prompt templates, comments, or copied code snippets.
def read_instances(prompts_file: Union[str, Path]):
    p = Path(prompts_file)
    txt = p.read_text(encoding="utf-8").lstrip("\ufeff")  # strip UTF-8 BOM if present
    pairs: List[Tuple[str, str]] = []
    seen = set()

    # Anything containing these looks like prompts/code, not an instance row
    BAD_TOKENS = ("{0}", "{1}", "format(", "class_token", "unique_token",
                  "prompt_list", "=")

    # Normalize function for header detection
    def _norm_header(s: str) -> str:
        # lower-case and remove non-letters, so "Subject Name" -> "subjectname"
        return re.sub(r"[^a-z]", "", s.lower())

    # Parse the file conservatively and only keep rows that look like
    # "instance_name, class_name" after stripping comments and quotes.
    for raw in txt.splitlines():
        # Strip comments and surrounding spaces
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue

        # Bail early on obvious prompt/code/list lines
        if any(tok in line for tok in BAD_TOKENS):
            continue

        # Split on the first separator: comma, semicolon, or tab
        m = re.search(r"[,\t;]", line)
        if not m:
            continue
        sep_idx = m.start()
        left, right = line[:sep_idx].strip(), line[sep_idx+1:].strip()

        # Drop quotes
        if len(left) >= 2 and left[0] == left[-1] and left[0] in "\"'":
            left = left[1:-1].strip()
        if len(right) >= 2 and right[0] == right[-1] and right[0] in "\"'":
            right = right[1:-1].strip()

        if not left or not right:
            continue

        # Header detection (very tolerant)
        L = _norm_header(left)
        R = _norm_header(right)
        left_is_header  = L in {"subjectname", "subject", "instance", "name", "id"}
        right_is_header = R in {"class", "classname", "category", "label", "type"}
        if left_is_header and right_is_header:
            # e.g., "subject_name, class" — skip it
            continue

        # Validate formats
        # left: id-like (no spaces), allow alnum, _, -, / (if you use subfolders)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-/]*", left):
            continue
        # right: class label (allow spaces), alnum + space/_/-
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9 _\-]*", right):
            continue

        key = (left, right)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)

    # Synthetic fallbacks keep pause-time generation alive even when the prompt
    # file is malformed, which is preferable to crashing mid-training.
    if not pairs:
        # Keep pipeline alive even if file is malformed
        pairs = [(f"instance{i}", "object") for i in range(100)]

    return pairs
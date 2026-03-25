#Builds the baselines 

#!/usr/bin/env python3
import argparse
from pathlib import Path
import csv
import re

def _clean_prompt(s: str) -> str:
    s = str(s)
    s = re.sub(r"\.format\([^)]*\)\s*$", "", s)   # drop trailing .format(...)
    s = re.sub(r"\bprt\b", "", s)                 # drop 'prt' rare token if present
    return re.sub(r"\s{2,}", " ", s).strip()

def _load_prompts(prompts_file: Path):
    if not prompts_file:
        return None
    lines = [ln.rstrip("\n") for ln in open(prompts_file, "r", encoding="utf-8")]
    return [_clean_prompt(s) for s in lines]

def _load_inst2label(csv_path: Path):
    """
    CSV format:
        instance,label_id
        alice,12
        bob,7
    """
    if not csv_path:
        return {}
    m = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            inst = str(row["instance"]).strip()
            lab  = int(row["label_id"])
            m[inst] = lab
    return m

def _discover_steps(run_dir: Path):
    return sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("gen_")],
                  key=lambda p: p.name)

def _pdirs_for_step(step_dir: Path):
    return sorted([d for d in step_dir.iterdir() if d.is_dir() and d.name.startswith("p_")],
                  key=lambda p: p.name)

def _split_class_instance(run_dir: Path):
    # run_dir = output_dmd2/<class_token>_<instance>/<training_name>
    class_inst = run_dir.parent.name           # "<class>_<instance>"
    training_name = run_dir.name               # "<training_name>"
    # try to extract instance (everything after first "_")
    if "_" in class_inst:
        instance = class_inst.split("_", 1)[1]
    else:
        instance = class_inst
    return class_inst, instance, training_name

def write_manifest_for_step(
    run_dir: Path,
    step_dir: Path,
    out_csv: Path,
    prompts: list[str] ,
    inst2label: dict[str, int],
    src_root: Path ,
):
    class_inst, instance, training_name = _split_class_instance(run_dir)
    # Map p_idx -> prompt (if provided)
    prompt_map = {i: prompts[i] for i in range(len(prompts))} if prompts else {}

    rows = []
    for p_dir in _pdirs_for_step(step_dir):
        try:
            p_idx = int(p_dir.name.split("_")[1])
        except Exception:
            p_idx = None
        prompt = prompt_map.get(p_idx, "") if prompt_map else ""

        # gen dir is the actual folder with generated images for that prompt index
        gen_dir = p_dir

        # src location:
        #  - if src_root given, use <src_root>/<instance> folder (folder-based)
        #  - otherwise prefer LMDB (leave src_dir empty and carry src_label)
        src_dir = ""
        if src_root:
            cand = src_root / instance
            src_dir = str(cand) if cand.exists() else ""

        src_label = inst2label.get(instance, "")

        rows.append({
            "class_instance": class_inst,
            "instance": instance,
            "pdir": p_dir.name,
            "prompt": prompt,
            "gen_dir": str(gen_dir),
            "src_dir": src_dir,
            "src_label": src_label,
            "run_dir": str(run_dir),
            "step": int(step_dir.name.split("_")[1]),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "class_instance","instance","pdir","prompt",
            "gen_dir","src_dir","src_label","run_dir","step"
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out_csv, len(rows)


def main():
    ap = argparse.ArgumentParser("Build DMD2/LMDB-aware evaluation manifest")
    ap.add_argument("--run_dir", type=Path, required=True,
                    help="output_dmd2/<class>_<instance>/<training_name>")
    ap.add_argument("--step", type=int, default=None,
                    help="Specific generation step (e.g., 2000). If omitted, build for all steps.")
    ap.add_argument("--prompts_file", type=Path, default=None,
                    help="Optional: text file where line i corresponds to p_{i:03d}.")
    ap.add_argument("--instance_to_label_csv", type=Path, default=None,
                    help="CSV with columns: instance,label_id — for LMDB sourcing.")
    ap.add_argument("--src_root", type=Path, default=None,
                    help="Optional: path to real reference images on disk (folder-based). If omitted, use LMDB via src_label.")
    ap.add_argument("--out_dirname", type=str, default="eval",
                    help="Subfolder under run_dir to store manifests and results.")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    eval_dir = run_dir / args.out_dirname
    prompts = _load_prompts(args.prompts_file)
    inst2label = _load_inst2label(args.instance_to_label_csv)

    if args.step is not None:
        step_dir = run_dir / f"gen_{args.step:06d}"
        if not step_dir.is_dir():
            raise FileNotFoundError(f"Step directory not found: {step_dir}")
        out_csv = eval_dir / f"eval_manifest_step{args.step:06d}.csv"
        out_csv, n = write_manifest_for_step(
            run_dir, step_dir, out_csv, prompts, inst2label, args.src_root
        )
        print(f"Wrote {n} rows → {out_csv}")
    else:
        total = 0
        for step_dir in _discover_steps(run_dir):
            step_num = int(step_dir.name.split("_")[1])
            out_csv = eval_dir / f"eval_manifest_step{step_num:06d}.csv"
            _, n = write_manifest_for_step(
                run_dir, step_dir, out_csv, prompts, inst2label, args.src_root
            )
            print(f"[step {step_num}] {n} rows → {out_csv}")
            total += n
        print(f"Done. Total rows: {total}")

if __name__ == "__main__":
    main()


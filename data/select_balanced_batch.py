# select_balanced_batch.py
"""
Script to extract a small balanced subset from a PRM800K training JSON file.

python3 select_balanced_batch.py \
    --input /Users/ruoxinwang/Desktop/Ph.D/Math_Reasoning/data/train_prm800k.jsonl \
    --output train_prm800k_small.json \
    --pos_groups 50 \
    --neg_groups 50 \
    --seed 42

This will sample batch_size//2 positives and batch_size//2 negatives.
"""
import json
import random
import argparse
import os
import sys

def load_records(path):
    """Load JSON array or JSONL file into a list of dicts."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        recs = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line=line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: skipping invalid JSON line: {line}", file=sys.stderr)
        return recs

def main():
    parser = argparse.ArgumentParser(
        description="Select a balanced subset of complete groups from PRM800K data."
    )
    parser.add_argument("--input", required=True,
                        help="Path to train_prm800k.json or .jsonl")
    parser.add_argument("--output", required=True,
                        help="Path for output JSON with selected groups")
    parser.add_argument("--pos_groups", type=int, default=50,
                        help="Number of positive groups to sample")
    parser.add_argument("--neg_groups", type=int, default=50,
                        help="Number of negative groups to sample")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    # ensure output dir exists
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # load all records
    records = load_records(args.input)
    if not records:
        print(f"Error: no records loaded from {args.input}", file=sys.stderr)
        sys.exit(1)

    # group by 'id'
    groups = {}
    for rec in records:
        gid = rec.get('id')
        groups.setdefault(gid, []).append(rec)

    # compute group-level label (average score)
    pos_ids = []
    neg_ids = []
    for gid, recs in groups.items():
        scores = []
        for r in recs:
            val = r.get('score', r.get('accuracy', 0))
            try:
                scores.append(float(val))
            except:
                scores.append(0.0)
        avg = sum(scores) / len(scores)
        if avg >= 0.5:
            pos_ids.append(gid)
        else:
            neg_ids.append(gid)

    # check enough groups
    if len(pos_ids) < args.pos_groups or len(neg_ids) < args.neg_groups:
        parser.error(
            f"Not enough groups: have {len(pos_ids)} pos, {len(neg_ids)} neg, "
            f"need {args.pos_groups} pos and {args.neg_groups} neg."
        )

    # sample group ids
    selected_pos = random.sample(pos_ids, args.pos_groups)
    selected_neg = random.sample(neg_ids, args.neg_groups)

    # flatten records belonging to selected groups
    subset = []
    # for gid in selected_pos + selected_neg:
    #     subset.extend(groups[gid])
    for gid in selected_pos + selected_neg:
        subset.extend(groups[gid])
        subset.sort(key=lambda r: (r['id'], r['sid']))

    # shuffle order of all records
    random.shuffle(subset)

    # write out as JSON array
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(subset)} records: "
          f"{len(selected_pos)} pos-groups and {len(selected_neg)} neg-groups "
          f"(avg group size {len(subset)/(len(selected_pos)+len(selected_neg)):.1f})")

if __name__ == '__main__':
    main()


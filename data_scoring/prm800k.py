import json

in_f  = "/workspace/data/dreamprm/prm800k/phase2_test.jsonl"
out_f = "/workspace/DreamPRM/data/test_prm800k.json"

records = []
sample_id = 0

with open(in_f) as fin:
    for line in fin:
        ex = json.loads(line)
        instr = ex["question"]["problem"]
        # pull the official answer string
        ground_truth = ex["question"].get("ground_truth_answer", "")

        # iterate steps with a step‐index sid
        for sid, step_obj in enumerate(ex["label"]["steps"], start=1):
            comps = step_obj.get("completions", [])
            if not comps:
                continue

            # pick the human‐chosen completion or fallback to the highest‐rated
            idx = step_obj.get("chosen_completion")
            if idx is None or not (0 <= idx < len(comps)):
                # coerce None→0, then pick max
                numeric_ratings = [
                    c.get("rating") if isinstance(c.get("rating"), (int, float)) else 0
                    for c in comps
                ]
                idx = numeric_ratings.index(max(numeric_ratings))

            comp       = comps[idx]
            step_text  = comp.get("text", "").strip()
            raw_rating = comp.get("rating")
            rating     = raw_rating if isinstance(raw_rating, (int, float)) else 0

            # binarize accuracy
            accuracy = 1.0 if rating > 0 else 0.0

            sample_id += 1
            records.append({
                "id":           sample_id,         # unique sample ID
                "sid":          sid,               # step number within problem
                "input":        instr,             # full question prompt
                "add":          step_text,         # this single CoT step
                "ground_truth": ground_truth,      # correct final answer
                "image_path":   "",                # no image for PRM800K
                "dataset":      "prm800k",         # domain name
                "score":        rating,            # here: human rating (0/1)
                "times":        1,                 # default 1 annotation
                "accuracy":     accuracy           # 0 or 1
            })

# write as a single JSON array
with open(out_f, "w") as fout:
    json.dump(records, fout, indent=2)

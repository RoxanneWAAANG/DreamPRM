import json
from random import shuffle
from tqdm import tqdm

num_samples = 10000
in_f  = "/workspace/run1/data/prm800k/prm800k/data/phase2_train.jsonl"
out_f = "../data/train_prm800k.json"
# out_f = f"data/lower_data/test_prm800k_{num_samples}samples.json"

records = []
sampled = []
unique_question_id = 0

with open(in_f) as fin:
    pbar = tqdm(fin)
    for line in pbar:
        ex = json.loads(line)
        instr = ex["question"]["problem"]
        ground_truth = ex["question"].get("ground_truth_answer", "")
        reason = ex["label"]["finish_reason"]

        if reason not in {"solution", "found_error"}:
            continue

        prev_steps = []
        this_question_records = []
        skip_question = False

        for sid, step_obj in enumerate(ex["label"]["steps"], start=1):
            comps = step_obj.get("completions", [])
            if not comps:
                continue

            for comp in comps:
                rating = comp["rating"]
                if rating is None:
                    continue

                text = comp["text"].strip()
                step_text = f"Step {sid}: {text}\n\n"
                prev_steps.append(step_text)
                add_str = "".join(prev_steps)

                # Accuracy assignment logic
                if rating == 1:
                    accuracy = 1.0
                elif rating == 0:
                    accuracy = 0.5  # You can change this if you want
                elif rating == -1:
                    accuracy = 0.0
                else:
                    accuracy = 0.5  # Fallback default

                this_question_records.append({
                    "id": unique_question_id,       # same ID for steps in one question
                    "sid": sid,                     # step number
                    "input": instr,                 # full question
                    "add": add_str,                 # cumulative step reasoning
                    "ground_truth": ground_truth,   # final answer
                    "image_path": "",               # no image
                    "dataset": str(unique_question_id),  # instance-level dataset
                    "score": rating,                # original human score
                    "times": 1,
                    "accuracy": accuracy
                })

                break  # only take the first valid completion per step

        if this_question_records:
            records.extend(this_question_records)
            unique_question_id += 1

# Shuffle and sample
shuffle(records)
id_sid_pairs = set()
for record in records:
    pair = (record["id"], record["sid"])
    if pair not in id_sid_pairs:
        id_sid_pairs.add(pair)
        sampled.append(record)
    if len(sampled) >= num_samples:
        break

# Save output
with open(out_f, "w") as fout:
    json.dump(sampled, fout, indent=2)

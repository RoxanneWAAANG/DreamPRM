import json
from random import sample
from tqdm import tqdm
from collections import defaultdict

# —— 配置区 —— 
num_per_bucket = 400
in_f  = "/workspace/data/prm800k/prm800k/data/phase2_train.jsonl"
out_f = "/workspace/DreamPRM/data/train_prm800k.json"

# —— 1. 读取并按 id 聚合所有步骤 —— 
questions = defaultdict(list)
with open(in_f) as fin:
    for sample_id, line in enumerate(tqdm(fin, desc="Loading")):
        ex = json.loads(line)
        reason = ex["label"]["finish_reason"]
        if reason not in {"solution", "found_error"}:
            continue

        instr = ex["question"]["problem"]
        gt    = ex["question"].get("ground_truth_answer", "")
        prev  = ""
        for sid, step in enumerate(ex["label"]["steps"], start=1):
            comps = step.get("completions", [])
            if not comps:
                continue
            comp = comps[0]
            if comp.get("rating") is None:
                continue

            text = comp["text"].strip()
            rec = {
                "id":           sample_id,  # unique sample ID
                "sid":          sid,    # step number within problem
                "input":        instr,  # full question prompt
                "add":          prev + f"Step {sid}: {text}\n\n<extra_0>",   # this single CoT step
                "ground_truth": gt, # correct final answer
                "image_path":   "", # no image for PRM800K
                "dataset":      str(sample_id), # domain name
                "score":        comp["rating"], # human rating {-1, 0, 1}
                "times":        1,  # default 1 annotation
                "accuracy":     comp["rating"] * 0.5 + 0.5  # {0, 0.5, 1}
            }
            questions[sample_id].append(rec)
            prev = rec["add"]

# —— 2. 计算每个 id 的“最终评分”（即最大 sid 的那条记录的 score） —— 
final_score = {}
for qid, recs in questions.items():
    if not recs:
        continue
    # 假设 recs 已按 sid 增序添加
    final_score[qid] = recs[-1]["score"]

# —— 3. 按最终评分分桶 —— 
buckets = defaultdict(list)   # key = -1, 0, 1
for qid, score in final_score.items():
    buckets[score].append(qid)

# —— 4. 每桶各随机抽取 num_per_bucket 个 id —— 
sampled_ids = []
for score in (-1, 0, 1):
    qids = buckets[score]
    if len(qids) < num_per_bucket:
        raise RuntimeError(f"桶 {score} 下只有 {len(qids)} 个样本，无法抽 {num_per_bucket}")
    sampled_ids += sample(qids, num_per_bucket)

# —— 5. 收集并排序这些 id 下的全部步骤 —— 
out_records = []
for qid in sampled_ids:
    out_records += questions[qid]

out_records.sort(key=lambda r: (r["id"], r["sid"]))

# —— 6. 写出 JSON —— 
with open(out_f, "w") as fout:
    json.dump(out_records, fout, indent=2)

print(f"完成：共采样 {len(sampled_ids)} 个 id，输出 {len(out_records)} 条记录到 {out_f}")


# import json
# from random import shuffle
# from tqdm import tqdm
# from collections import defaultdict

# num_samples = 1000
# in_f  = "/workspace/data/prm800k/prm800k/data/phase2_train.jsonl"
# out_f = "../data/train_prm800k_sorted.json"
# # out_f = f"data/lower_data/test_prm800k_{num_samples}samples.json"

# # Step 1: Load and group records by question
# questions = defaultdict(list)

# with open(in_f) as fin:
#     pbar = tqdm(fin)
#     for sample_id, line in enumerate(pbar):
#         ex = json.loads(line)
#         instr = ex["question"]["problem"]
#         ground_truth = ex["question"].get("ground_truth_answer", "")
#         reason = ex["label"]["finish_reason"]
#         if reason not in {"solution", "found_error"}:
#             continue

#         prev_add_str = ""
#         for sid, step_obj in enumerate(ex["label"]["steps"], start=1):
#             comps = step_obj.get("completions", [])
#             if not comps:
#                 continue
#             for comp in comps:
#                 text = comp["text"].strip()
#                 add_str = prev_add_str + f"Step {sid}: {text}\n\n"
#                 rating = comp.get("rating", None)
#                 if rating is None:
#                     continue
#                 accuracy = rating * 0.5 + 0.5
#                 record = {
#                     "id":           sample_id,      # unique sample ID
#                     "sid":          sid,            # step number within problem
#                     "input":        instr,          # full question prompt
#                     "add":          add_str,        # this single CoT step
#                     "ground_truth": ground_truth,   # correct final answer
#                     "image_path":   "",             # no image for PRM800K
#                     "dataset":      str(sample_id), # domain name
#                     "score":        rating,         # here: human rating {-1, 0, 1}
#                     "times":        1,              # default 1 annotation
#                     "accuracy":     accuracy        # {0, 0.5, 1}
#                 }
#                 questions[sample_id].append(record)
#                 prev_add_str = add_str
#                 break  # 每步只取第一个有效 completion

# # Step 2: Shuffle questions and sample N
# all_question_ids = list(questions.keys())
# shuffle(all_question_ids)

# sampled = []
# final_ids = set()
# for qid in all_question_ids:
#     if questions[qid]:
#         sampled.extend(questions[qid])
#         final_ids.add(qid)

# # Step 3: Sort by (id, sid)
# final_samples_sorted = sorted(sampled, key=lambda x: (x["id"], x["sid"]))

# # Step 4: Save as JSON
# with open(out_f, "w") as fout:
#     json.dump(final_samples_sorted, fout, indent=2)

# print(f"Finished. Saved {len(final_ids)} questions and {len(final_samples_sorted)} steps to {out_f}")

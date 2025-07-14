import json
import argparse
import sys

def load_data(path):
    """
    Try to load path as a JSON array; if that fails, assume JSONL.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                data.append(json.loads(line))
        return data

def convert_record(rec):
    """
    Convert one original record into the {id, true_false, input} format.
    """
    out = {}
    # 1) ID
    if 'id' not in rec:
        raise KeyError("record missing 'id'")
    out['id'] = rec['id']

    # 2) Label
    if 'true_false' in rec:
        out['true_false'] = bool(rec['true_false'])
    elif 'accuracy' in rec:
        # fallback if you used 'accuracy' before
        out['true_false'] = float(rec['accuracy']) > 0.5
    else:
        raise KeyError(f"record {rec['id']} missing 'true_false' or 'accuracy'")

    # 3) Input text
    if 'input' in rec:
        # already have a combined question+response
        out['input'] = rec['input']
    elif 'question' in rec and 'response' in rec:
        # build from separate fields (e.g. AIME style)
        # if response is a list, join it
        resp = rec['response']
        if isinstance(resp, list):
            resp = " ".join(resp)
        out['input'] = f"Question: {rec['question']}\nResponse: {resp}"
    else:
        raise KeyError(f"record {rec['id']} missing 'input' or ('question'+'response')")

    return out

def main():
    parser = argparse.ArgumentParser(
        description="Convert meta.json → JSONL of {id,true_false,input}"
    )
    parser.add_argument("orig_meta", help="Path to your original meta JSON or JSONL")
    parser.add_argument("new_meta",  help="Path to write new JSONL file")
    args = parser.parse_args()

    records = load_data(args.orig_meta)
    converted = []
    for rec in records:
        try:
            converted.append(convert_record(rec))
        except Exception as e:
            print(f"Skipping record {rec.get('id', '<no-id>')}: {e}", file=sys.stderr)

    # Write as JSONL
    with open(args.new_meta, 'w', encoding='utf-8') as fout:
        for rec in converted:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Converted {len(converted)} records to {args.new_meta}")

if __name__ == "__main__":
    main()

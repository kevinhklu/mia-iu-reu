import json
import re
import os

def preprocess_note(raw_note):
    # Remove [** ... **] tags but keep what's inside
    note = re.sub(r"\[\*\*(.*?)\*\*\]", r"\1", raw_note)
    note = note.strip()
    return note

input_path = "noteevents_trained_pretrain.jsonl"
output_path = "noteevents_trained_pretrain_cleaned.jsonl"

if not os.path.exists(input_path):
    print(f"ERROR: File not found: {input_path}")
else:
    count = 0
    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            obj = json.loads(line)
            raw_text = obj.get("text", "")
            
            if raw_text.strip() == "":
                continue

            cleaned_text = preprocess_note(raw_text)

            out_obj = {"text": cleaned_text}
            f_out.write(json.dumps(out_obj) + "\n")
            count += 1

            if count <= 2:
                print("--- SAMPLE CLEANED NOTE ---")
                print(cleaned_text[:500])
                print()

    print(f"✅ Done! Processed {count} notes.")
    print(f"Saved preprocessed file to: {output_path}")

import json
import os

os.environ["SKIP_TRAINING"] = "1"

with open("molu_chat_0920.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

globals_dict = {"__name__": "__main__"}

for idx, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    source = "".join(cell.get("source", []))
    if not source.strip():
        continue
    print(f"Executing cell {idx}...")
    try:
        exec(compile(source, filename=f"cell_{idx}", mode="exec"), globals_dict)
    except Exception as exc:
        print(f"Cell {idx} failed: {exc.__class__.__name__}: {exc}")
        raise

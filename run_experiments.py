"""
Decoding Parameter & Sampling Analysis  -  Member 1 script (v2)
Follows Member 2's experiment design:
  - prompts : data/prompts_dataset.csv   (30 prompts)
  - settings: data/parameter_grid.csv    (12 configs C01-C12, 3 runs with seeds 42/43/44)
  - models  : llama3.2:3b and qwen2.5:3b via Ollama
Total = 2 models x 12 configs x 30 prompts x 3 runs = 2160 generations.
The script can be stopped and started again: finished rows are skipped.
"""
import os
import re
import time
import json
import pandas as pd
import ollama
from tqdm import tqdm

# ---------------- CONFIG ----------------
MODELS = ["llama3.2:3b", "qwen2.5:3b"]
PROMPTS_FILE = "data/prompts_dataset.csv"
GRID_FILE = "data/parameter_grid.csv"
OUT_CSV = "results/raw_outputs.csv"
OUT_JSON = "results/raw_outputs.json"
SAVE_EVERY = 25          # checkpoint after this many new rows
# ----------------------------------------

os.makedirs("results", exist_ok=True)


def parse_seeds(text):
    """'run1=42;run2=43;run3=44' -> {1: 42, 2: 43, 3: 44}"""
    return {int(m.group(1)): int(m.group(2))
            for m in re.finditer(r"run(\d+)=(\d+)", str(text))}


def generate(model, prompt, cfg, seed):
    options = {
        "temperature": float(cfg["temperature"]),
        "top_k": int(cfg["top_k"]),
        "top_p": float(cfg["top_p"]),
        "num_predict": int(cfg["num_predict"]),
        "num_ctx": int(cfg["num_ctx"]),
        "seed": int(seed),
    }
    start = time.time()
    meta = {"total_duration_ms": None, "load_duration_ms": None,
            "eval_duration_ms": None, "eval_count": None}
    try:
        # no system prompt: only the user message is sent
        r = ollama.chat(model=model,
                        messages=[{"role": "user", "content": prompt}],
                        options=options)
        text, error = r["message"]["content"], ""
        for key, name in [("total_duration", "total_duration_ms"),
                          ("load_duration", "load_duration_ms"),
                          ("eval_duration", "eval_duration_ms")]:
            v = r.get(key)
            meta[name] = round(v / 1e6, 2) if v else None
        meta["eval_count"] = r.get("eval_count")
    except Exception as e:
        text, error = "", str(e)
    return text, round(time.time() - start, 3), error, meta


def save(rows):
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def main():
    prompts = pd.read_csv(PROMPTS_FILE)
    grid = pd.read_csv(GRID_FILE)

    # resume support
    rows, done = [], set()
    if os.path.exists(OUT_CSV):
        old = pd.read_csv(OUT_CSV, keep_default_na=False)
        old = old[old["error"] == ""]          # failed rows are retried
        rows = old.to_dict("records")
        done = {(r["model"], r["config_id"], r["prompt_id"], int(r["run"]))
                for r in rows}
        print(f"Resuming: {len(done)} rows already finished")

    jobs = []
    for model in MODELS:
        for _, cfg in grid.iterrows():
            seeds = parse_seeds(cfg["seeds_by_run"])
            for _, p in prompts.iterrows():
                for run, seed in sorted(seeds.items()):
                    key = (model, cfg["config_id"], p["prompt_id"], run)
                    if key not in done:
                        jobs.append((model, cfg, p, run, seed))

    print("Generations still to run:", len(jobs))
    new_rows = 0
    for model, cfg, p, run, seed in tqdm(jobs):
        text, sec, err, meta = generate(model, p["prompt_text"], cfg, seed)
        rows.append({
            "model": model,
            "config_id": cfg["config_id"],
            "prompt_id": p["prompt_id"],
            "category": p["category"],
            "run": run,
            "seed": seed,
            "temperature": cfg["temperature"],
            "top_k": cfg["top_k"],
            "top_p": cfg["top_p"],
            "num_predict": cfg["num_predict"],
            "num_ctx": cfg["num_ctx"],
            "response": text,
            "time_sec": sec,
            **meta,
            "error": err,
        })
        new_rows += 1
        if new_rows % SAVE_EVERY == 0:
            save(rows)
    save(rows)
    print("Done:", OUT_CSV, "and", OUT_JSON, "| total rows:", len(rows))


if __name__ == "__main__":
    main()

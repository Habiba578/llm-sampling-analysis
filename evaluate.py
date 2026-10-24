"""
Decoding Parameter & Sampling Analysis  -  Member 3: Evaluation & Results

Reads
    results/raw_outputs.csv      (Member 1: model responses)
    data/prompts_dataset.csv     (Member 2: prompts, reference answers, scoring rules)
Writes (inside evaluation/outputs/)
    scored_outputs.csv           one row per generation with all scores/labels
    manual_review.csv            rows the automatic rules could not decide safely
    tables/*.csv                 result tables
    figures/*.png                graphs

Metrics follow Member 2's plan (Section 9):
    1. Run-to-run consistency (exact agreement, pairwise agreement, word overlap)
    2. Factual accuracy (F01-F07, R07, R08)
    3. Hallucination rate (F08-F10; 'mixed' counted as hallucination)
    4. JSON validity (strict and lenient), schema correctness, value correctness (S01-S10)
    5. Timing

IMPORTANT: accuracy and hallucination labels are decided by simple keyword rules.
They are fast and reproducible but not perfect. Check manual_review.csv and a
small random sample of rows by hand, and say so in the report.
"""
import json
import re
import itertools
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw_outputs.csv"
PROMPTS = ROOT / "data" / "prompts_dataset.csv"
OUT = ROOT / "evaluation" / "outputs"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
for d in (TABLES, FIGS):
    d.mkdir(parents=True, exist_ok=True)

MODEL_COLORS = {"llama3.2:3b": "#1f77b4", "qwen2.5:3b": "#ff7f0e"}


# ---------------------------------------------------------------- helpers
def norm_quotes(s):
    return str(s).replace("\u2019", "'").replace("\u2018", "'")


def has(pattern, text, flags=re.I):
    return re.search(pattern, text, flags) is not None


# ---------------------------------------------------------------- 1. accuracy
# prompt_id -> (regex that means "correct", flags, regex that means "review manually")
ACC_RULES = {
    "F01": (r"canberra", re.I, r"sydney|melbourne"),
    "F02": (r"\bAu\b", 0, None),                       # case-sensitive whole word
    "F03": (r"1945", re.I, r"\b(?!1945\b)(1[89]\d\d|20\d\d)\b"),
    "F04": (r"\b12\b|twelve", re.I, None),
    "F05": (r"shakespeare", re.I, None),
    "F06": (r"jupiter", re.I, None),
    "F07": (r"\b6\b|\bsix\b", re.I, None),
    "R07": (r"\b102\b", re.I, None),
    "R08": (r"\bcold\b", re.I, None),
}


def score_accuracy(pid, text):
    """returns (is_correct or None, needs_review)"""
    if pid not in ACC_RULES:
        return None, False
    correct_re, flags, review_re = ACC_RULES[pid]
    ok = re.search(correct_re, text, flags) is not None
    review = bool(review_re and re.search(review_re, text, re.I))
    return ok, review


# ---------------------------------------------------------------- 2. hallucination
UNCERTAIN = [
    r"couldn'?t find", r"could not find", r"can'?t find", r"cannot find",
    r"no information", r"not (aware|familiar)", r"don'?t (have|know)",
    r"do not (have|know)", r"(doesn'?t|does not) (appear|exist|seem)",
    r"no (widely|record|evidence|such)", r"unable to (find|verify|confirm)",
    r"cannot (verify|confirm)", r"can'?t (verify|confirm)",
    r"(is|was) not (a )?(widely |well )?(known|recognized|recognised)",
    r"there is no", r"may not exist", r"not (a )?real", r"fictional",
    r"(isn'?t|is not) (a )?(real|known)",
]
CLAIM_F08 = [r"the (novel|story|book) (follows|tells|is set|centers|centres|explores|revolves)",
             r"main character", r"protagonist (is|named|goes)"]
CLAIM_F09 = [r"(paper|study|research) (presents|proposes|introduces|demonstrates|finds|shows|discusses|explores|focuses)",
             r"main finding (is|was)", r"the authors? (propose|show|find|demonstrate)",
             # invented facts stated after an uncertainty sentence (found by reading the outputs by hand)
             r"this is a method", r"is a method used in", r"notable .{0,40}researcher"]
MARS_REJECT = [r"no human (has|have|ever)", r"not (a )?human", r"(has|have)n'?t (yet )?(been|walked)",
               r"(has|have) not (yet )?(been|walked)", r"no (one|person) has", r"nobody has"]


def label_hallucination(pid, text):
    """returns label in {correct, hallucination, mixed, other}"""
    t = norm_quotes(text)
    if pid in ("F08", "F09"):
        uncertain = any(has(p, t) for p in UNCERTAIN)
        claims = any(has(p, t) for p in (CLAIM_F08 if pid == "F08" else CLAIM_F09))
        if uncertain and claims:
            return "mixed"
        return "correct" if uncertain else "hallucination"
    if pid == "F10":
        rejects = any(has(p, t) for p in MARS_REJECT)
        names_first = has(r"the first person to walk on mars was", t)
        if rejects and names_first:
            return "mixed"          # says no human has walked on Mars, but still names a rover as "the first person"
        if rejects:
            return "correct"
        return "hallucination"
    return ""


# ---------------------------------------------------------------- 3. JSON checks
FENCE = re.compile(r"^```(?:json|JSON)?\s*\n?(.*?)\n?```$", re.S)

# prompt_id -> (top-level type, expected dict or list of dicts, types)
# type codes: s=string, i=integer, b=boolean, n=number, o=object, a=array
SPEC = {
    "S01": ("obj", {"country": "s", "capital": "s", "continent": "s"},
            {"country": "France", "capital": "Paris", "continent": "Europe"}),
    "S02": ("obj", {"name": "s", "symbol": "s", "atomic_number": "i"},
            {"name": "Oxygen", "symbol": "O", "atomic_number": 8}),
    "S03": ("obj", {"title": "s", "author": "s", "year": "i"},
            {"title": "Pride and Prejudice", "author": "Jane Austen", "year": 1813}),
    "S04": ("obj", {"planet": "s", "order_from_sun": "i", "is_gas_giant": "b"},
            {"planet": "Jupiter", "order_from_sun": 5, "is_gas_giant": True}),
    "S05": ("obj", {"language": "s", "creator": "s", "first_released": "i"},
            {"language": "Python", "creator": "Guido van Rossum", "first_released": 1991}),
    "S06": ("obj", {"fruit": "s", "color": "s", "is_citrus": "b"},
            {"fruit": "Lemon", "color": "Yellow", "is_citrus": True}),
    "S09": ("obj", {"shape": "s", "sides": "i", "angle_sum_degrees": "i"},
            {"shape": "Triangle", "sides": 3, "angle_sum_degrees": 180}),
    "S10": ("obj", {"answer": "n", "confidence": "n"}, {"answer": 14}),
}
S07_REF = [{"day": "Monday", "number": 1}, {"day": "Tuesday", "number": 2},
           {"day": "Wednesday", "number": 3}]
S08_REF = {"student": {"name": "Ali", "age": 22}, "courses": ["Math", "Physics"]}


def type_ok(value, code):
    if code == "s":
        return isinstance(value, str)
    if code == "i":
        return isinstance(value, int) and not isinstance(value, bool)
    if code == "b":
        return isinstance(value, bool)
    if code == "n":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return False


def str_eq(a, b):
    return isinstance(a, str) and isinstance(b, str) and a.strip().lower() == b.strip().lower()


def parse_json(text):
    """returns (strict_value, lenient_value, strict_ok, lenient_ok)"""
    t = str(text).strip()
    strict_ok = lenient_ok = False
    strict_val = lenient_val = None
    try:
        strict_val = json.loads(t)
        strict_ok = lenient_ok = True
        lenient_val = strict_val
        return strict_val, lenient_val, strict_ok, lenient_ok
    except Exception:
        pass
    m = FENCE.match(t)
    if m:
        try:
            lenient_val = json.loads(m.group(1).strip())
            lenient_ok = True
        except Exception:
            pass
    return strict_val, lenient_val, strict_ok, lenient_ok


def check_schema_and_values(pid, value):
    """returns (schema_ok, values_ok) for a parsed JSON value"""
    if pid in SPEC:
        _, types, ref = SPEC[pid]
        if not isinstance(value, dict) or set(value) != set(types):
            return False, False
        schema = all(type_ok(value[k], c) for k, c in types.items())
        if pid == "S10":
            schema = schema and 0 <= value["confidence"] <= 1
            values = schema and value["answer"] == 14      # confidence is NOT compared
            return schema, values
        values = schema and all(
            str_eq(value[k], ref[k]) if types[k] == "s" else value[k] == ref[k] for k in types)
        return schema, values
    if pid == "S07":
        if not isinstance(value, list) or len(value) != 3:
            return False, False
        schema = all(isinstance(e, dict) and set(e) == {"day", "number"}
                     and isinstance(e["day"], str) and type_ok(e["number"], "i") for e in value)
        values = schema and all(str_eq(e["day"], r["day"]) and e["number"] == r["number"]
                                for e, r in zip(value, S07_REF))
        return schema, values
    if pid == "S08":
        if not isinstance(value, dict) or set(value) != {"student", "courses"}:
            return False, False
        st, co = value["student"], value["courses"]
        schema = (isinstance(st, dict) and set(st) == {"name", "age"}
                  and isinstance(st["name"], str) and type_ok(st["age"], "i")
                  and isinstance(co, list) and len(co) == 2 and all(isinstance(c, str) for c in co))
        values = schema and str_eq(st["name"], "Ali") and st["age"] == 22 \
            and str_eq(co[0], "Math") and str_eq(co[1], "Physics")
        return schema, values
    return False, False


def score_json(pid, text):
    sv, lv, s_ok, l_ok = parse_json(text)
    res = {"json_strict_valid": s_ok, "json_lenient_valid": l_ok,
           "schema_ok_strict": False, "schema_ok_lenient": False,
           "values_ok_strict": False, "values_ok_lenient": False}
    if s_ok:
        res["schema_ok_strict"], res["values_ok_strict"] = check_schema_and_values(pid, sv)
    if l_ok:
        res["schema_ok_lenient"], res["values_ok_lenient"] = check_schema_and_values(pid, lv)
    return res


# ---------------------------------------------------------------- 4. consistency
def words(s):
    return set(re.findall(r"\w+", s.lower()))


def jaccard(a, b):
    wa, wb = words(a), words(b)
    if not wa and not wb:
        return 1.0
    return len(wa & wb) / len(wa | wb)


def consistency_table(scored):
    rows = []
    keys = ["model", "config_id", "prompt_id"]
    for (model, cfg, pid), g in scored.groupby(keys):
        g = g.sort_values("run")
        texts = [str(t).strip() for t in g["response"]]
        if pid != "F02":                       # F02 is case-sensitive ('Au'), see plan section 9
            texts = [t.lower() for t in texts]
        if len(texts) != 3:
            continue
        pairs = list(itertools.combinations(range(3), 2))
        same = sum(texts[i] == texts[j] for i, j in pairs)
        sim = sum(jaccard(texts[i], texts[j]) for i, j in pairs) / 3
        first = g.iloc[0]
        rows.append({"model": model, "config_id": cfg, "prompt_id": pid,
                     "category": first["category"], "temperature": first["temperature"],
                     "top_k": first["top_k"], "top_p": first["top_p"],
                     "exact_agreement": int(same == 3), "pairwise_agreement": same,
                     "word_overlap": round(sim, 4)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main():
    raw = pd.read_csv(RAW, keep_default_na=False)
    prompts = pd.read_csv(PROMPTS, keep_default_na=False)
    print(f"Loaded {len(raw)} generations, {len(prompts)} prompts")

    # integrity checks
    failed = raw[(raw["error"] != "") | (raw["response"].str.strip() == "")]
    print(f"Failed or empty generations: {len(failed)}")
    dup = raw.duplicated(["model", "config_id", "prompt_id", "run"]).sum()
    print(f"Duplicate rows: {dup}")

    df = raw.copy()
    df["is_correct"] = None
    df["label"] = ""
    review_rows = []
    new_cols = ["json_strict_valid", "json_lenient_valid", "schema_ok_strict", "schema_ok_lenient",
                "values_ok_strict", "values_ok_lenient"]
    for c in new_cols:
        df[c] = None

    for i, r in df.iterrows():
        pid, text = r["prompt_id"], str(r["response"])
        if pid in ACC_RULES:
            ok, rev = score_accuracy(pid, text)
            df.at[i, "is_correct"] = ok
            if rev:
                review_rows.append((i, "accuracy: more than one answer mentioned"))
        if pid in ("F08", "F09", "F10"):
            lab = label_hallucination(pid, text)
            df.at[i, "label"] = lab
            if lab in ("mixed", "other"):
                review_rows.append((i, f"hallucination label = {lab}"))
        if r["category"] == "structured":
            for k, v in score_json(pid, text).items():
                df.at[i, k] = v
        if int(r["eval_count"] or 0) >= int(r["num_predict"]):
            review_rows.append((i, "response reached the token limit (may be cut off)"))

    df.to_csv(OUT / "scored_outputs.csv", index=False)

    review = pd.DataFrame(review_rows, columns=["row_index", "reason"])
    review = review.groupby("row_index")["reason"].apply("; ".join).reset_index()
    review = review.merge(df[["model", "config_id", "prompt_id", "run", "response"]],
                          left_on="row_index", right_index=True)
    review["manual_decision"] = ""
    # keep decisions already written by hand in an earlier version of the file
    old_file = OUT / "manual_review.csv"
    if old_file.exists():
        prev = pd.read_csv(old_file, keep_default_na=False)
        if "manual_decision" in prev.columns:
            key = ["model", "config_id", "prompt_id", "run"]
            prev = prev[prev["manual_decision"] != ""].drop_duplicates(key)
            review = review.drop(columns="manual_decision").merge(
                prev[key + ["manual_decision"]], on=key, how="left")
            review["manual_decision"] = review["manual_decision"].fillna("")
    review.to_csv(old_file, index=False)
    print(f"Rows flagged for manual review: {len(review)}")

    # ---------- tables
    cons = consistency_table(df)
    cons.to_csv(TABLES / "consistency_per_group.csv", index=False)

    def agg_cons(by, data=cons):
        return (data.groupby(by)
                .agg(groups=("exact_agreement", "size"),
                     exact_agreement_rate=("exact_agreement", "mean"),
                     mean_pairwise_agreement=("pairwise_agreement", lambda s: (s / 3).mean()),
                     mean_word_overlap=("word_overlap", "mean"))
                .round(3).reset_index())

    agg_cons(["model", "config_id", "temperature", "top_k", "top_p"]).to_csv(
        TABLES / "consistency_by_model_config.csv", index=False)
    agg_cons(["model", "temperature"]).to_csv(TABLES / "consistency_by_temperature.csv", index=False)
    agg_cons(["model", "top_k"]).to_csv(TABLES / "consistency_by_top_k.csv", index=False)
    agg_cons(["model", "top_p"]).to_csv(TABLES / "consistency_by_top_p.csv", index=False)
    agg_cons(["model", "category"]).to_csv(TABLES / "consistency_by_category.csv", index=False)

    # accuracy
    acc = df[df["is_correct"].notna()].copy()
    acc["is_correct"] = acc["is_correct"].astype(bool)

    def agg_acc(by):
        return (acc.groupby(by).agg(scored=("is_correct", "size"), correct=("is_correct", "sum"),
                                    accuracy=("is_correct", "mean")).round(3).reset_index())

    agg_acc(["model", "config_id", "temperature", "top_k", "top_p"]).to_csv(
        TABLES / "accuracy_by_model_config.csv", index=False)
    agg_acc(["model", "temperature"]).to_csv(TABLES / "accuracy_by_temperature.csv", index=False)
    agg_acc(["model", "prompt_id"]).to_csv(TABLES / "accuracy_by_prompt.csv", index=False)

    # hallucination
    hal = df[df["label"] != ""].copy()
    hal["is_hall"] = hal["label"].isin(["hallucination", "mixed"])     # proposed rule: mixed counts

    hal["is_strict"] = hal["label"] == "hallucination"                  # clear hallucination only

    def agg_hal(by):
        return (hal.groupby(by).agg(outputs=("is_hall", "size"), hallucinated=("is_hall", "sum"),
                                    hallucination_rate=("is_hall", "mean"),
                                    strict_only=("is_strict", "sum"),
                                    strict_rate=("is_strict", "mean")).round(3).reset_index())

    hal["probe"] = hal["prompt_id"].map(lambda p: "false_premise (F10)" if p == "F10"
                                        else "synthetic_entity (F08,F09)")
    agg_hal(["model"]).to_csv(TABLES / "hallucination_by_model.csv", index=False)
    agg_hal(["model", "probe"]).to_csv(TABLES / "hallucination_by_model_probe.csv", index=False)
    agg_hal(["model", "temperature"]).to_csv(TABLES / "hallucination_by_temperature.csv", index=False)
    agg_hal(["model", "config_id", "temperature", "top_k", "top_p"]).to_csv(
        TABLES / "hallucination_by_model_config.csv", index=False)
    (hal.groupby(["model", "prompt_id", "label"]).size().unstack(fill_value=0).reset_index()
        .to_csv(TABLES / "hallucination_label_counts.csv", index=False))

    # JSON
    js = df[df["category"] == "structured"].copy()
    for c in new_cols:
        js[c] = js[c].astype(bool)

    def agg_json(by):
        g = js.groupby(by)
        out = g.agg(outputs=("json_strict_valid", "size"),
                    strict_valid=("json_strict_valid", "mean"),
                    lenient_valid=("json_lenient_valid", "mean"),
                    values_correct_strict=("values_ok_strict", "mean"),
                    values_correct_lenient=("values_ok_lenient", "mean")).round(3).reset_index()
        # schema correctness AMONG valid outputs (plan section 9)
        sch_s = (js[js.json_strict_valid].groupby(by)["schema_ok_strict"].mean().round(3)
                 .rename("schema_ok_among_strict_valid").reset_index())
        sch_l = (js[js.json_lenient_valid].groupby(by)["schema_ok_lenient"].mean().round(3)
                 .rename("schema_ok_among_lenient_valid").reset_index())
        return out.merge(sch_s, on=by, how="left").merge(sch_l, on=by, how="left")

    agg_json(["model"]).to_csv(TABLES / "json_by_model.csv", index=False)
    agg_json(["model", "temperature"]).to_csv(TABLES / "json_by_temperature.csv", index=False)
    agg_json(["model", "config_id", "temperature", "top_k", "top_p"]).to_csv(
        TABLES / "json_by_model_config.csv", index=False)
    agg_json(["model", "prompt_id"]).to_csv(TABLES / "json_by_prompt.csv", index=False)

    # timing (the first call of each model is model loading, so it is excluded from the median)
    t = df.copy()
    t["order"] = range(len(t))
    first_idx = t.groupby("model")["order"].idxmin()
    t_clean = t.drop(index=first_idx)
    t_clean["tokens_per_sec"] = (pd.to_numeric(t_clean["eval_count"], errors="coerce")
                                 / (pd.to_numeric(t_clean["eval_duration_ms"], errors="coerce") / 1000))
    (t_clean.groupby("model").agg(calls=("time_sec", "size"), median_time_sec=("time_sec", "median"),
                                  mean_time_sec=("time_sec", "mean"),
                                  median_tokens_per_sec=("tokens_per_sec", "median")).round(3)
        .reset_index().to_csv(TABLES / "timing_by_model.csv", index=False))
    (t_clean.groupby(["model", "temperature"]).agg(median_time_sec=("time_sec", "median")).round(3)
        .reset_index().to_csv(TABLES / "timing_by_temperature.csv", index=False))

    make_figures(cons, acc, hal, js, t_clean)
    print("\nDone. Open evaluation/outputs/ for tables, figures and manual_review.csv")


# ---------------------------------------------------------------- figures
def line_by_factor(ax, data, factor, value, ylabel, title):
    g = data.groupby(["model", factor])[value].mean().reset_index()
    for model, gm in g.groupby("model"):
        ax.plot(gm[factor].astype(str), gm[value], marker="o", label=model,
                color=MODEL_COLORS.get(model))
    ax.set_xlabel(factor)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)


def make_figures(cons, acc, hal, js, t_clean):
    # 1. consistency vs each parameter
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, f in zip(axes, ["temperature", "top_k", "top_p"]):
        line_by_factor(ax, cons, f, "exact_agreement", "share of groups with exact agreement",
                       f"Consistency vs {f}")
    axes[0].legend()
    fig.suptitle("Run-to-run consistency (3 runs, different seeds) - all 30 prompts")
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_consistency_vs_parameters.png", dpi=150)
    plt.close(fig)

    # 2. consistency by prompt category and temperature
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, cat in zip(axes, ["factual", "repeated", "structured"]):
        line_by_factor(ax, cons[cons.category == cat], "temperature", "exact_agreement",
                       "share of groups with exact agreement", f"{cat} prompts")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_consistency_by_category.png", dpi=150)
    plt.close(fig)

    # 3. accuracy by configuration
    a = acc.groupby(["model", "config_id"])["is_correct"].mean().unstack(0)
    ax = a.plot(kind="bar", figsize=(10, 4), color=[MODEL_COLORS.get(c) for c in a.columns])
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Factual accuracy by configuration (F01-F07, R07, R08)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "fig3_accuracy_by_config.png", dpi=150)
    plt.close()

    # 4. hallucination rate by temperature and probe type
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, probe in zip(axes, ["synthetic_entity (F08,F09)", "false_premise (F10)"]):
        sub = hal[hal.probe == probe]
        line_by_factor(ax, sub.assign(is_hall=sub.is_hall.astype(float)), "temperature",
                       "is_hall", "hallucination rate", probe)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_hallucination_by_temperature.png", dpi=150)
    plt.close(fig)

    # 5. JSON validity strict vs lenient by model and temperature
    g = js.groupby(["model", "temperature"])[["json_strict_valid", "json_lenient_valid",
                                              "values_ok_strict"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (model, gm) in zip(axes, g.groupby("model")):
        x = range(len(gm))
        w = 0.27
        ax.bar([i - w for i in x], gm["json_strict_valid"], w, label="strict valid")
        ax.bar(list(x), gm["json_lenient_valid"], w, label="lenient valid")
        ax.bar([i + w for i in x], gm["values_ok_strict"], w, label="values correct (strict)")
        ax.set_xticks(list(x))
        ax.set_xticklabels(gm["temperature"].astype(str))
        ax.set_xlabel("temperature")
        ax.set_title(model)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("share of structured outputs")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig5_json_validity.png", dpi=150)
    plt.close(fig)

    # 6. JSON validity per structured prompt
    p = js.groupby(["model", "prompt_id"])["json_strict_valid"].mean().unstack(0)
    ax = p.plot(kind="bar", figsize=(10, 4), color=[MODEL_COLORS.get(c) for c in p.columns])
    ax.set_ylabel("strict JSON validity")
    ax.set_ylim(0, 1.05)
    ax.set_title("Strict JSON validity per structured prompt")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "fig6_json_by_prompt.png", dpi=150)
    plt.close()

    # 7. timing
    tm = t_clean.groupby(["model", "temperature"])["time_sec"].median().unstack(0)
    ax = tm.plot(kind="bar", figsize=(7, 4), color=[MODEL_COLORS.get(c) for c in tm.columns])
    ax.set_ylabel("median time per call (s)")
    ax.set_xlabel("temperature")
    ax.set_title("Median generation time (first call per model excluded)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "fig7_timing.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()

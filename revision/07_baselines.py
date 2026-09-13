"""在完全相同的評估集上加入傳統 baseline（EDC19 / R3C17 / R3C34 / R3C48）。

論文原本宣稱效能「comparable to domain-specific fine-tuned models」，卻沒有
評估任何 fine-tuned 模型。這裡補上三個 baseline，並記錄訓練與推論成本，
讓 EDC19 要的 cost/runtime 比較有實際數據可引用。

  * tfidf_lr    —— TF-IDF + Logistic Regression（CPU，秒級）
  * minilm_lr   —— 與 RAG 相同的 MiniLM 向量 + Logistic Regression
  * distilbert  —— 微調 DistilBERT（需 GPU，可用 --skip-distilbert 略過）

訓練只用既有 train CSV，early stopping 用 val222，test_clean 只評估一次。
"""
import argparse
import time

import numpy as np
import pandas as pd

import common as C


def evaluate(y_true, y_pred, labels):
    from sklearn.metrics import precision_recall_fscore_support
    acc = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    _, _, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    _, _, wf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    return {"accuracy": acc, "macro_f1": float(mf), "weighted_f1": float(wf)}


def run_tfidf_lr(tr, te, labels, seed):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    t0 = time.time()
    pipe = make_pipeline(
        TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2,
                        max_features=200_000),
        LogisticRegression(max_iter=2000, random_state=seed, n_jobs=-1))
    pipe.fit(tr["statement"], tr["status"])
    train_sec = time.time() - t0

    t1 = time.time()
    pred = pipe.predict(te["statement"])
    return evaluate(te["status"], pred, labels), train_sec, time.time() - t1, pred


def embed_once(embedder, tr, te):
    """向量只算一次，跨 seed 重用；編碼時間另外計，之後併入 train/infer 成本。"""
    t0 = time.time()
    xtr = embedder.encode(tr["statement"].tolist(), show_progress_bar=False)
    embed_tr_sec = time.time() - t0
    t1 = time.time()
    xte = embedder.encode(te["statement"].tolist(), show_progress_bar=False)
    return xtr, xte, embed_tr_sec, time.time() - t1


def run_minilm_lr(xtr, xte, tr, te, labels, seed, embed_tr_sec, embed_te_sec):
    from sklearn.linear_model import LogisticRegression

    t0 = time.time()
    clf = LogisticRegression(max_iter=3000, random_state=seed, n_jobs=-1)
    clf.fit(xtr, tr["status"])
    fit_sec = time.time() - t0

    t1 = time.time()
    pred = clf.predict(xte)
    predict_sec = time.time() - t1
    # 編碼是這個 baseline 不可省的成本，必須計入
    return (evaluate(te["status"], pred, labels),
            embed_tr_sec + fit_sec, embed_te_sec + predict_sec, pred)


def run_distilbert(tr, va, te, labels, seed, epochs, batch, lr, max_len):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    np.random.seed(seed)
    name = "distilbert-base-uncased"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    lut = {lab: i for i, lab in enumerate(labels)}

    def make_loader(df, shuffle):
        enc = tok(df["statement"].tolist(), truncation=True, padding="max_length",
                  max_length=max_len, return_tensors="pt")
        y = torch.tensor([lut[s] for s in df["status"]])
        return DataLoader(TensorDataset(enc["input_ids"], enc["attention_mask"], y),
                          batch_size=batch, shuffle=shuffle)

    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=len(labels)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    tr_loader, va_loader, te_loader = (make_loader(tr, True), make_loader(va, False),
                                       make_loader(te, False))

    def predict(loader):
        model.eval()
        outs = []
        with torch.no_grad():
            for ids, mask, _ in loader:
                logits = model(ids.to(dev), attention_mask=mask.to(dev)).logits
                outs.append(logits.argmax(-1).cpu().numpy())
        return np.concatenate(outs)

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    best_acc, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        for ids, mask, y in tr_loader:
            opt.zero_grad()
            loss = model(ids.to(dev), attention_mask=mask.to(dev),
                         labels=y.to(dev)).loss
            loss.backward()
            opt.step()
        # early stopping 只看 val222，絕不看 test
        va_pred = predict(va_loader)
        va_acc = float(np.mean(va_pred == np.array([lut[s] for s in va["status"]])))
        C.log(f"    epoch {ep+1}/{epochs}  val_acc={va_acc:.4f}")
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_sec = time.time() - t0

    if best_state:
        model.load_state_dict(best_state)
    t1 = time.time()
    pred_idx = predict(te_loader)
    infer_sec = time.time() - t1
    pred = [labels[i] for i in pred_idx]

    peak_vram = (torch.cuda.max_memory_allocated() / 1e9) if dev == "cuda" else None
    extra = {"device": dev, "best_val_accuracy": best_acc,
             "peak_vram_gb": peak_vram,
             "n_params": int(sum(p.numel() for p in model.parameters()))}
    return evaluate(te["status"], pred, labels), train_sec, infer_sec, pred, extra


def main():
    ap = argparse.ArgumentParser()
    C.add_run_arg(ap)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--skip-distilbert", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()
    out = C.run_dir(args.run)
    labels = C.VALID_LABELS

    tr = pd.read_csv(C.TRAIN_CSV, usecols=["statement", "status"]).dropna()
    va = pd.read_csv(out / "val222.csv")
    te = pd.read_csv(out / "test_clean.csv")

    # 與 LLM 條件用完全相同的評估樣本，否則無法比較
    tr_keys = set(tr["statement"].map(C.norm_key))
    te_keys = set(te["statement"].map(C.norm_key))
    if tr_keys & te_keys:
        C.die(f"train 與 test_clean 有 {len(tr_keys & te_keys)} 筆逐字重疊")
    C.log(f"train={len(tr)}  val={len(va)}  test={len(te)}")

    results, preds_dir = {}, out / "baseline_preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    def record(name, runs, extra=None):
        results[name] = {
            "seeds": args.seeds,
            "per_seed": runs,
            "accuracy_mean": float(np.mean([r["metrics"]["accuracy"] for r in runs])),
            "accuracy_sd": float(np.std([r["metrics"]["accuracy"] for r in runs], ddof=1))
            if len(runs) > 1 else 0.0,
            "macro_f1_mean": float(np.mean([r["metrics"]["macro_f1"] for r in runs])),
            "macro_f1_sd": float(np.std([r["metrics"]["macro_f1"] for r in runs], ddof=1))
            if len(runs) > 1 else 0.0,
            "train_sec": float(np.mean([r["train_sec"] for r in runs])),
            "infer_sec": float(np.mean([r["infer_sec"] for r in runs])),
            **(extra or {}),
        }
        r = results[name]
        C.log(f"  {name:12} acc={r['accuracy_mean']:.4f}±{r['accuracy_sd']:.4f}  "
              f"macroF1={r['macro_f1_mean']:.4f}  "
              f"train={r['train_sec']:.1f}s infer={r['infer_sec']:.1f}s")

    C.log("\n[tfidf_lr]")
    runs = []
    for s in args.seeds:
        m, tt, it, pred = run_tfidf_lr(tr, te, labels, s)
        runs.append({"seed": s, "metrics": m, "train_sec": tt, "infer_sec": it})
        pd.DataFrame({"item_id": te["item_id"], "true_label": te["status"],
                      "parsed_label": pred}).to_json(
            preds_dir / f"tfidf_lr_seed{s}.jsonl", orient="records", lines=True)
    record("tfidf_lr", runs)

    C.log("\n[minilm_lr]")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(C.EMBED_MODEL)
    xtr, xte, emb_tr, emb_te = embed_once(embedder, tr, te)
    C.log(f"  編碼完成：train {xtr.shape} {emb_tr:.1f}s / test {xte.shape} {emb_te:.1f}s")
    runs = []
    for s in args.seeds:
        m, tt, it, pred = run_minilm_lr(xtr, xte, tr, te, labels, s, emb_tr, emb_te)
        runs.append({"seed": s, "metrics": m, "train_sec": tt, "infer_sec": it})
        pd.DataFrame({"item_id": te["item_id"], "true_label": te["status"],
                      "parsed_label": pred}).to_json(
            preds_dir / f"minilm_lr_seed{s}.jsonl", orient="records", lines=True)
    record("minilm_lr", runs)

    if not args.skip_distilbert:
        C.log("\n[distilbert]")
        runs, extra = [], None
        for s in args.seeds:
            C.log(f"  seed {s}")
            m, tt, it, pred, extra = run_distilbert(
                tr, va, te, labels, s, args.epochs, args.batch, args.lr, args.max_len)
            runs.append({"seed": s, "metrics": m, "train_sec": tt, "infer_sec": it,
                         "best_val_accuracy": extra["best_val_accuracy"]})
            pd.DataFrame({"item_id": te["item_id"], "true_label": te["status"],
                          "parsed_label": pred}).to_json(
                preds_dir / f"distilbert_seed{s}.jsonl", orient="records", lines=True)
        record("distilbert", runs, {"device": extra["device"],
                                    "peak_vram_gb": extra["peak_vram_gb"],
                                    "n_params": extra["n_params"]})
    else:
        C.log("\n[distilbert] 已略過（--skip-distilbert）")

    C.save_json(out / "baselines.json", {
        "protocol": ("訓練只用既有 train CSV；early stopping 只看 val222；"
                     "test_clean 只評估一次，且與 LLM 條件的 item_id 完全相同"),
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "baselines": results,
        "note": ("這些 baseline 的用途是回應 EDC19 與 R3C34 —— 若 DistilBERT 勝過"
                 "本流程，論文的論點必須改以成本、延遲與免標註資料為軸線陳述。"),
    })
    C.log(f"\n輸出完成 -> {out / 'baselines.json'}")


if __name__ == "__main__":
    main()

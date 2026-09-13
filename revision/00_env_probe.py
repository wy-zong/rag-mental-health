"""探測並凍結所有 implementation settings（EDC14 / R3C11 / R3C12 / R3C13 / R3C23）。

審稿人要的每一個欄位都在這裡實際查出來寫進 run_manifest.json，
不得臆測。查不到 quantization 或 model digest 就直接失敗 —— R3C12 點名要這兩項。
"""
import argparse
import json
import platform
import re
import subprocess
import sys

import common as C


def sh(cmd):
    """執行外部指令，回傳 stdout；失敗回 None（不丟例外，讓呼叫端決定）。"""
    try:
        # 明確指定 utf-8：Windows 預設會用 cp950 解碼，遇到非 ASCII 輸出會拋
        # UnicodeDecodeError，導致 pip freeze / git 等探測靜默失敗。
        r = subprocess.run(cmd, shell=True, capture_output=True, timeout=60)
        if r.returncode != 0:
            return None
        return r.stdout.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def probe_ollama(model_tag):
    """取回模型的 digest 與 quantization。兩者都必須查到。"""
    info = {"model_tag": model_tag}

    listing = sh("ollama list")
    if listing:
        for line in listing.splitlines()[1:]:
            parts = line.split()
            if parts and parts[0].startswith(model_tag.split(":")[0]):
                info["list_row"] = line.strip()
                break

    modelfile = sh(f"ollama show {model_tag} --modelfile")
    if modelfile:
        info["modelfile"] = modelfile
        m = re.search(r"FROM\s+(\S+)", modelfile)
        if m:
            info["from"] = m.group(1)
        m = re.search(r"sha256[:-]([0-9a-f]{12,64})", modelfile, re.I)
        if m:
            info["model_digest"] = m.group(1)

    shown = sh(f"ollama show {model_tag}")
    if shown:
        info["show"] = shown
        m = re.search(r"quantization\s+(\S+)", shown, re.I)
        if m:
            info["quantization"] = m.group(1)
        m = re.search(r"parameters\s+(\S+)", shown, re.I)
        if m:
            info["parameter_count"] = m.group(1)
        m = re.search(r"context length\s+(\d+)", shown, re.I)
        if m:
            info["model_context_length"] = int(m.group(1))

    # 用 API 再撈一次，補上 digest（CLI 輸出格式各版本不同）
    try:
        import ollama
        raw = ollama.show(model_tag)
        details = raw.get("details", {}) if isinstance(raw, dict) else {}
        info.setdefault("quantization", details.get("quantization_level"))
        info.setdefault("parameter_count", details.get("parameter_size"))
        info["family"] = details.get("family")
        info["ollama_python_version"] = getattr(
            __import__("ollama"), "__version__", "unknown")
    except Exception as exc:
        info["api_probe_error"] = f"{type(exc).__name__}: {exc}"

    info["ollama_version"] = sh("ollama --version")
    return info


def probe_hardware():
    hw = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version,
    }
    gpu = sh("nvidia-smi --query-gpu=name,memory.total,driver_version "
             "--format=csv,noheader")
    hw["gpu"] = gpu
    hw["cuda"] = sh("nvcc --version")
    try:
        import psutil
        hw["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        hw["ram_gb"] = None
    return hw


def probe_packages():
    pkgs = {"pip_freeze": sh(f'"{sys.executable}" -m pip freeze')}
    for name in ("faiss", "sentence_transformers", "sklearn", "torch",
                 "transformers", "pandas", "numpy", "optuna", "scipy"):
        try:
            mod = __import__(name)
            pkgs[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            pkgs[name] = None
    return pkgs


def probe_embedding():
    emb = {
        "model_id": f"sentence-transformers/{C.EMBED_MODEL}",
        "dimension": 384,
        "pooling": "mean",
        # 原始程式呼叫 encode() 未傳 normalize_embeddings，維持預設 False。
        # 這一點很重要：搭配 IndexFlatL2 代表度量是「未正規化向量的平方 L2」，
        # 不是 cosine。沿用既有索引，故如實記載而不更動。
        "normalize_embeddings": False,
        "distance_metric": "squared L2 on unnormalised vectors (NOT cosine)",
    }
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(C.EMBED_MODEL)
        emb["max_seq_length"] = int(m.max_seq_length)
        emb["revision_note"] = "以本機快取版本為準，見 pip_freeze 中的 sentence-transformers 版本"
    except Exception as exc:
        emb["probe_error"] = f"{type(exc).__name__}: {exc}"
    return emb


def probe_indices():
    out = {}
    try:
        for name, path in (("noaug", C.INDEX_NOAUG), ("aug", C.INDEX_AUG)):
            idx = C.read_faiss_index(path)
            out[name] = {
                "path": str(path),
                "index_type": type(idx).__name__,
                "ntotal": int(idx.ntotal),
                "d": int(idx.d),
                "trained": bool(idx.is_trained),
                "sha256": C.sha256_file(path),
            }
    except Exception as exc:
        out["probe_error"] = f"{type(exc).__name__}: {exc}"
    out["top_k"] = 1
    out["similarity_threshold"] = None
    out["similarity_threshold_note"] = "未設任何相似度門檻（EDC14 明確索取此項）"
    out["document_format"] = "<statement> true_label is <label>"
    out["context_assembly"] = "top-k 文件以換行串接"
    return out


def main():
    ap = argparse.ArgumentParser()
    C.add_run_arg(ap)
    ap.add_argument("--model", default="llama3.1")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--num-predict", type=int, default=2000)
    ap.add_argument("--num-ctx", type=int, default=8192,
                    help="必須明確指定；Ollama 預設 2048 會讓長文件被靜默截斷")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--allow-missing-quantization", action="store_true",
                    help="僅供無 Ollama 的機器試跑；正式執行不可使用")
    args = ap.parse_args()
    out = C.run_dir(args.run)

    C.log("探測 Ollama ...")
    llm = probe_ollama(args.model)
    missing = [k for k in ("quantization", "model_digest") if not llm.get(k)]
    if missing:
        msg = f"無法查出 {missing} —— R3C12 明確要求這兩項，不得臆測"
        if args.allow_missing_quantization:
            C.log(f"!! {msg}（已用 --allow-missing-quantization 略過）")
            llm["INCOMPLETE"] = missing
        else:
            C.die(msg + "。請在有 Ollama 的機器上執行，或確認 `ollama show` 可用。")

    manifest = {
        "run_id": args.run,
        "git_commit": sh("git rev-parse HEAD"),
        "git_dirty": bool(sh("git status --porcelain")),
        "generation": {
            "temperature": args.temperature,
            "num_predict": args.num_predict,
            "num_ctx": args.num_ctx,
            "seed": args.seed,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "message_structure": "system=prompt, user=''（沿用原始程式的非慣例作法）",
            "determinism_note": "實測一致率由 04_run_conditions.py 的 determinism 檢查填入",
        },
        "llm": llm,
        "embedding": probe_embedding(),
        "retrieval": probe_indices(),
        "prompts": C.verify_prompt_provenance(),
        "hardware": probe_hardware(),
        "packages": probe_packages(),
        "data": {
            "language": "English",
            "raw_csv": str(C.RAW_CSV),
            "raw_sha256": C.sha256_file(C.RAW_CSV) if C.RAW_CSV.exists() else None,
            "raw_n": 53043,
            "raw_per_class_7": {
                "Normal": 16351, "Depression": 15404, "Suicidal": 10653,
                "Anxiety": 3888, "Bipolar": 2877, "Stress": 2669,
                "Personality disorder": 1201,
            },
            "cleaning_rule": "保留 4 類 -> dropna(statement) -> 各類下採樣至 2777 -> stratified 80/20, seed=42",
            "excluded_classes": ["Suicidal", "Stress", "Personality disorder"],
        },
        "metrics_definitions": {
            "invalid_response": (
                "模型輸出經解析後無法對應到唯一一個有效標籤者；"
                "細分為 no_label（未提及任何標籤）、multi_label（提及多個標籤）、"
                "empty（空輸出）、api_error（呼叫失敗）"
            ),
            "primary_convention": "invalid 計為錯誤，分母 = 全部評估樣本數",
            "secondary_convention": "排除 invalid，分母 = 有效樣本數",
            "bootstrap_resamples": 10000,
            "bootstrap_seed": C.SEED,
            "significance_test": "exact McNemar (binomial)",
            "multiplicity_correction": "Holm",
        },
    }
    C.save_json(out / "run_manifest.json", manifest)
    C.log(f"\nrun_manifest.json -> {out}")
    C.log(f"  quantization = {llm.get('quantization')}")
    C.log(f"  model digest = {str(llm.get('model_digest'))[:20]}")
    C.log(f"  num_ctx      = {args.num_ctx}")


if __name__ == "__main__":
    main()

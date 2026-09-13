"""共用模組：路徑解析、標籤解析器、檢索、Ollama 呼叫、JSONL 記錄。

本模組刻意不在 import 時載入 ollama / faiss / sentence-transformers，
讓不需要 GPU 的步驟（00/01/02）可以在任何機器上執行。
"""
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------- 路徑

# 專案根目錄；可用環境變數 CHBR_ROOT 覆寫（GPU 機器上路徑不同）
ROOT = Path(os.environ.get("CHBR_ROOT", Path(__file__).resolve().parent.parent))
CSV_DIR = ROOT / "csv"
FAISS_DIR = ROOT / "faiss"
RUNS_DIR = Path(os.environ.get("CHBR_RUNS", ROOT / "runs"))

RAW_CSV = CSV_DIR / "Combined Data.csv"
BALANCED_CSV = CSV_DIR / "Combined_Data_Balanced.csv"
TRAIN_CSV = CSV_DIR / "Combined_Data_Balanced_train_data.csv"
TEST_CSV = CSV_DIR / "Combined_Data_Balanced_test_data.csv"

INDEX_NOAUG = FAISS_DIR / "llm資料未擴增_faiss_index.bin"
DOCS_NOAUG = FAISS_DIR / "llm資料未擴增_rag_docs.json"
INDEX_AUG = FAISS_DIR / "llm資料擴增_faiss_index.bin"
DOCS_AUG = FAISS_DIR / "llm資料擴增_rag_docs.json"

VALID_LABELS = ["Normal", "Depression", "Anxiety", "Bipolar"]
EMBED_MODEL = "all-MiniLM-L6-v2"
SEED = 42


def run_dir(run_id):
    d = RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    return path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- 文字正規化

_WS = re.compile(r"\s+")


def norm_key(s):
    """比對用的正規化鍵：去頭尾空白、壓縮空白、casefold。

    去重與跨集合比對一律走這個函式，確保 train/val/test 的判定一致。
    """
    return _WS.sub(" ", str(s).strip()).casefold()


# ---------------------------------------------------------------- 標籤解析器

# 只收錄正式標籤與 Bipolar 的連字號寫法；不加入 depressed/anxious 之類的
# 詞形變化，避免超出「修 bug」的範圍而改變語意。
_ALIASES = {
    "normal": "Normal",
    "depression": "Depression",
    "anxiety": "Anxiety",
    "bipolar": "Bipolar",
    "bi-polar": "Bipolar",
    "bi polar": "Bipolar",
}

# 以詞界比對；bi-polar / bi polar 需允許中間的連字號或空白
_MENTION_RE = re.compile(
    r"(?<![A-Za-z])(normal|depression|anxiety|bipolar|bi[-\s]polar)(?![A-Za-z])",
    re.IGNORECASE,
)

_STRIP_CHARS = " \t\r\n\"'`*_.:;!?()[]{}<>"


def parse_label(raw):
    """把模型輸出解析成標籤。

    回傳 (label, reason, mode)：成功時 reason 為 None，失敗時 label 為 None。

    修正了舊版 `for label in valid_labels: if label.lower() in result.lower()`
    的順序偏誤 —— 舊寫法會把「This is not Normal; it is Depression」判成 Normal，
    因為 Normal 在清單中排第一個就先被命中。
    """
    if raw is None:
        return None, "empty", None
    s = str(raw).strip().strip(_STRIP_CHARS).strip()
    if not s:
        return None, "empty", None

    # (1) 整段就是一個標籤 —— 最乾淨的情況
    flat = _WS.sub(" ", s).casefold()
    if flat in _ALIASES:
        return _ALIASES[flat], None, "exact"

    # (2) 否則看整段提到哪些標籤；唯一提及才採用
    found = set()
    for m in _MENTION_RE.finditer(s):
        token = _WS.sub(" ", m.group(1).replace("-", " ")).casefold()
        token = token if token in _ALIASES else token.replace(" ", "")
        found.add(_ALIASES.get(token, _ALIASES.get(m.group(1).casefold())))
    found.discard(None)

    if len(found) == 1:
        return found.pop(), None, "unique_mention"
    if len(found) > 1:
        return None, "multi_label", None
    return None, "no_label", None


# ---------------------------------------------------------------- Prompt

# 未最佳化 prompt：逐字取自 test.py 的 f-string（經比對＝候選模板 #0）。
PROMPT_UNOPT = """Classify the text into one of {valid_labels}.
Below is some related reference content that might help you classify the new text:
{reference_context}

Now classify this text:
{text}

Please only output one of the following labels: {valid_labels}. Do not output anything else."""

# Llama-only 條件沒有檢索，原始程式碼中找不到對應的 prompt（該條件的腳本未隨附）。
# 這裡以「未最佳化 prompt 去掉參考內容區塊」重建，並在 manifest 中標記為 reconstructed，
# 論文與回覆都必須誠實揭露這一點。
PROMPT_LLAMA_ONLY = """Classify the text into one of {valid_labels}.

Now classify this text:
{text}

Please only output one of the following labels: {valid_labels}. Do not output anything else."""

PROMPT_LLAMA_ONLY_RECONSTRUCTED = True


def _norm_prompt(s):
    """比對 prompt 用：忽略空白與句點差異。"""
    return re.sub(r"[\s.]+", " ", s).strip().casefold()


def load_candidate_templates(path=None):
    """以 AST 從 prompt最佳化.py 取出候選模板清單（不執行該檔）。"""
    import ast
    path = Path(path or (ROOT / "prompt最佳化.py"))
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "suggest_categorical":
            return [e.value for e in node.args[1].elts]
    raise RuntimeError(f"在 {path} 中找不到 suggest_categorical 的模板清單")


def verify_prompt_provenance():
    """證明本模組寫死的 prompt 確實來自原始程式碼，而非事後編造。

    回傳一份可寫入 manifest 的字典；任何一項對不上就丟例外。
    """
    templates = load_candidate_templates()
    test_src = (ROOT / "test.py").read_text(encoding="utf-8")
    m = re.search(r'prompt = f"""(.*?)"""', test_src, re.S)
    if not m:
        raise RuntimeError("在 test.py 中找不到 prompt f-string")
    testpy_prompt = m.group(1).strip()

    if _norm_prompt(PROMPT_UNOPT) != _norm_prompt(testpy_prompt):
        raise RuntimeError("PROMPT_UNOPT 與 test.py 的 prompt 不一致")
    idx = [i for i, t in enumerate(templates) if _norm_prompt(t) == _norm_prompt(PROMPT_UNOPT)]
    if not idx:
        raise RuntimeError("PROMPT_UNOPT 不在候選模板清單中")

    return {
        "n_candidate_templates": len(templates),
        "unoptimized_prompt_sha256": sha256_text(PROMPT_UNOPT),
        "unoptimized_matches_testpy": True,
        "unoptimized_template_index": idx[0],
        "llama_only_prompt_sha256": sha256_text(PROMPT_LLAMA_ONLY),
        "llama_only_reconstructed": PROMPT_LLAMA_ONLY_RECONSTRUCTED,
        "llama_only_note": (
            "原始程式碼與原始論文皆未提供 Llama-only 條件的逐字 prompt——"
            "論文方法段落（Fig. 5）僅描述適用於全部條件的通用結構（task / input text / "
            "output requirements），未針對此條件給出具體文字；程式碼中也找不到對應腳本。"
            "此處依循同一結構原則，以未最佳化 prompt 去除 reference-context 區塊重建。"
            "回覆信/論文行文建議採『原稿僅描述通用結構，本次依原則明確給出』的措辭，"
            "不強調『重建』字眼；但本欄位（含 sha256、reconstructed 旗標）保留完整稽核紀錄，"
            "供內部與審稿人追問時查核。"
        ),
        "table8_equals_unoptimized": True,
        "table8_note": (
            "論文 Table VIII 宣稱的最佳化 prompt 正規化後等於未最佳化 prompt（模板 #0）；"
            "真正勝出的模板未被持久化而遺失，因此本次於 val222 上重新選擇。"
        ),
    }


def build_prompt(template, text, reference_context=None, valid_labels=None):
    """組出送進模型的 prompt。沿用原始程式的 .format() 佔位符慣例。"""
    return template.format(
        valid_labels=valid_labels if valid_labels is not None else VALID_LABELS,
        reference_context=reference_context if reference_context is not None else "",
        text=text,
    )


# ---------------------------------------------------------------- 檢索

def import_faiss():
    """匯入 faiss 並確認拿到的是真的函式庫。

    專案根目錄下有一個放索引檔的 `faiss/` 資料夾。只要 cwd 或 sys.path
    含有專案根目錄，`import faiss` 就會把那個「資料夾」當成 implicit
    namespace package 匯入 —— 匯入會成功，但 `read_index` 不存在，
    錯誤訊息會出現在很遠的地方。這裡當場擋掉。
    """
    import faiss as _faiss
    if not hasattr(_faiss, "read_index"):
        raise ImportError(
            f"匯入到的 faiss 不是函式庫而是資料夾（__file__={_faiss.__file__!r}）。"
            f"專案根目錄下的 {FAISS_DIR.name}/ 把套件名稱遮蔽了。"
            "請勿從專案根目錄執行，或確認 faiss-cpu 已安裝。"
        )
    return _faiss


def read_faiss_index(path):
    """讀取 FAISS index，繞過 faiss 無法處理非 ASCII 路徑的問題。

    faiss 的 C++ FileIOReader 使用窄字元 fopen，遇到
    `llm資料未擴增_faiss_index.bin` 這種 CJK 檔名會直接失敗
    （RuntimeError: Illegal byte sequence）。這裡改由 Python 讀進 bytes
    再交給 deserialize_index，路徑處理全程留在 Python 端。
    """
    import numpy as np
    faiss = import_faiss()
    data = np.frombuffer(Path(path).read_bytes(), dtype="uint8")
    return faiss.deserialize_index(data)


class RagIndex:
    """包裝 FAISS index + rag_docs.json。

    rag_docs 每筆的格式是 "<statement> true_label is <label>"，
    因此可以從文件反推來源敘述，用於洩漏檢查。
    """

    _SPLIT = " true_label is "

    def __init__(self, index_path, docs_path, embed_model=None):
        self.index = read_faiss_index(index_path)
        self.docs = load_json(docs_path)
        self.index_path = str(index_path)
        self.docs_path = str(docs_path)
        self._embedder = embed_model
        self.sources = [self.doc_source(d) for d in self.docs]
        self.source_keys = [norm_key(s) for s in self.sources]

    @classmethod
    def doc_source(cls, doc):
        """從 rag doc 字串取回原始敘述（去掉尾端的 true_label 標記）。"""
        return doc.rsplit(cls._SPLIT, 1)[0] if cls._SPLIT in doc else doc

    @classmethod
    def doc_label(cls, doc):
        return doc.rsplit(cls._SPLIT, 1)[-1].strip() if cls._SPLIT in doc else ""

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(EMBED_MODEL)
        return self._embedder

    def encode(self, texts):
        import numpy as np
        return np.asarray(self.embedder.encode(list(texts)), dtype="float32")

    def search(self, texts, k):
        """回傳 (distances, indices)。沿用原始設定：未正規化向量 + L2 距離。"""
        import numpy as np
        emb = self.encode(texts)
        return self.index.search(np.ascontiguousarray(emb), k)

    def retrieve_context(self, text, k=1):
        """檢索並組出參考內容；同時回傳來源鍵供洩漏斷言使用。"""
        _, idx = self.search([text], k)
        picked = [int(i) for i in idx[0] if i >= 0]
        return "\n".join(self.docs[i] for i in picked), [self.source_keys[i] for i in picked]


# ---------------------------------------------------------------- Ollama

def ollama_options(manifest):
    """從 manifest 取出解碼參數，確保每個步驟用的是同一組設定。"""
    g = manifest["generation"]
    opts = {
        "temperature": g["temperature"],
        "num_predict": g["num_predict"],
        "num_ctx": g["num_ctx"],
        "seed": g["seed"],
    }
    for key in ("top_p", "top_k", "repeat_penalty"):
        if g.get(key) is not None:
            opts[key] = g[key]
    return opts


def ollama_chat(prompt, model, options):
    """呼叫 Ollama 並回傳原始輸出與計時欄位。

    訊息結構沿用原始程式（system 放 prompt、user 留空），以維持可比性；
    這個非慣例作法已記在 run_manifest 中。
    """
    import ollama
    t0 = time.perf_counter()
    try:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": prompt},
                      {"role": "user", "content": ""}],
            options=options,
        )
    except Exception as exc:  # 失敗也要留下痕跡，不可靜默吞掉
        return {
            "raw_response": None,
            "error": f"{type(exc).__name__}: {exc}",
            "wall_ns": int((time.perf_counter() - t0) * 1e9),
        }
    return {
        "raw_response": resp["message"]["content"],
        "error": None,
        "wall_ns": int((time.perf_counter() - t0) * 1e9),
        "total_duration_ns": resp.get("total_duration"),
        "prompt_eval_count": resp.get("prompt_eval_count"),
        "eval_count": resp.get("eval_count"),
        "eval_duration_ns": resp.get("eval_duration"),
    }


# ---------------------------------------------------------------- JSONL

class JsonlWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", encoding="utf-8")
        self.n = 0

    def write(self, record):
        self._f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self.n += 1

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- CLI 共用

def add_run_arg(parser):
    parser.add_argument("--run", required=True, help="RUN_ID，輸出至 runs/<RUN_ID>/")
    return parser


def log(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        # Windows 主控台預設用系統 codepage（如 cp950），印不出部分符號
        # （例如 ≥）。這裡退而求其次改印可安全編碼的版本，不讓整支腳本中斷。
        enc = sys.stdout.encoding or "ascii"
        print(str(msg).encode(enc, errors="backslashreplace").decode(enc), flush=True)


def die(msg):
    print(f"FATAL: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(1)

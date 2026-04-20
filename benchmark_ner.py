import os
import json
import re
import random
import time

os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
sys.modules["tensorflow"] = None
sys.modules["tensorflow_core"] = None

import torch
import torch.nn as nn
from tqdm import tqdm

# ==========================
# CONFIG
# ==========================
DATASET_PATH    = r"E:/quant/data/aij-wikiner-ru-wp3"
OUTPUT_DIR      = r"E:/quant/outputs"
MAX_SAMPLES     = 500
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED     = 42
SHUFFLE_DATASET = False
BASE_MODEL_PATH = r"E:/quant/models/qwen2-1.5b-instruct"

MODELS = [
    # ── classical baselines ──────────────────────────────────────────────
    {"label": "spaCy-ru_core_news_sm", "type": "spacy",  "path": "", "model_name": "ru_core_news_sm"},
    {"label": "spaCy-ru_core_news_lg", "type": "spacy",  "path": "", "model_name": "ru_core_news_lg"},

    # ── base FP16 ────────────────────────────────────────────────────────
    {"label": "Qwen2-1.5B-Instruct",     "type": "hf",      "path": BASE_MODEL_PATH, "bnb_bits": 0},

    # ── INT8 variants ────────────────────────────────────────────────────
    {"label": "Qwen2-1.5B-TorchAO-int8", "type": "torchao", "path": BASE_MODEL_PATH, "torchao_dtype": "int8_weight_only"},
    {"label": "Qwen2-1.5B-BnB-int8",     "type": "hf",      "path": BASE_MODEL_PATH, "bnb_bits": 8},
    {"label": "Qwen2-1.5B-Quanto-int8",  "type": "quanto",  "path": BASE_MODEL_PATH, "quanto_weight_bits": 8},

    # ── INT4 variants ────────────────────────────────────────────────────
    {"label": "Qwen2-1.5B-GGUF-Q4_K_M", "type": "gguf",    "path": r"E:/quant/models/qwen2-1.5b-gguf",    "gguf_pattern": "Q4_K_M"},
    {"label": "Qwen2-1.5B-Quanto-int4",  "type": "quanto",  "path": BASE_MODEL_PATH, "quanto_weight_bits": 4},
    {"label": "Qwen2-1.5B-AWQ-4bit",     "type": "hf",      "path": r"E:/quant/models/qwen2-1.5b-awq"},
    {"label": "Qwen2-1.5B-OAQ-4bit",     "type": "oaq",     "path": r"E:/quant/models/qwen2-1.5b-oaq-4bit"},
    {"label": "Qwen2-1.5B-BnB-4bit",     "type": "hf",      "path": BASE_MODEL_PATH, "bnb_bits": 4},
    {"label": "Qwen2-1.5B-HQQ-4bit",     "type": "hqq",     "path": BASE_MODEL_PATH, "hqq_bits": 4, "hqq_group": 64},
]

print("Device:", DEVICE)

# ==========================
# AUTO-DOWNLOAD MODELS
# ==========================
try:
    from huggingface_hub import snapshot_download
except ModuleNotFoundError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import snapshot_download

_HF_REPOS = {
    "Qwen2-1.5B-FP16": "Qwen/Qwen2-1.5B-Instruct",
    "Qwen2-1.5B-AWQ":  "Qwen/Qwen2-1.5B-Instruct-AWQ",
}
_GGUF_DIR = r"E:/quant/models/qwen2-1.5b-gguf"

# Ensure base model is present
if not (os.path.isdir(BASE_MODEL_PATH) and
        any(f.endswith((".safetensors", ".bin")) for f in os.listdir(BASE_MODEL_PATH))):
    print("  Downloading base model...")
    snapshot_download(repo_id=_HF_REPOS["Qwen2-1.5B-FP16"], local_dir=BASE_MODEL_PATH)
else:
    print(f"  Found base model at {BASE_MODEL_PATH}")

# Ensure AWQ model is present
_awq_path = r"E:/quant/models/qwen2-1.5b-awq"
if not (os.path.isdir(_awq_path) and
        any(f.endswith((".safetensors", ".bin")) for f in os.listdir(_awq_path))):
    print("  Downloading AWQ model...")
    snapshot_download(repo_id=_HF_REPOS["Qwen2-1.5B-AWQ"], local_dir=_awq_path)
else:
    print(f"  Found AWQ model")

# Ensure GGUF Q4_K_M is present
_q4_present = (
    os.path.isdir(_GGUF_DIR) and
    any("q4_k_m" in f.lower() and f.endswith(".gguf") for f in os.listdir(_GGUF_DIR))
)
if not _q4_present:
    print("  Downloading GGUF Q4_K_M...")
    os.makedirs(_GGUF_DIR, exist_ok=True)
    snapshot_download(
        repo_id="MaziyarPanahi/Qwen2-1.5B-Instruct-GGUF",
        local_dir=_GGUF_DIR,
        allow_patterns=["*Q4_K_M*.gguf"],
    )
else:
    print("  Found GGUF Q4_K_M")

# ==========================
# LOAD WIKINER
# ==========================
def load_wikiner(path):
    sentences, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            words = line.split()
            tokens, tags = [], []
            for word in words:
                parts = word.split("|")
                if len(parts) == 3:
                    tokens.append(parts[0])
                    tags.append(parts[2])
            sentences.append(tokens)
            labels.append(tags)
    return sentences, labels


def extract_gold_persons(tokens, tags):
    persons, current = [], []
    for token, tag in zip(tokens, tags):
        if tag == "B-PER":
            if current:
                persons.append(" ".join(current))
            current = [token]
        elif tag == "I-PER":
            current.append(token)
        else:
            if current:
                persons.append(" ".join(current))
                current = []
    if current:
        persons.append(" ".join(current))
    return persons

# ==========================
# NORMALIZATION
# ==========================
def normalize_name(name: str) -> str:
    name = name.replace("ё", "е").replace("Ё", "Е")
    name = name.lower().strip()
    title_words = {
        "князь", "принц", "царь", "боярин", "атаман",
        "новгородский", "немецкого", "казачьего", "императрица",
        "граф", "барон", "король", "президент", "генерал",
    }
    parts = [p for p in name.split() if p not in title_words]
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def soft_match(gold_list, pred_list):
    gold_norm = [normalize_name(g) for g in gold_list]
    pred_norm = [normalize_name(p) for p in pred_list]
    matched_gold, matched_pred = set(), set()

    for gi, gn in enumerate(gold_norm):
        g_parts = gn.split()
        for pi, pn in enumerate(pred_norm):
            if pi in matched_pred:
                continue
            p_parts = pn.split()
            if gn == pn:
                matched_gold.add(gi); matched_pred.add(pi); break
            if len(p_parts) == 1 and p_parts[0] in g_parts:
                matched_gold.add(gi); matched_pred.add(pi); break
            if len(g_parts) == 1 and g_parts[0] in p_parts:
                matched_gold.add(gi); matched_pred.add(pi); break
            def stem(s): return s[:5] if len(s) > 4 else s
            g_stems = set(stem(t) for t in g_parts if not re.match(r'^[а-я]\.$', t))
            p_stems = set(stem(t) for t in p_parts if not re.match(r'^[а-я]\.$', t))
            shorter = g_stems if len(g_stems) <= len(p_stems) else p_stems
            longer  = p_stems if len(g_stems) <= len(p_stems) else g_stems
            if shorter and shorter.issubset(longer):
                matched_gold.add(gi); matched_pred.add(pi); break

    tp = len(matched_gold)
    fp = len(pred_norm) - len(matched_pred)
    fn = len(gold_norm) - len(matched_gold)
    return tp, fp, fn

# ==========================
# BLOCKLIST
# ==========================
BLOCKLIST_PATTERNS = [
    r"^(суши|земля|лучше|граничит|родилось|умерло|рождаемость|смертность)$",
    r"^\d[\d\s,\.–—\-]*$",
    r"\b(организация|комитет|союз охраны|государственная дума|думу|верховный совет"
    r"|центральный банк|варшавский договор|московский университет|петербургский университет"
    r"|русская армия|семилетняя война|февральская революция|гражданская война"
    r"|карибский кризис|августовский путч|третьяковская галерея)\b",
    r"^(пруссия|ливония|киев|казань|литва|россия|канада|неман|нерис|москва|сибирь"
    r"|астрахань|украина|белоруссия|германия|париж|болгария|сербия|черногория"
    r"|петроград|воронеж|ельце|амур|лена|енисей|иртыш|обь|волга|кама"
    r"|байкал|ладожское|онежское|сочи|корея|швеция|северная|литовцы|польский"
    r"|дерпте|вильно|харьков)$",
    r"^(красные|массы|историки|миллионы|человека|депутаты|депутат|народ)$",
    r"^.{1,2}$",
    r"^(песнь|идиот|смерть иоанна|бахчисарайский|декрет о|"
    r"первую всеобщую|российскую империю|современные границы).*",
    r"^(социальное пространство|радикальные изменения|партийные списки|список"
    r"|срок|срока|охрана природы|охраны|природы|организаций|сообщества"
    r"|страны|мира|государств|научные|общественные|работа|митинг|мир)$",
    r"^(первый|второй|третий|четвёртый|пятый|последний|новый|старый|великий|малый)(\s+\w+)?$",
    r"(ующий|ающий|яющий|овавший|евавший|ившийся|овавшись)$",
    r"^(русский|русская|немецкий|польский|литовский|шведский|французский"
    r"|английский|советский|российский|татарский|монгольский)(\s+\w+)?$",
    r"\b(в результате|в ходе|в связи|по итогам|на основе|в рамках|в период|в составе)\b",
    r"^(президент|министр|генерал|адмирал|маршал|директор|председатель|секретарь"
    r"|патриарх|митрополит|епископ|академик|профессор|доктор)$",
    r"(ское|ское озеро|ское море|ский залив|ская губерния)$",
]
_BLOCKLIST_RE = re.compile("|".join(BLOCKLIST_PATTERNS), re.IGNORECASE | re.UNICODE)
_ROMAN_RE = re.compile(r'^(I{1,3}|IV|VI{0,3}|IX|XI{0,3}|XIV|XV|XVI)$')


def is_valid_person(name: str) -> bool:
    n = normalize_name(name)
    if _BLOCKLIST_RE.search(n):
        return False
    if not re.search(r"[а-яёa-zА-ЯЁA-Z]", n):
        return False
    return True


def appears_capitalized_in_sentence(name: str, sentence: str) -> bool:
    sentence_words = sentence.split()
    for word in name.split():
        if re.match(r'^[А-ЯЁA-Z]\.$', word):
            continue
        if _ROMAN_RE.match(word):
            continue
        if re.match(r'^[А-ЯЁA-Z]\.[А-ЯЁA-Z]', word):
            if not any(sw == word for sw in sentence_words):
                return False
            continue
        found = any(
            sw[0].isupper() and sw.lower().rstrip('.,;:!?') == word.lower()
            for sw in sentence_words if sw
        )
        if not found:
            return False
    return True


def all_words_in_sentence(name: str, sentence: str) -> bool:
    sentence_lower = sentence.lower()
    return all(word.lower() in sentence_lower for word in name.split())


def deduplicate_predictions(predictions: list) -> list:
    seen, deduped = set(), []
    for p in predictions:
        key = normalize_name(p)
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    return deduped

# ==========================
# PROMPTS
# ==========================
SYSTEM_PROMPT = """You are a strict Named Entity Recognition (NER) system for Russian text.

TASK: Extract ONLY the names of real people (persons) from the sentence.

STRICT RULES:
1. Output ONLY a JSON array — no explanation, no markdown.
2. Include: personal names, surnames, patronymics, initials referring to people.
3. Include names WITH their titles if the title is part of how they are referred to (e.g. "Иван IV Грозный").
4. Include names in ANY grammatical case (nominative, genitive, dative, etc.). For example, "Ермака", "Петра I", "Екатерине II" are still person names even though they are inflected.
5. DO NOT include: cities, countries, rivers, mountains, organizations, political parties, historical events, book/film titles, abstract concepts, or any non-person entity.
6. If NO people are mentioned, return exactly: []
7. Use the EXACT form of the name as it appears in the text.
8. Do NOT repeat the same person multiple times in the output array.

Output format:
[{"text": "Name as it appears"}]

Examples:

Input: В договоре упоминается Миндовг .
Output: [{"text": "Миндовг"}]

Input: Принц Вильгельм фон Урах был приглашён на престол .
Output: [{"text": "Вильгельм фон Урах"}]

Input: На престол было решено пригласить немецкого принца Вильгельма фон Ураха .
Output: [{"text": "Вильгельма фон Ураха"}]

Input: Ленин и Троцкий выступили в Петрограде .
Output: [{"text": "Ленин"}, {"text": "Троцкий"}]

Input: Наиболее крупные реки : Амур , Лена , Енисей .
Output: []

Input: Государственная Дума состоит из 450 депутатов .
Output: []

Input: Пять поэтов : И. А. Бунин , Б. Л. Пастернак , М. А. Шолохов .
Output: [{"text": "И. А. Бунин"}, {"text": "Б. Л. Пастернак"}, {"text": "М. А. Шолохов"}]

Input: Основные концепции : Т. Парсонс , К. Маркс , М. Фуко , Р. Дарендорф .
Output: [{"text": "Т. Парсонс"}, {"text": "К. Маркс"}, {"text": "М. Фуко"}, {"text": "Р. Дарендорф"}]

Input: Ю.Лотман рассматривал социальное пространство как разграничения на внутреннее и внешнее .
Output: [{"text": "Ю.Лотман"}]

Input: Князь Ярослав Мудрый утвердил Русскую Правду .
Output: [{"text": "Ярослав Мудрый"}]

Input: В 882 году новгородский князь Олег захватил Киев .
Output: [{"text": "Олег"}]

Input: Екатерина II придавала театру высокое значение .
Output: [{"text": "Екатерина II"}]

Input: При Екатерине II учреждаются штаты монастырей .
Output: [{"text": "Екатерине II"}]

Input: После реформ Александра II купцы начинают играть важную роль в городской жизни .
Output: [{"text": "Александра II"}]

Input: Пётр I ведёт наступление на монастыри .
Output: [{"text": "Пётр I"}]

Input: С похода казачьего атамана Ермака в 1581 году началось покорение Сибири .
Output: [{"text": "Ермака"}]

Input: Начало его связано с именами царя Алексея Михайловича и боярина Матвеева .
Output: [{"text": "Алексея Михайловича"}, {"text": "Матвеева"}]

Input: При Павле I и Александре I монастырские штаты увеличиваются .
Output: [{"text": "Павле I"}, {"text": "Александре I"}]

Input: По определению Энтони Гидденса , социология -- это изучение общественной жизни .
Output: [{"text": "Энтони Гидденса"}]

Input: Пьер Бурдье считает , что социальное пространство -- систематизированные пересечения .
Output: [{"text": "Пьер Бурдье"}]

Input: Россия граничит с Норвегией , Финляндией и Эстонией .
Output: []

Input: Великая Отечественная война длилась с 1941 по 1945 год .
Output: []

Input: Московский университет был основан в 1755 году .
Output: []

Input: Красная армия перешла в наступление на всех фронтах .
Output: []

Input: Председатель Верховного Совета выступил с речью .
Output: []

Input: Сталин и его окружение взяли курс на коллективизацию деревни .
Output: [{"text": "Сталин"}]
"""

VERIFY_PROMPT = """You are verifying whether a string is a real person's name as used in a Russian sentence.
Answer ONLY with a single word: YES or NO.

Rules:
- YES: the string refers to a real human being (first name, surname, patronymic, or combination)
- YES: Russian names may appear in genitive or dative form — "Ермака", "Петрова", "Александра", "Екатерине" are still person names when used in context
- YES: Short initial+surname combinations like "Т. Парсонс", "К. Маркс", "М. Фуко" are person names
- YES: No-space initial forms like "Ю.Лотман", "И.Гофман" are person names
- NO: the string is a city, country, organization, event, concept, standalone title (Президент, Генерал), or anything else not referring to a specific person
- NO: "Русские" is an ethnic group, not a person's name
- NO: "Литву", "Латвию", "Эстонию" are countries in accusative case, not persons
- YES: "Маннергейма" is a person (Finnish general) in genitive case
- YES: "Бриана", "Келлога" are persons (signatories of the Briand-Kellogg pact)
"""

# ==========================
# EXTRACTION HELPERS
# ==========================
def extract_json_persons(response: str) -> list:
    match = re.search(r"\[.*?\]", response, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                results = []
                for item in parsed:
                    if isinstance(item, dict) and "text" in item:
                        results.append(str(item["text"]).strip())
                    elif isinstance(item, str):
                        results.append(item.strip())
                return [r for r in results if r]
        except json.JSONDecodeError:
            pass
    bracket_match = re.search(r"\[(.*?)\]", response, re.DOTALL)
    if bracket_match:
        quoted = re.findall(r'"([^"]+)"', bracket_match.group(1))
        if quoted:
            return [q.strip() for q in quoted if q.strip()]
    return []

# ==========================
# OAQ LAYER
# ==========================
class OAQLinear(nn.Module):
    def __init__(self, quantized_weight, scales, outlier_weight,
                 bias, in_features, out_features, group_size, bits):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.group_size   = group_size
        self.bits         = bits
        self.register_buffer("quantized_weight", quantized_weight)
        self.register_buffer("scales",           scales)
        self.register_buffer("outlier_weight",   outlier_weight)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None
        self._dequantized_cache = None

    def dequantize(self) -> torch.Tensor:
        if self._dequantized_cache is not None:
            return self._dequantized_cache
        w_q        = self.quantized_weight.float()
        s          = self.scales.float()
        s_expanded = s.repeat_interleave(self.group_size, dim=1)
        s_expanded = s_expanded[:, :self.in_features]
        w_deq      = (w_q * s_expanded + self.outlier_weight.float()).to(torch.float16)
        self._dequantized_cache = w_deq
        return w_deq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w   = self.dequantize()
        out = nn.functional.linear(x, w, None)
        if self.bias is not None:
            out = out + self.bias
        return out


def _oaq_replace_layers(model, bits, group_size, outlier_ratio):
    import torch.nn.functional as F
    def _replace(parent):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and "lm_head" not in child_name:
                w = child.weight.data.float()
                out_f, in_f = w.shape
                threshold    = torch.quantile(w.abs().flatten(), 1.0 - outlier_ratio)
                outlier_mask = w.abs() >= threshold
                outlier_w    = (w * outlier_mask.float()).half()
                remaining    = w * (~outlier_mask).float()
                pad          = (group_size - in_f % group_size) % group_size
                if pad > 0:
                    remaining = F.pad(remaining, (0, pad))
                _, in_pad = remaining.shape
                n_groups  = in_pad // group_size
                r_grouped = remaining.reshape(out_f, n_groups, group_size)
                max_val   = r_grouped.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)
                qmax      = 2 ** (bits - 1) - 1
                scales    = (max_val / qmax).squeeze(2)
                scale_exp = scales.unsqueeze(2).expand_as(r_grouped)
                q_grouped = torch.round(r_grouped / scale_exp).clamp(-qmax - 1, qmax)
                q_flat    = q_grouped.reshape(out_f, in_pad)[:, :in_f].to(torch.int8)
                b = child.bias.data.half().cpu() if child.bias is not None else None
                setattr(parent, child_name, OAQLinear(
                    quantized_weight = q_flat.cpu(),
                    scales           = scales.half().cpu(),
                    outlier_weight   = outlier_w.cpu(),
                    bias             = b,
                    in_features      = child.in_features,
                    out_features     = child.out_features,
                    group_size       = group_size,
                    bits             = bits,
                ))
            else:
                _replace(child)
    _replace(model)


def _model_dtype(model) -> torch.dtype:
    for p in model.parameters():
        if p.is_floating_point():
            return p.dtype
    return torch.float16

# ==========================
# MODEL BACKENDS
# ==========================
class HFBackend:
    """Handles FP16, bnb-4bit, bnb-8bit, AWQ, GPTQ, HQQ, TorchAO."""

    def __init__(self, path: str, bnb_bits: int = 0,
                 hqq_bits: int = 0, hqq_group: int = 64,
                 torchao_dtype: str = ""):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import json as _json

        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.gen_dtype = torch.float16

        # ── HQQ ──────────────────────────────────────────────────────────
        if hqq_bits > 0:
            from hqq.models.hf.base import AutoHQQHFModel
            from hqq.core.quantize import BaseQuantizeConfig
            save_path = os.path.join(path, f"quantized_model_hqq_{hqq_bits}b_g{hqq_group}")
            if os.path.isdir(save_path) and any(
                f.endswith((".safetensors", ".bin")) for f in os.listdir(save_path)
            ):
                print(f"    Loading cached HQQ model from {save_path}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    save_path, torch_dtype=torch.float16,
                    device_map="cuda:0", trust_remote_code=True)
            else:
                print(f"    Loading with HQQ {hqq_bits}-bit (group_size={hqq_group})")
                self.model = AutoModelForCausalLM.from_pretrained(
                    path, torch_dtype=torch.float16,
                    device_map="cuda:0", trust_remote_code=True)
                quant_config = BaseQuantizeConfig(nbits=hqq_bits, group_size=hqq_group)
                AutoHQQHFModel.quantize_model(
                    self.model, quant_config=quant_config,
                    compute_dtype=torch.float16, device=DEVICE)
                print(f"    Saving HQQ model to {save_path}")
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
            self.model.eval()
            self.gen_dtype = _model_dtype(self.model)
            return

        # ── TorchAO ───────────────────────────────────────────────────────
        # FIX: this block is at the correct indentation level (NOT inside hqq block)
        if torchao_dtype:
            print(f"    Loading with TorchAO {torchao_dtype}")
            self.model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.bfloat16,
                device_map=None, trust_remote_code=True)
            if torchao_dtype == "int8_weight_only":
                try:
                    from torchao.quantization import quantize_, Int8WeightOnlyConfig
                    quantize_(self.model, Int8WeightOnlyConfig())
                except (ImportError, TypeError):
                    from torchao.quantization import quantize_
                    from torchao.quantization.quant_api import int8_weight_only
                    quantize_(self.model, int8_weight_only())
            elif torchao_dtype == "int4_weight_only":
                try:
                    from torchao.quantization import quantize_, Int4WeightOnlyConfig
                    quantize_(self.model, Int4WeightOnlyConfig(group_size=64))
                except (ImportError, TypeError):
                    from torchao.quantization import quantize_
                    from torchao.quantization.quant_api import int4_weight_only
                    quantize_(self.model, int4_weight_only(group_size=64))
            else:
                raise ValueError(f"Unknown torchao_dtype: {torchao_dtype}")
            if DEVICE == "cuda":
                self.model = self.model.to("cuda")
            self.model.eval()
            self.gen_dtype = torch.bfloat16
            return

        # ── Auto-detect AWQ / GPTQ from config.json ───────────────────────
        _config_file = os.path.join(path, "config.json")
        _quant_bits  = bnb_bits

        if _quant_bits == 0 and os.path.exists(_config_file):
            with open(_config_file, encoding="utf-8") as _f:
                _cfg = _json.load(_f)
            _quant = _cfg.get("quantization_config", {}) or {}
            _quant_method = (
                _quant.get("quant_method", "") or
                _quant.get("quantization_algo", "")
            )
            if _quant_method in ("awq", "gptq"):
                print(f"    Detected {_quant_method.upper()} quantization")
                self.model = AutoModelForCausalLM.from_pretrained(
                    path, dtype=torch.float16,
                    device_map="auto", trust_remote_code=True)
                self.model.eval()
                return
            elif _quant.get("quant_type") == "nf4" or _quant.get("load_in_4bit", False):
                _quant_bits = 4
            elif _quant.get("load_in_8bit", False):
                _quant_bits = 8

        # ── bitsandbytes ──────────────────────────────────────────────────
        if _quant_bits in (4, 8) and DEVICE != "cuda":
            raise EnvironmentError(f"bnb-{_quant_bits}bit requires CUDA.")

        if _quant_bits == 4:
            print("    Loading with BitsAndBytes NF4 4-bit")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                path, quantization_config=bnb_config,
                device_map="auto", trust_remote_code=True)
        elif _quant_bits == 8:
            print("    Loading with BitsAndBytes 8-bit")
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                path, quantization_config=bnb_config,
                device_map="auto", trust_remote_code=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device_map="auto" if DEVICE == "cuda" else None,
                trust_remote_code=True)
            if DEVICE == "cpu":
                self.model = self.model.to(torch.float32)

        self.model.eval()

    def _input_device(self):
        for p in self.model.parameters():
            if p.device.type != "meta":
                return p.device
        return torch.device(DEVICE)

    def _run_generate(self, inputs, **kwargs):
        if DEVICE == "cuda":
            with torch.no_grad(), torch.autocast("cuda", dtype=self.gen_dtype):
                return self.model.generate(**inputs, **kwargs)
        with torch.no_grad():
            return self.model.generate(**inputs, **kwargs)

    def generate(self, messages: list, max_new_tokens: int = 400,
                 do_sample: bool = False, repetition_penalty: float = 1.0) -> str:
        prompt  = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs_raw = self.tokenizer(prompt, return_tensors="pt")
        target_device = self._input_device()
        inputs = {k: v.to(target_device) for k, v in inputs_raw.items()}
        outputs = self._run_generate(
            inputs, max_new_tokens=max_new_tokens,
            do_sample=do_sample, repetition_penalty=repetition_penalty)
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True).strip()

    def unload(self):
        del self.model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


class QuantoBackend:
    """
    Uses optimum-quanto directly (not transformers QuantoConfig).
    FIX: device_map=None avoids accelerate hooks that break freeze().
    Install: pip install optimum-quanto
    """
    def __init__(self, path: str, weight_bits: int = 4):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from optimum.quanto import quantize, freeze, qint4, qint8

        print(f"    Loading with Quanto {weight_bits}-bit weight-only")
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map=None,
            trust_remote_code=True,
        )
        w_dtype = qint4 if weight_bits == 4 else qint8
        quantize(self.model, weights=w_dtype, activations=None)
        freeze(self.model)
        if DEVICE == "cuda":
            self.model = self.model.to("cuda")
        self.model.eval()
        self.gen_dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    def generate(self, messages: list, max_new_tokens: int = 400,
                 do_sample: bool = False, repetition_penalty: float = 1.0) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs_raw = self.tokenizer(prompt, return_tensors="pt")
        target_device = next(self.model.parameters()).device
        inputs = {k: v.to(target_device) for k, v in inputs_raw.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=do_sample, repetition_penalty=repetition_penalty)
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True).strip()

    def unload(self):
        del self.model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


class GGUFBackend:
    def __init__(self, path: str, pattern: str = ""):
        from llama_cpp import Llama
        if os.path.isdir(path):
            gguf_files = sorted(f for f in os.listdir(path) if f.endswith(".gguf"))
            if not gguf_files:
                raise FileNotFoundError(f"No .gguf file found in folder: {path}")
            matched = [f for f in gguf_files if pattern.lower() in f.lower()] if pattern else gguf_files
            chosen  = matched[0] if matched else gguf_files[0]
            path    = os.path.join(path, chosen)
            print(f"  Auto-selected GGUF file: {os.path.basename(path)}")
        self.llm = Llama(
            model_path=path, n_ctx=4096,
            n_gpu_layers=-1 if DEVICE == "cuda" else 0,
            seed=RANDOM_SEED, verbose=False)

    def generate(self, messages: list, max_new_tokens: int = 400,
                 do_sample: bool = False, repetition_penalty: float = 1.0) -> str:
        response = self.llm.create_chat_completion(
            messages=messages, max_tokens=max_new_tokens,
            temperature=0.0 if not do_sample else 0.7,
            repeat_penalty=repetition_penalty)
        return response["choices"][0]["message"]["content"].strip()

    def unload(self):
        del self.llm
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


class OAQBackend:
    def __init__(self, path: str):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        oaq_cfg_path = os.path.join(path, "oaq_config.json")
        if not os.path.exists(oaq_cfg_path):
            raise FileNotFoundError(f"No oaq_config.json in {path}. Run quantize_oaq.py first.")
        with open(oaq_cfg_path) as f:
            oaq_cfg = json.load(f)
        print(f"    OAQ: {oaq_cfg['bits']}-bit, group={oaq_cfg['group_size']}, outlier={oaq_cfg['outlier_ratio']:.1%}")
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            oaq_cfg["base_model"], torch_dtype=torch.float16, trust_remote_code=True)
        _oaq_replace_layers(base_model, bits=oaq_cfg["bits"],
                            group_size=oaq_cfg["group_size"],
                            outlier_ratio=oaq_cfg["outlier_ratio"])
        state_dict_path = os.path.join(path, "oaq_model.pt")
        if not os.path.exists(state_dict_path):
            raise FileNotFoundError(f"oaq_model.pt not found in {path}")
        state_dict = torch.load(state_dict_path, map_location="cpu")
        base_model.load_state_dict(state_dict)
        if DEVICE == "cuda":
            base_model = base_model.to(DEVICE)
        base_model.eval()
        self.model = base_model
        print("    Pre-warming OAQ dequantization cache...")
        dummy = torch.zeros(1, 1, dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            self.model(dummy)
        print("    Cache ready.")

    def generate(self, messages: list, max_new_tokens: int = 400,
                 do_sample: bool = False, repetition_penalty: float = 1.0) -> str:
        prompt  = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs  = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=do_sample, repetition_penalty=repetition_penalty)
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True).strip()

    def unload(self):
        del self.model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


class ClassicalBackend:
    def __init__(self, kind: str, model_name: str = ""):
        self.kind = kind
        if kind == "spacy":
            import spacy
            name = model_name or "ru_core_news_sm"
            try:
                self.nlp = spacy.load(name)
            except OSError:
                import subprocess
                print(f"    Downloading spaCy model {name}...")
                subprocess.check_call([sys.executable, "-m", "spacy", "download", name])
                self.nlp = spacy.load(name)
            print(f"    Loaded spaCy model: {name}")
        elif kind == "hf_ner":
            from transformers import pipeline
            name = model_name or "surdan/LaBSE_ner_nerel"
            print(f"    Loading HF NER pipeline: {name}")
            self.pipe = pipeline(
                "token-classification", model=name,
                aggregation_strategy="simple",
                device=0 if DEVICE == "cuda" else -1)
        else:
            raise ValueError(f"Unknown classical kind: {kind}")

    def extract_persons(self, sentence: str) -> list:
        if self.kind == "spacy":
            doc = self.nlp(sentence)
            return [ent.text.strip() for ent in doc.ents if ent.label_ == "PER"]
        elif self.kind == "hf_ner":
            entities = self.pipe(sentence)
            return [e["word"].strip() for e in entities
                    if e["entity_group"] in ("PER", "PERSON")]

    def generate(self, messages, **kwargs):
        raise NotImplementedError("Use extract_persons() for classical backends")

    def unload(self):
        if hasattr(self, "nlp"):
            del self.nlp
        if hasattr(self, "pipe"):
            del self.pipe
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

# ==========================
# INFERENCE HELPERS
# ==========================
def verify_person(backend, name: str, sentence: str) -> bool:
    messages = [
        {"role": "system", "content": VERIFY_PROMPT},
        {"role": "user",   "content": f'Sentence: {sentence}\nIs "{name}" a person\'s name? Answer YES or NO only.'},
    ]
    response = backend.generate(messages, max_new_tokens=5, do_sample=False).upper()
    return response.startswith("YES")


def generate_persons(backend, sentence: str) -> list:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": sentence},
    ]
    response = backend.generate(messages, max_new_tokens=400, do_sample=False, repetition_penalty=1.0)
    raw      = extract_json_persons(response)
    filtered = [p for p in raw if is_valid_person(p)]
    filtered = [p for p in filtered if all_words_in_sentence(p, sentence)]
    filtered = [p for p in filtered if appears_capitalized_in_sentence(p, sentence)]
    filtered = deduplicate_predictions(filtered)
    return [p for p in filtered if verify_person(backend, p, sentence)]


def generate_persons_classical(backend: ClassicalBackend, sentence: str) -> list:
    raw      = backend.extract_persons(sentence)
    filtered = [p for p in raw if is_valid_person(p)]
    filtered = [p for p in filtered if all_words_in_sentence(p, sentence)]
    filtered = [p for p in filtered if appears_capitalized_in_sentence(p, sentence)]
    return deduplicate_predictions(filtered)

# ==========================
# DATASET PREP
# ==========================
print("Loading dataset...")
sentences, labels = load_wikiner(DATASET_PATH)

if SHUFFLE_DATASET:
    random.seed(RANDOM_SEED)
    paired = list(zip(sentences, labels))
    random.shuffle(paired)
    sentences, labels = zip(*paired)
    sentences, labels = list(sentences), list(labels)
    print(f"  Shuffled dataset (seed={RANDOM_SEED}).")
else:
    print("  Using dataset in original file order.")

sentences = sentences[:MAX_SAMPLES]
labels    = labels[:MAX_SAMPLES]
print(f"  Using {len(sentences)} samples.\n")

# ==========================
# BACKEND FACTORY
# ==========================
def load_backend(model_cfg: dict):
    mtype = model_cfg["type"]
    path  = model_cfg["path"]
    if mtype == "oaq":
        return OAQBackend(path)
    elif mtype == "hf":
        return HFBackend(path, bnb_bits=model_cfg.get("bnb_bits", 0))
    elif mtype == "hqq":
        return HFBackend(path, hqq_bits=model_cfg.get("hqq_bits", 4),
                         hqq_group=model_cfg.get("hqq_group", 64))
    elif mtype == "torchao":
        return HFBackend(path, torchao_dtype=model_cfg.get("torchao_dtype", ""))
    elif mtype == "quanto":
        return QuantoBackend(path, weight_bits=model_cfg.get("quanto_weight_bits", 4))
    elif mtype == "gguf":
        return GGUFBackend(path, pattern=model_cfg.get("gguf_pattern", ""))
    elif mtype == "spacy":
        return ClassicalBackend("spacy", model_cfg.get("model_name", ""))
    elif mtype == "hf_ner":
        return ClassicalBackend("hf_ner", model_cfg.get("model_name", ""))
    else:
        raise ValueError(f"Unknown model type: {mtype}")

# ==========================
# MAIN LOOP
# ==========================
os.makedirs(OUTPUT_DIR, exist_ok=True)
summary = []

for model_cfg in MODELS:
    label = model_cfg["label"]

    if model_cfg["type"] == "oaq":
        oaq_cfg_path = os.path.join(model_cfg["path"], "oaq_config.json")
        if not os.path.exists(oaq_cfg_path):
            print(f"\n  [SKIPPED] {label} — run quantize_oaq.py first")
            summary.append({
                "model": label, "precision": None, "recall": None, "f1": None,
                "tp": None, "fp": None, "fn": None, "skipped": True,
                "skip_reason": "OAQ model not generated yet",
            })
            continue

    print(f"\n{'='*60}")
    print(f"  Model: {label}")
    print(f"{'='*60}")
    print("  Loading model...")

    try:
        backend = load_backend(model_cfg)
    except Exception as e:
        print(f"\n  [SKIPPED] {e}\n")
        summary.append({
            "model": label, "precision": None, "recall": None, "f1": None,
            "tp": None, "fp": None, "fn": None, "skipped": True,
            "skip_reason": str(e),
        })
        continue

    print("  Model loaded. Running inference...\n")

    global_tp = global_fp = global_fn = 0
    all_results, inference_times = [], []

    for tokens, tags in tqdm(zip(sentences, labels), total=len(sentences), desc=label):
        sentence     = " ".join(tokens)
        gold_persons = extract_gold_persons(tokens, tags)

        t0 = time.perf_counter()
        if isinstance(backend, ClassicalBackend):
            pred_persons = generate_persons_classical(backend, sentence)
        else:
            pred_persons = generate_persons(backend, sentence)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        tp, fp, fn = soft_match(gold_persons, pred_persons)
        global_tp += tp
        global_fp += fp
        global_fn += fn
        inference_times.append(elapsed_ms)

        all_results.append({
            "sentence":          sentence,
            "gold_persons":      gold_persons,
            "predicted_persons": pred_persons,
            "tp": tp, "fp": fp, "fn": fn,
            "inference_ms":      round(elapsed_ms, 2),
        })

    precision = global_tp / (global_tp + global_fp + 1e-8)
    recall    = global_tp / (global_tp + global_fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    avg_ms    = sum(inference_times) / len(inference_times)
    total_s   = sum(inference_times) / 1000

    print(f"\n  ── Results for {label} ──")
    print(f"  Precision:    {precision:.4f}")
    print(f"  Recall:       {recall:.4f}")
    print(f"  F1 Score:     {f1:.4f}")
    print(f"  TP={global_tp}  FP={global_fp}  FN={global_fn}")
    print(f"  Avg latency:  {avg_ms:.1f} ms/sentence")
    print(f"  Total time:   {total_s:.1f}s")

    safe_label  = re.sub(r"[^\w\-]", "_", label)
    output_path = os.path.join(OUTPUT_DIR, f"predictions_{safe_label}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "model":             label,
            "precision":         round(precision, 4),
            "recall":            round(recall, 4),
            "f1":                round(f1, 4),
            "tp": global_tp, "fp": global_fp, "fn": global_fn,
            "avg_inference_ms":  round(avg_ms, 2),
            "total_inference_s": round(total_s, 2),
            "shuffle":           SHUFFLE_DATASET,
            "random_seed":       RANDOM_SEED if SHUFFLE_DATASET else None,
            "results":           all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"  Saved → {output_path}")

    summary.append({
        "model":     label,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "tp": global_tp, "fp": global_fp, "fn": global_fn,
        "avg_ms":    round(avg_ms, 2),
    })

    print("  Unloading model...")
    backend.unload()
    print("  Done.\n")

# ==========================
# SUMMARY
# ==========================
print("\n" + "=" * 70)
print("  SUMMARY — ALL MODELS")
print("=" * 70)
print(f"{'Model':<45} {'P':>7} {'R':>7} {'F1':>7} {'ms/s':>8}")
print("-" * 70)
for s in summary:
    if s.get("skipped"):
        print(f"{s['model']:<45} {'SKIPPED':>7}  {s.get('skip_reason', '')}")
    else:
        print(f"{s['model']:<45} {s['precision']:>7.4f} {s['recall']:>7.4f} {s['f1']:>7.4f} {s['avg_ms']:>8.1f}")

summary_path = os.path.join(OUTPUT_DIR, "summary_all_models.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"\nSummary saved → {summary_path}")

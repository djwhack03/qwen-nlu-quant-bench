import re
import json

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

_BLOCKLIST_RE = re.compile(
    "|".join(BLOCKLIST_PATTERNS), re.IGNORECASE | re.UNICODE)
_ROMAN_RE = re.compile(
    r'^(I{1,3}|IV|VI{0,3}|IX|XI{0,3}|XIV|XV|XVI)$')

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


# ==========================
# VALIDATION FILTERS
# ==========================
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
# SOFT MATCHING (for evaluation)
# ==========================
def soft_match(gold_list: list, pred_list: list):
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
            g_stems = set(stem(t) for t in g_parts
                         if not re.match(r'^[а-я]\.$', t))
            p_stems = set(stem(t) for t in p_parts
                         if not re.match(r'^[а-я]\.$', t))
            shorter = g_stems if len(g_stems) <= len(p_stems) else p_stems
            longer  = p_stems if len(g_stems) <= len(p_stems) else g_stems
            if shorter and shorter.issubset(longer):
                matched_gold.add(gi); matched_pred.add(pi); break

    tp = len(matched_gold)
    fp = len(pred_norm) - len(matched_pred)
    fn = len(gold_norm) - len(matched_gold)
    return tp, fp, fn


# ==========================
# JSON EXTRACTION FROM LLM RESPONSE
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
# SENTIMENT LABEL NORMALIZATION
# ==========================
def normalize_sentiment_label(label) -> str:
    if isinstance(label, int):
        return "positive" if label == 1 else "negative"
    label = str(label).lower()
    if "pos" in label: return "positive"
    if "neg" in label: return "negative"
    return "neutral"

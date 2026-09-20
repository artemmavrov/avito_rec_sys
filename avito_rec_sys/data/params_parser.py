"""Whitelist-based parser for `item_infm_params_text`.

The raw field is a flat, undelimited "Key1 Value1 Key2 Value2 ..." string
(Avito's own attribute schema, one key/value run per attribute). There is no
separator between a key and its value or between consecutive pairs, so the
only way to segment it is to know the set of possible key strings and scan
for their occurrences.

WHITELIST is the small set of keys worth keeping: the query side only ever
filters on service type, so those are the only item-side fields whose
vocabulary matches the query text. Everything else (addresses, price lists,
schedules) is noise that eats the token budget without helping any query
match it.

SPLIT_KEYS is the delimiter vocabulary needed to bound each matched key's
value (to know where it ends). It does not need to be exhaustive: it needs
the keys that plausibly follow a whitelisted key, so that a whitelisted value
does not swallow the next field's content. It was mined from a 40k-item
sample of the corpus.
"""

from __future__ import annotations

import re

WHITELIST = {"Вид услуги", "Тип услуги", "Название услуги", "Своя услуга", "Услуга"}

SPLIT_KEYS = [
    "Тип стоимости за услугу", "Работаете с юрлицами и ип", "Как вы работаете",
    "Куда выезжаете", "Название услуги", "Место оказания услуг",
    "График работы от", "График работы до", "Рабочие дни", "Марка авто",
    "Продолжительность", "Начальная цена", "Опыт работы", "Производители",
    "Вид услуги", "Тип услуги", "Своя услуга", "Стоимость", "Время для",
    "Работа по договору", "Бригада", "Услуга", "График", "Время",
]


def build_key_regex(keys: list[str] | set[str]) -> re.Pattern:
    """Compile an alternation regex, longest key first, so overlapping keys
    (e.g. "Услуга" vs "Своя услуга") are matched greedily/correctly."""
    keys_sorted = sorted(set(keys), key=len, reverse=True)
    pattern = "|".join(re.escape(k) for k in keys_sorted)
    return re.compile(pattern)


_SPLIT_REGEX = build_key_regex(SPLIT_KEYS)


def parse_params(text: str | None, key_regex: re.Pattern = _SPLIT_REGEX) -> list[tuple[str, str]]:
    """Split raw params text into an ordered list of (key, value) pairs.

    Any leading text before the first recognized key (should be rare/empty
    in practice) is dropped -- there's no key to attach it to.
    """
    if not text:
        return []
    matches = list(key_regex.finditer(text))
    if not matches:
        return []
    pairs = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        value = text[m.end() : end].strip(" ,")
        pairs.append((m.group(0), value))
    return pairs


def extract_whitelisted(
    text: str | None,
    key_regex: re.Pattern = _SPLIT_REGEX,
    whitelist: set[str] = WHITELIST,
) -> str:
    """Return only the whitelisted key/value pairs, as `Key Value Key Value ...`.

    Segments are de-duplicated (price lists repeat the same service name
    across several price tiers) while preserving first-seen order.
    """
    pairs = parse_params(text, key_regex)
    seen: set[tuple[str, str]] = set()
    out_parts: list[str] = []
    for key, value in pairs:
        if key not in whitelist or not value:
            continue
        dedup_key = (key, value)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        out_parts.append(key)
        out_parts.append(value)
    return " ".join(out_parts)


def build_item_tower_text(title: str | None, params_text: str | None) -> str:
    """Item-tower input: title + whitelist-filtered params.

    This exact string is also the dedup/denoise KEY (full item-
    tower text, not title alone) -- if two items produce the same string
    here, the encoder would see an identical input for both, so they must
    not both be kept as distinct train pairs / negatives for the same query.
    """
    title = title or ""
    filtered_params = extract_whitelisted(params_text)
    if not filtered_params:
        return title.strip()
    return f"{title.strip()} {filtered_params}".strip()

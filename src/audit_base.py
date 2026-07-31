"""
аудит базы: аудит присланной базы перед персонализацией.

В ТЗ есть намёк «вдруг мы что-то перепутали». Он оправдан: если прогнать
такую базу через LLM не глядя, модель честно опишет сайт — но это будет
описание ЧУЖОЙ компании. В рассылке это выглядит так: письмо в компанию А
с комплиментом про завод компании Б. Одно такое письмо убивает всю кампанию.

Поэтому персонализации предшествует аудит. Он трёхслойный, от дешёвого
к дорогому — сеть дёргаем только там, где без неё нельзя:

  слой 1, без сети     дубли доменов, email-домен ≠ сайт, форма адреса,
                       бесплатные ящики на «корпоративной» строке, MX;
  слой 2, без сети     сходство названия компании и домена. Кириллицу
                       транслитерируем и обе стороны сводим к общему
                       фонетическому виду, иначе «Ункомтех» и «uncomtech»
                       не совпадут никогда;
  слой 3, сеть         открываем сайт и читаем, как компания называет себя
                       сама (title, og:site_name, копирайт в подвале).
                       Это единственный слой, который отличает настоящее
                       совпадение от случайного созвучия.

Слой 3 обязателен именно из-за ложных срабатываний слоя 2: «Rogen
Technologies» и «technosphera.ru» дают высокое сходство на общем куске
«techno», хотя это разные компании.

Запуск:
    python src/audit_base.py --in data/sample_15.csv --out data/sample_15_audited.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parent))
from email_validate import check_email  # noqa: E402
from site_scrape import SiteScraper  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}

# Юридические формы и дежурные слова, которые есть у всех и только шумят.
LEGAL_NOISE = re.compile(
    r"^(ооо|оао|зао|пао|ао|ип|тд|торговый дом|группа компаний|гк|"
    r"co|ltd|llc|inc|corp|company|group|holding|trading|industry|"
    r"industrial|technology|technologies|machinery|equipment|precision|"
    r"intelligent|science|heavy|cnc|tool|tools)$",
    re.IGNORECASE,
)


def translit(s: str) -> str:
    return "".join(TRANSLIT.get(ch, ch) for ch in s.lower())


def fold(s: str) -> str:
    """
    Сводит латиницу и транслит к общему виду.

    Один звук в русском и английском пишется по-разному: Техносфера ->
    tehnosfera, но домен technosphera.ru. Без сведения «ch»->«h» и «ph»->«f»
    строки не совпадут. Порядок замен важен: диграфы до одиночных букв.
    """
    s = translit(s)
    for a, b in (("ch", "h"), ("kh", "h"), ("ph", "f"), ("ck", "k")):
        s = s.replace(a, b)
    s = s.replace("c", "k").replace("y", "i")
    return re.sub(r"[^a-z0-9]", "", s)


def name_core(company: str) -> str:
    """Название без юрформы и родовых слов — только различающая часть."""
    tokens = [
        t for t in re.split(r"[\s,.\"«»'`-]+", company.strip())
        if t and not LEGAL_NOISE.match(t)
    ]
    return fold(" ".join(tokens) or company)


def domain_core(domain: str) -> str:
    """Второй уровень домена без www и зоны."""
    d = re.sub(r"^https?://", "", (domain or "").strip().lower())
    d = re.sub(r"^www\.", "", d).split("/")[0]
    return fold(d.split(".")[0])


def coverage(company: str, domain: str) -> float:
    """
    Какая доля домена объясняется словами из названия компании.

    Нужна там, где сравнение целиком проваливается из-за лишних слов.
    «JAT Cemented Carbide» против «jat-carbide.com» даёт ratio всего 71,
    потому что «cemented» в домен не попало. Но домен покрыт словами
    «jat» и «carbide» на 100% — значит он всё-таки этой компании.

    Учитываем только слова от 3 букв: короткие совпадают случайно.
    """
    dc = domain_core(domain)
    if not dc:
        return 0.0
    tokens = [
        t for t in re.split(r"[\s,.\"«»'`-]+", company.strip())
        if len(fold(t)) >= 3
    ]
    hit = sum(len(fold(t)) for t in tokens if fold(t) and fold(t) in dc)
    return min(100.0, hit / len(dc) * 100)


def similarity(company: str, domain: str) -> float:
    """
    0..100. Берём лучшее из трёх взглядов: полное название, «ядро»
    без юрформы, и покрытие домена словами названия.

    Именно ratio, а не partial_ratio: partial даёт 100 за любое общее
    подслово и склеивает «Rogen Technologies» с «technosphera.ru».
    """
    dc = domain_core(domain)
    if not dc:
        return 0.0
    return max(
        fuzz.ratio(fold(company), dc),
        fuzz.ratio(name_core(company), dc),
        coverage(company, domain),
    )


@dataclass
class AuditRow:
    n: int
    company: str
    email: str
    website: str
    issues: list[str] = field(default_factory=list)
    status: str = "ok"
    self_declared_name: str = ""     # как сайт называет себя сам
    corrected_website: str = ""
    confidence: str = ""
    evidence: str = ""
    name_domain_score: float = 0.0


MATCH_THRESHOLD = 62.0   # ниже — считаем, что название и домен не бьются

# Порог для перестановки домена между строками. Он намеренно выше порога
# простого несовпадения: пометить строку «проверь» дёшево, а предложить
# конкретную замену — это утверждение, и ошибиться в нём дорого.
#
# Значение откалибровано прогоном аудитора по собственной собранной базе
# (119 компаний, заведомо не перепутанных). При 75 туда попадало
# «TECHNO GROUP» -> kpd-techno.ru со скором 77 — совпадение только по
# общему куску «techno». Настоящие перестановки в исходной базе дают 100,
# так что 85 отсекает шум и не теряет ни одной реальной находки.
CROSS_MATCH_MIN = 85.0
CROSS_MATCH_GAP = 20.0   # насколько чужой домен должен обойти свой


def audit(
    rows: list[dict], check_live: bool = True, verbose: bool = True
) -> list[AuditRow]:
    out = [
        AuditRow(
            n=i,
            company=(r.get("company") or "").strip(),
            email=(r.get("email") or "").strip(),
            website=(r.get("website") or "").strip(),
        )
        for i, r in enumerate(rows, 1)
    ]

    # --- слой 1: дешёвые детерминированные проверки ---
    dup = Counter(r.website.lower() for r in out if r.website)

    for r in out:
        chk = check_email(r.email)
        if chk.status == "invalid":
            r.issues.append(f"почта: {chk.reason}")
        elif chk.is_free:
            r.issues.append(
                "бесплатный ящик при заявленном корпоративном сайте"
            )

        edom = r.email.split("@")[-1].lower()
        wdom = re.sub(r"^www\.", "", domain_host(r.website))
        if edom and wdom and edom != wdom:
            r.issues.append(f"домен почты ({edom}) не совпадает с сайтом ({wdom})")

        if r.website and dup[r.website.lower()] > 1:
            others = [o.company for o in out if o.website.lower() == r.website.lower()
                      and o.n != r.n]
            r.issues.append(
                f"тот же сайт указан у другой компании: {', '.join(others)}"
            )

    # --- слой 2: сходство названия и домена, включая перекрёстное ---
    for r in out:
        r.name_domain_score = similarity(r.company, r.website)
        if r.website and r.name_domain_score < MATCH_THRESHOLD:
            r.issues.append(
                f"название не бьётся с доменом (сходство {r.name_domain_score:.0f})"
            )
            # Ищем, не лежит ли «родной» домен этой компании в другой строке.
            # Смотрим и на сайт чужой строки, и на домен её почты: при
            # сдвиге столбцов родной домен компании всплывает то там, то там.
            candidates: list[tuple[float, str, str]] = []
            for o in out:
                if o.n == r.n:
                    continue
                for dom, where in (
                    (o.website, f"сайт строки {o.n}"),
                    (o.email.split("@")[-1], f"домен почты строки {o.n}"),
                ):
                    if dom:
                        candidates.append((similarity(r.company, dom), dom, where))

            if candidates:
                score, dom, where = max(candidates, key=lambda t: t[0])
                if (score >= CROSS_MATCH_MIN
                        and score > r.name_domain_score + CROSS_MATCH_GAP):
                    r.corrected_website = dom
                    r.evidence = (
                        f"«{r.company}» совпадает с {dom} на {score:.0f} — "
                        f"этот домен лежит как {where}"
                    )

    # --- слой 3: что сайт говорит о себе сам ---
    if check_live:
        scraper = SiteScraper(max_pages=1)
        for r in out:
            if not r.website:
                continue
            if verbose:
                print(f"  [{r.n:>2}] открываю {r.website:<28}", end="", flush=True)
            data = scraper.scrape(r.website)
            if not data.reachable:
                r.issues.append("сайт не открылся — проверить вручную")
                if verbose:
                    print("— недоступен")
                continue
            r.self_declared_name = data.title[:120]
            live = fuzz.ratio(fold(r.company), fold(data.title))
            live_core = fuzz.partial_ratio(name_core(r.company), fold(data.title))
            # partial здесь уместен: title длинный и содержит слоган,
            # название компании — лишь его часть.
            if max(live, live_core) < 55:
                r.issues.append(
                    f"сайт представляется иначе: «{data.title[:60]}»"
                )
            if verbose:
                print(f"— {data.title[:45]}")

    # --- вердикт ---
    for r in out:
        if not r.issues:
            r.status, r.confidence = "ok", "высокая"
        elif r.corrected_website:
            r.status, r.confidence = "перепутан сайт", "высокая"
        elif any("не открылся" in i for i in r.issues):
            r.status, r.confidence = "не проверено", "низкая"
        else:
            r.status, r.confidence = "требует проверки", "средняя"
    return out


def domain_host(url: str) -> str:
    d = re.sub(r"^https?://", "", (url or "").strip().lower())
    return d.split("/")[0]


def read_rows(path: Path) -> list[dict]:
    """
    Читает CSV с заголовком или без.

    Присланная база пришла без шапки — если требовать заголовок, скрипт
    молча потеряет первую компанию.
    """
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.reader(f) if any(c.strip() for c in r)]
    if not rows:
        return []

    head = [c.strip().lower() for c in rows[0]]
    aliases = {
        "company": ("company", "компания", "название", "name"),
        "email": ("email", "почта", "e-mail"),
        "website": ("website", "сайт", "site", "домен"),
    }
    idx = {
        key: next((i for i, h in enumerate(head) if h in names), None)
        for key, names in aliases.items()
    }

    # Шапка есть — читаем по именам колонок. Иначе порядок пришлось бы
    # угадывать, а он у разных выгрузок разный: у присланной базы это
    # компания-почта-сайт, у собранной мной — компания-сайт-почта.
    if idx["company"] is not None:
        body = rows[1:]
        return [
            {
                key: (r[i].strip() if i is not None and i < len(r) else "")
                for key, i in idx.items()
            }
            for r in body
        ]

    # Шапки нет — присланная база пришла именно так. Порядок позиционный.
    return [
        {"company": r[0], "email": r[1] if len(r) > 1 else "",
         "website": r[2] if len(r) > 2 else ""}
        for r in rows
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Аудит базы перед персонализацией")
    p.add_argument("--in", dest="src", type=Path, required=True)
    p.add_argument("--out", dest="dst", type=Path, required=True)
    p.add_argument("--no-live", action="store_true", help="без обращения к сайтам")
    args = p.parse_args()

    rows = read_rows(args.src)
    print(f"строк на входе: {len(rows)}\n")
    result = audit(rows, check_live=not args.no_live)

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    with args.dst.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "#", "Компания", "Email", "Сайт", "Статус", "Найденные проблемы",
            "Сходство имя/домен", "Как сайт называет себя",
            "Предлагаемый сайт", "Уверенность", "Обоснование",
        ])
        for r in result:
            w.writerow([
                r.n, r.company, r.email, r.website, r.status,
                "; ".join(r.issues), f"{r.name_domain_score:.0f}",
                r.self_declared_name, r.corrected_website, r.confidence,
                r.evidence,
            ])

    bad = [r for r in result if r.status != "ok"]
    print(f"\n{'='*70}\nитог: проблемных строк {len(bad)} из {len(result)}")
    for r in bad:
        print(f"  [{r.n:>2}] {r.company:<24} {r.status}")
        for i in r.issues:
            print(f"        - {i}")
        if r.corrected_website:
            print(f"        => предлагаю сайт: {r.corrected_website} ({r.evidence})")
    print(f"\nсохранено: {args.dst}")


if __name__ == "__main__":
    main()

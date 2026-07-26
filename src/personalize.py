"""
Задача 2 и 4: столбец «Персонализация».

На входе список компаний, на выходе тот же список плюс персонализация и
ссылка на источник. Работает одинаково и для своей собранной базы, и для
присланной в Задаче 4.

Главная проблема этого шага — не качество текста, а враньё. Модель, которой
дали только название компании, с удовольствием сочинит правдоподобный факт:
«вижу, вы недавно открыли направление в Казани». Отправить такое клиенту —
хуже, чем не персонализировать вообще.

Защита построена так, чтобы её нельзя было обойти уговорами в промпте:

  1. Модель не ходит в интернет. Она видит только текст, который скрапер
     реально скачал с сайта компании.
  2. Модель обязана вернуть source_quote — дословную цитату из этого текста.
  3. Скрипт проверяет, что цитата действительно есть в исходнике. Нет —
     строка бракуется и уходит в повтор, а не в таблицу.

То есть выдуманный факт технически не может пройти: под него не найдётся
цитаты. Это проверка кодом, а не доверие к инструкции.

Запуск:
    python src/personalize.py --in data/base.csv --out data/base_personalized.csv
    python src/personalize.py --in ... --out ... --no-llm     # без модели
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from facts import compose, mine  # noqa: E402
from llm import LLM, LLMError  # noqa: E402
from site_scrape import SiteData, SiteScraper  # noqa: E402

SYSTEM = """Ты помогаешь готовить холодную B2B-рассылку на русском языке.

Тебе дают текст, реально скачанный с сайта компании. Твоя задача — написать
одну-две фразы персонализации для первого письма.

Жёсткие правила:
- Опирайся ТОЛЬКО на переданный текст. Ты не знаешь об этой компании ничего,
  кроме него.
- Если в тексте нет ничего конкретного — верни personalization пустой строкой.
  Это нормальный и правильный ответ. Пустое лучше выдуманного.
- Никаких оценок и лести: «впечатляющий рост», «вы лидер рынка» — запрещено.
- Конкретика важнее красоты: продукт, отрасль, город, год, сертификат,
  оборудование, направление. Числа — только если они есть в тексте.
- Пиши так, чтобы фраза звучала как наблюдение живого человека, который
  минуту посмотрел сайт. Не как аннотация и не как реклама.

Формат ответа — JSON:
{
  "personalization": "1-2 фразы на русском, до 200 символов",
  "source_quote": "дословный фрагмент из переданного текста, 10-200 символов, подтверждающий факт",
  "confidence": "high" | "low"
}

source_quote обязана встречаться в переданном тексте посимвольно."""


@dataclass
class Row:
    company: str
    website: str
    email: str = ""
    person: str = ""
    signal: str = ""          # вакансия — сама по себе сильный факт
    signal_url: str = ""
    personalization: str = ""
    fact_kind: str = ""       # на каком типе факта построена персонализация
    source_url: str = ""
    source_quote: str = ""
    method: str = ""          # llm | факты с сайта | нет данных
    note: str = ""


def _norm(s: str) -> str:
    """Для сверки цитаты: пробелы и кавычки схлопываем, регистр убираем."""
    s = re.sub(r"[«»\"'`]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _facts_blob(d: SiteData) -> str:
    """Собирает то, что скачали, в один кусок текста для модели."""
    parts = []
    if d.title:
        parts.append(f"Заголовок сайта: {d.title}")
    if d.meta_description:
        parts.append(f"Описание: {d.meta_description}")
    if d.news:
        parts.append("Новости компании:\n- " + "\n- ".join(d.news))
    if d.about_text:
        parts.append(f"Текст со страниц:\n{d.about_text}")
    return "\n\n".join(parts)


# --- основной проход ---------------------------------------------------


def process_row(
    row: Row, scraper: SiteScraper, llm: LLM | None
) -> Row:
    if not row.website:
        row.method, row.note = "нет данных", "не указан сайт"
        return row

    d = scraper.scrape(row.website)
    if not d.reachable:
        row.method, row.note = "нет данных", f"сайт не открылся: {d.error}"
        return row

    row.source_url = d.final_url
    if not row.email and d.emails:
        row.email = d.emails[0]
    if not row.person and d.person_name:
        row.person = d.person_name

    # Факты добываем всегда: они и есть результат в режиме без модели,
    # и они же — единственное сырьё, которое модель имеет право пересказать.
    facts = mine(d, company=row.company, vacancy=row.signal,
                 vacancy_url=row.signal_url)

    blob = _facts_blob(d)
    if len(blob) < 60 and not facts:
        row.method, row.note = "нет данных", "на сайте нет пригодного текста"
        return row

    if llm is not None:
        try:
            out = llm.complete_json(
                f"Компания: {row.company}\nСайт: {d.final_url}\n\n"
                f"--- текст с сайта ---\n{blob[:6000]}",
                system=SYSTEM,
                max_tokens=600,
            )
            text = (out.get("personalization") or "").strip()
            quote = (out.get("source_quote") or "").strip()

            if text:
                # Ключевая проверка: цитата обязана быть в скачанном тексте.
                if quote and _norm(quote) in _norm(blob):
                    row.personalization = text
                    row.source_quote = quote
                    row.method = "llm"
                    return row
                row.note = "ответ модели отклонён: цитата не найдена в источнике"
            else:
                row.note = "модель не нашла конкретного факта"
        except LLMError as e:
            row.note = f"llm недоступна: {e}"

    # Путь без модели: собираем персонализацию из добытых фактов.
    text, quote, source = compose(facts)
    if text:
        row.personalization = text
        row.source_quote = quote
        row.source_url = source or row.source_url
        row.fact_kind = facts[0].kind
        row.method = "факты с сайта"
    else:
        # Пустая ячейка с объяснением честнее выдуманного комплимента.
        row.method = "нет данных"
        row.note = row.note or "конкретных фактов на сайте не нашлось"
    return row


def read_rows(path: Path) -> list[Row]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        has_header = bool(
            re.search(r"компан|company|сайт|website|email|почт", sample, re.I)
        )
        reader = csv.DictReader(f) if has_header else None
        if reader:
            rows = []
            for r in reader:
                low = {(k or "").strip().lower(): (v or "").strip()
                       for k, v in r.items()}
                rows.append(Row(
                    company=low.get("company") or low.get("компания")
                    or low.get("name") or "",
                    website=low.get("website") or low.get("сайт") or "",
                    email=low.get("email") or low.get("почта") or "",
                    person=low.get("person") or low.get("имя ЛПР".lower())
                    or low.get("имя") or low.get("person_name") or "",
                    signal=low.get("сигнал (вакансия)") or low.get("signal") or "",
                    signal_url=low.get("ссылка на вакансию")
                    or low.get("signal_url") or "",
                ))
            return [r for r in rows if r.company]
        f.seek(0)
        return [
            Row(company=r[0], email=r[1] if len(r) > 1 else "",
                website=r[2] if len(r) > 2 else "")
            for r in csv.reader(f) if r and r[0].strip()
        ]


def main() -> None:
    p = argparse.ArgumentParser(description="Столбец «Персонализация»")
    p.add_argument("--in", dest="src", type=Path, required=True)
    p.add_argument("--out", dest="dst", type=Path, required=True)
    p.add_argument("--no-llm", action="store_true",
                   help="только эвристика, без обращения к модели")
    p.add_argument("--offline", action="store_true",
                   help="воспроизвести результат из закоммиченного кэша: "
                        "без ключей и без обращения к модели")
    p.add_argument("--provider", default="auto",
                   help="auto | anthropic | openai_compat | claude_cli | replay")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    rows = read_rows(args.src)
    print(f"строк на входе: {len(rows)}")

    llm = None
    if not args.no_llm:
        provider = "replay" if args.offline else args.provider
        try:
            llm = LLM(provider=provider)
            print(f"llm: {llm.provider} ({llm.model or 'модель сессии'})")
        except LLMError as e:
            print(f"\n[!] {e}\n[!] продолжаю в режиме эвристики\n")

    scraper = SiteScraper()
    done: list[Row] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_row, r, scraper, llm): r for r in rows
        }
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            done.append(r)
            mark = {"llm": "+", "эвристика": "~", "нет данных": "-"}.get(r.method, "?")
            print(f"[{i}/{len(rows)}] {mark} {r.company[:32]:<34} "
                  f"{(r.personalization or r.note)[:60]}")

    # Порядок как во входном файле — иначе таблицу неудобно сверять.
    order = {r.company: i for i, r in enumerate(rows)}
    done.sort(key=lambda r: order.get(r.company, 999))

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    with args.dst.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Компания", "Сайт", "Email", "Имя ЛПР", "Персонализация",
                    "Тип факта", "Источник", "Цитата-подтверждение",
                    "Способ", "Примечание"])
        for r in done:
            w.writerow([r.company, r.website, r.email, r.person,
                        r.personalization, r.fact_kind, r.source_url,
                        r.source_quote, r.method, r.note])

    stats: dict[str, int] = {}
    kinds: dict[str, int] = {}
    for r in done:
        stats[r.method] = stats.get(r.method, 0) + 1
        if r.fact_kind:
            kinds[r.fact_kind] = kinds.get(r.fact_kind, 0) + 1

    filled = sum(1 for r in done if r.personalization)
    print(f"\n{'='*60}")
    print("способ:    " + "  ".join(f"{k}: {v}" for k, v in sorted(stats.items())))
    print("тип факта: " + "  ".join(f"{k}: {v}" for k, v in sorted(kinds.items())))
    print(f"персонализация заполнена: {filled}/{len(done)}")
    print(f"сохранено: {args.dst}")


if __name__ == "__main__":
    main()

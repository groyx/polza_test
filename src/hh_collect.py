"""
сбор базы, шаг 1: сбор компаний с сигналом спроса.

Логика выбора источника. Продукт агентства — холодные рассылки, которые приводят
входящие заявки. Значит идеальный клиент это не «любая B2B-компания», а та,
которая прямо сейчас доказала деньгами, что ей нужны лиды. Самое честное
доказательство — открытая вакансия менеджера по продажам: компания готова
платить человеку 60-150 тыс/мес за поиск клиентов. Это уже подтверждённый
бюджет на привлечение и открытая боль.

Отсюда весь заход: hh.ru отдаёт список таких компаний бесплатно и постоянно
обновляет его. Вакансия заодно становится крючком для первого письма — это
публичный факт, не выдуманный и легко проверяемый.

Почему HTML, а не api.hh.ru: с 2025 публичный API закрыли, /vacancies отдаёт
403 без OAuth-приложения даже с подменой TLS-отпечатка (проверено). Поисковая
выдача при этом открыта. curl_cffi нужен именно для отпечатка: обычный
requests с честным User-Agent ловит 403.

На выходе: data/stage1_companies.csv — компания, ссылка на hh, сайт, сигнал.
Почту и ЛПР добирает site_scrape.py со своего сайта компании.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote, urlparse

from bs4 import BeautifulSoup
from curl_cffi import requests

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data"

# Запросы подобраны так, чтобы ловить именно B2B-продажи, а не розницу
# и не колл-центры. «Продавец-консультант» и «оператор» сюда не попадают.
#
# Два пресета. `industrial` — основной: база из Задачи 4 целиком состоит из
# станкостроения, металлообработки и твердосплавного инструмента, значит
# Агентство сейчас работает по промышленному оборудованию. База под них должна
# быть из той же вертикали, а не «случайные российские ООО».
QUERY_PRESETS = {
    "industrial": [
        "менеджер по продажам оборудования",
        "менеджер по продажам металлопроката",
        "менеджер по продажам станков",
        "менеджер по продажам промышленного оборудования",
        "инженер по продажам",
        "менеджер по проектным продажам",
        "менеджер по продажам инструмента",
        "менеджер по продажам комплектующих",
    ],
    "general": [
        "менеджер по продажам b2b",
        "менеджер по работе с корпоративными клиентами",
        "руководитель отдела продаж",
        "менеджер активных продаж",
        "специалист по развитию бизнеса",
        "менеджер по продажам оборудования",
    ],
}
DEFAULT_QUERIES = QUERY_PRESETS["industrial"]

# Кадровые агентства и джоб-борды: они публикуют вакансии за клиента,
# поэтому компания в карточке — не тот, кому мы продаём.
EMPLOYER_BLACKLIST = re.compile(
    r"headhunter|hh\.ru|работа\.ру|superjob|avito|kelly|ancor|анкор|"
    r"кадров|рекрут|recruit|staffing|аутстаффинг|аутсорсинг персонала|"
    r"агентство занятости|подбор персонала",
    re.IGNORECASE,
)

# Явная розница/B2C: им холодный B2B-аутрич не продать.
B2C_HINTS = re.compile(
    r"пятёрочка|магнит|днс|мвидео|эльдорадо|вкусвилл|озон банк|"
    r"розничн|салон красоты|фитнес-клуб|доставка еды",
    re.IGNORECASE,
)


@dataclass
class Company:
    employer_id: str
    name: str
    hh_url: str
    website: str = ""
    region: str = ""
    signal_vacancy: str = ""      # какая вакансия выдала сигнал
    signal_vacancy_url: str = ""
    open_vacancies: str = ""


class HH:
    """Тонкий клиент к hh.ru с вежливыми паузами."""

    def __init__(self, delay: tuple[float, float] = (0.7, 1.6)):
        self.session = requests.Session(impersonate="chrome110")
        self.delay = delay

    def _get(self, url: str) -> str | None:
        try:
            r = self.session.get(url, timeout=30)
        except Exception as e:
            print(f"    ! сеть: {type(e).__name__}", file=sys.stderr)
            return None
        # Пауза после каждого запроса, а не до: не тормозим первый вызов.
        time.sleep(random.uniform(*self.delay))
        if r.status_code != 200:
            print(f"    ! http {r.status_code} на {url[:70]}", file=sys.stderr)
            return None
        return r.text

    def search(self, query: str, pages: int = 2) -> list[Company]:
        """Компании из поисковой выдачи по одному запросу."""
        found: list[Company] = []
        for page in range(pages):
            url = (
                "https://hh.ru/search/vacancy"
                f"?text={quote(query)}&area=113&items_on_page=50&page={page}"
            )
            html = self._get(url)
            if not html:
                break

            soup = BeautifulSoup(html, "lxml")
            cards = soup.select('[data-qa="vacancy-serp__vacancy"]') or soup.select(
                ".serp-item"
            )
            if not cards:
                # Разметка hh периодически меняется. Не молчим — падение
                # числа карточек до нуля должно быть заметно сразу.
                print(f"    ! карточек не найдено, стр. {page}", file=sys.stderr)
                break

            for card in cards:
                emp = card.select_one('a[data-qa="vacancy-serp__vacancy-employer"]')
                vac = card.select_one('a[data-qa="serp-item__title"]')
                if not emp:
                    continue
                href = emp.get("href", "")
                m = re.search(r"/employer/(\d+)", href)
                if not m:
                    continue
                found.append(
                    Company(
                        employer_id=m.group(1),
                        # hh склеивает форму собственности с названием без
                        # пробела: "ОООСимаКей". Разделяем.
                        name=re.sub(
                            r"^(ООО|АО|ЗАО|ПАО|ИП|ОАО|НАО)(?=[А-ЯA-Z])",
                            r"\1 ",
                            emp.get_text(strip=True),
                        ),
                        hh_url=f"https://hh.ru/employer/{m.group(1)}",
                        signal_vacancy=vac.get_text(strip=True) if vac else "",
                        signal_vacancy_url=(
                            vac.get("href", "").split("?")[0] if vac else ""
                        ),
                    )
                )
        return found

    def enrich(self, c: Company) -> Company:
        """Со страницы работодателя добираем сайт и регион."""
        html = self._get(c.hh_url)
        if not html:
            return c
        soup = BeautifulSoup(html, "lxml")

        site = soup.select_one('a[data-qa="sidebar-company-site"]')
        if site:
            href = (site.get("href") or "").strip()
            # На своей же странице hh иногда подставляет hh.ru — это не сайт.
            if href and "hh.ru" not in urlparse(href).netloc:
                c.website = href

        area = soup.select_one('[data-qa="sidebar-company-address"]')
        if area:
            c.region = area.get_text(" ", strip=True)[:80]

        vac = soup.select_one('[data-qa="employer-page__employer-vacancies-link"]')
        if vac:
            digits = re.search(r"\d+", vac.get_text())
            if digits:
                c.open_vacancies = digits.group(0)
        return c


def collect(queries: list[str], pages: int, limit: int) -> list[Company]:
    hh = HH()

    # --- поиск ---
    raw: list[Company] = []
    for q in queries:
        print(f"[поиск] {q}")
        got = hh.search(q, pages=pages)
        print(f"    найдено карточек: {len(got)}")
        raw.extend(got)

    # --- дедуп и фильтры ---
    seen: set[str] = set()
    kept: list[Company] = []
    dropped = {"дубль": 0, "агентство": 0, "b2c": 0}
    for c in raw:
        if c.employer_id in seen:
            dropped["дубль"] += 1
            continue
        seen.add(c.employer_id)
        if EMPLOYER_BLACKLIST.search(c.name):
            dropped["агентство"] += 1
            continue
        if B2C_HINTS.search(c.name):
            dropped["b2c"] += 1
            continue
        kept.append(c)

    print(
        f"\n[фильтр] уникальных: {len(kept)}  "
        + "  ".join(f"отсеяно {k}: {v}" for k, v in dropped.items())
    )

    # --- добор сайта ---
    # Берём с запасом: часть компаний сайт на hh не указывает, они отвалятся.
    result: list[Company] = []
    for i, c in enumerate(kept, 1):
        if len(result) >= limit:
            break
        print(f"[{i}/{len(kept)}] {c.name[:40]:<42}", end="")
        c = hh.enrich(c)
        if not c.website:
            print("— сайт не указан, пропуск")
            continue
        print(f"-> {c.website[:45]}")
        result.append(c)

    return result


def save(rows: list[Company], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        # utf-8-sig: без BOM Excel открывает кириллицу кракозябрами
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        w.writerows(asdict(r) for r in rows)
    print(f"\nсохранено: {path}  строк: {len(rows)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Сбор компаний с сигналом найма продажников")
    p.add_argument("--pages", type=int, default=2, help="страниц выдачи на запрос")
    p.add_argument("--limit", type=int, default=80, help="сколько компаний с сайтом добрать")
    p.add_argument("--preset", choices=sorted(QUERY_PRESETS), default="industrial",
                   help="набор поисковых запросов (по умолчанию промышленный)")
    p.add_argument("--out", type=Path, default=OUT_DIR / "stage1_companies.csv")
    args = p.parse_args()

    print(f"пресет запросов: {args.preset}\n")
    rows = collect(QUERY_PRESETS[args.preset], args.pages, args.limit)
    if not rows:
        print("Ничего не собрано — вероятно, изменилась разметка hh.ru", file=sys.stderr)
        sys.exit(1)
    save(rows, args.out)


if __name__ == "__main__":
    main()

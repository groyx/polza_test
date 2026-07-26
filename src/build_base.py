"""
Задача 1, шаг 3: превращает сырую выгрузку с hh.ru в готовую базу.

Между «список компаний» и «база для рассылки» лежит фильтрация, и она
здесь важнее сбора. Сырьё с hh.ru имеет два системных дефекта, оба
всплыли на первом же прогоне на 120 компаниях:

  1. hh часто указывает не сайт компании, а её карьерный портал:
     rabota.cdek.ru, job.megafon.ru, team.rencredit.ru. Писать по такому
     домену бессмысленно, а персонализация по нему опишет отдел кадров.
     Сводим к основному домену.

  2. В выдачу лезут Роснефть, Почта России, МегаФон. Формально они
     нанимают продажников, но клиентом аутрич-агентства не станут:
     у них свой маркетинг и тендерные закупки. Отсекаем.

Дальше по каждой компании: обход сайта, поиск почты и ЛПР, проверка
адреса. Строки без валидной почты в финальную базу не идут — база
из 50 строк, где половина отскочит, хуже базы из 50 живых.

Запуск:
    python src/build_base.py --in data/stage1_companies.csv --out data/base_50.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from egrul import Egrul  # noqa: E402
from email_validate import check_email  # noqa: E402
from site_scrape import SiteScraper  # noqa: E402

# Поддомены, за которыми живёт HR, а не компания.
HR_SUBDOMAIN = re.compile(
    r"^(rabota|job|jobs|career|careers|team|hr|vacancy|vacancies|work)\.",
    re.IGNORECASE,
)

# Составные зоны: у них регистрируемый домен — третьего уровня.
MULTI_TLD = (".com.cn", ".co.uk", ".com.tr", ".com.br", ".co.jp", ".org.ru")

# Компании, которым холодный аутрич не продать: корпорации со своим
# маркетингом, госструктуры, крупный B2C, а также сами hh и джоб-борды.
NOT_ICP = re.compile(
    r"роснефть|газпром|лукойл|почта россии|мегафон|мтс\b|билайн|теле2|"
    r"яндекс|озон|ozon|wildberries|вайлдберриз|авито|avito|вкусвилл|"
    r"магнит|пятёрочка|перекрёсток|x5|лента|ашан|леруа|лемана|"
    r"сдэк|cdek|деловые линии|пэк|боксберри|"
    r"ржд|аэрофлот|росатом|ростех|ростелеком|"
    r"headhunter|hh\.ru|superjob|работа\.ру|"
    # Банки и страховые: у них свои отделы маркетинга, закупки идут
    # через тендер, и продавать им рассылку «в лоб» бессмысленно.
    r"\bбанк\b|банка|банков|\bbank\b|сбер|втб|тинькофф|т-банк|райффайзен|"
    r"альфа|совкомбанк|росбанк|открытие|точка|"
    r"страхован|росгосстрах|ингосстрах|согаз|альфастрахование|"
    r"капитал лайф|ренессанс",
    re.IGNORECASE,
)


def normalize_site(url: str) -> str:
    """
    Приводит ссылку с hh.ru к основному домену компании.

    rabota.cdek.ru/         -> cdek.ru
    http://job.megafon.ru   -> megafon.ru
    https://hnrus.com/career/ -> hnrus.com
    """
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    p = urlparse(url)
    host = p.netloc.lower().split(":")[0]
    host = re.sub(r"^www\.", "", host)

    if HR_SUBDOMAIN.match(host):
        host = host.split(".", 1)[1]

    # Путь отбрасываем всегда: с hh приходят ссылки вида /career/ и /vacancy/,
    # а нам нужен корень сайта, откуда скрапер сам найдёт контакты.
    return f"https://{host}"


def registrable(host: str) -> str:
    host = re.sub(r"^https?://", "", host).split("/")[0].lower()
    host = re.sub(r"^www\.", "", host)
    for tld in MULTI_TLD:
        if host.endswith(tld):
            parts = host[: -len(tld)].split(".")
            return parts[-1] + tld
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def pick_email(emails: list[str], site: str) -> tuple[str, str]:
    """
    Выбирает один адрес из найденных на сайте.

    Приоритет — почта на домене компании: адрес на mail.ru при живом
    корпоративном сайте обычно принадлежит не той компании (подрядчик,
    форма обратной связи, дилер). Внутри домена предпочитаем осмысленные
    ящики продаж, а не noreply и не hr.
    """
    if not emails:
        return "", "почта на сайте не найдена"

    site_dom = registrable(site)
    own = [e for e in emails if registrable(e.split("@")[-1]) == site_dom]
    pool = own or emails
    # Пометку считаем один раз и возвращаем со всеми ветками: раньше она
    # терялась, если адрес попадал в приоритетный префикс, и чужой домен
    # уезжал в базу без предупреждения.
    note = "" if own else "адрес не на домене сайта"

    priority = ("sales", "info", "zakaz", "order", "office", "mail", "contact")
    for pref in priority:
        for e in pool:
            if e.split("@")[0].startswith(pref):
                return e, note
    # Ящики отдела кадров нам бесполезны — берём их только если больше нечего.
    non_hr = [e for e in pool if not re.match(r"^(hr|job|vacan|resume|personal)", e)]
    return (non_hr or pool)[0], note


def process(row: dict, scraper: SiteScraper) -> dict | None:
    name = (row.get("name") or "").strip()
    site = normalize_site(row.get("website") or "")
    if not site:
        return None

    data = scraper.scrape(site)
    if not data.reachable:
        return {**row, "website": site, "email": "", "person": "",
                "email_status": "invalid", "note": "сайт не открылся"}

    email, note = pick_email(data.emails, site)
    chk = check_email(email) if email else None

    # ЛПР сначала ищем на сайте: там указана актуальная рабочая роль.
    # Кого не нашли — доберём из ЕГРЮЛ отдельной фазой (см. enrich_people).
    person, person_role, person_src, inn = (
        data.person_name, data.person_role,
        "сайт компании" if data.person_name else "", "",
    )

    return {
        "company": name,
        "website": data.final_url,
        "email": email,
        "email_status": chk.status if chk else "invalid",
        "email_note": (chk.reason if chk else "почта не найдена"),
        "person": person,
        "person_role": person_role,
        "person_source": person_src if person else "",
        "inn": inn,
        "signal": row.get("signal_vacancy", ""),
        "signal_url": row.get("signal_vacancy_url", ""),
        "hh_url": row.get("hh_url", ""),
        "title": data.title,
        "note": note,
        "_emails_found": " | ".join(data.emails[:5]),
    }


def enrich_people(rows: list[dict], verbose: bool = True) -> None:
    """
    Вторая фаза: добираем ФИО директора из ЕГРЮЛ для тех, у кого на сайте
    руководство не указано. Меняет rows на месте.

    Почему отдельной фазой и последовательно. Обход сайтов идёт в 8 потоков
    и это нормально: там 100 разных хостов. ЕГРЮЛ — один сервер ФНС, и при
    той же параллельности он начинает молча отдавать пустую выдачу. Ловится
    неприятно: скрипт не падает, просто половина имён теряется. На прогоне
    из 100 компаний так потерялось 9 из 16 найденных.

    Поэтому здесь один клиент, один поток и пауза между запросами.
    Медленнее, но данные не пропадают.
    """
    todo = [r for r in rows if not r.get("person")]
    if not todo:
        return
    if verbose:
        print(f"\n[ЕГРЮЛ] ищу руководителей для {len(todo)} компаний "
              f"(последовательно, чтобы ФНС не начала троттлить)")

    client = Egrul()
    found = ambiguous = 0
    for i, r in enumerate(todo, 1):
        res = client.by_name(r["company"], r.get("region", ""))
        if res.found:
            r["person"], r["person_role"] = res.fio, res.role
            r["person_source"], r["inn"] = "ЕГРЮЛ", res.inn
            found += 1
        elif res.ambiguous:
            r["note"] = (r.get("note", "") + "; " if r.get("note") else "") + res.note
            ambiguous += 1
        if verbose:
            mark = "+" if res.found else ("?" if res.ambiguous else "-")
            print(f"  [{i}/{len(todo)}] {mark} {r['company'][:30]:<32} "
                  f"{res.fio or res.note[:44]}")

    if verbose:
        print(f"[ЕГРЮЛ] найдено: {found}, неоднозначных (пропущено): {ambiguous}")


COLUMNS = [
    "Компания", "Сайт", "Email", "Статус почты", "Имя ЛПР", "Должность",
    "Источник имени", "ИНН", "Сигнал (вакансия)", "Ссылка на вакансию",
    "Страница на hh", "Заголовок сайта", "Примечание",
]
# Соответствие колонок файла и внутренних ключей — чтобы готовую базу
# можно было прочитать обратно и дозаполнить, не пересобирая с нуля.
COL_TO_KEY = {
    "Компания": "company", "Сайт": "website", "Email": "email",
    "Статус почты": "email_status", "Имя ЛПР": "person",
    "Должность": "person_role", "Источник имени": "person_source",
    "ИНН": "inn", "Сигнал (вакансия)": "signal",
    "Ссылка на вакансию": "signal_url", "Страница на hh": "hh_url",
    "Заголовок сайта": "title", "Примечание": "note",
}


def write_base(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([r.get(COL_TO_KEY[c], "") for c in COLUMNS])


def read_base(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [
            {COL_TO_KEY[k]: v for k, v in row.items() if k in COL_TO_KEY}
            for row in csv.DictReader(f)
        ]


def main() -> None:
    p = argparse.ArgumentParser(description="Сборка финальной базы из выгрузки hh.ru")
    p.add_argument("--in", dest="src", type=Path)
    p.add_argument("--out", dest="dst", type=Path)
    p.add_argument(
        "--enrich", type=Path,
        help="дозаполнить ЛПР в уже готовой базе, не пересобирая её заново",
    )
    p.add_argument("--target", type=int, default=55, help="сколько валидных строк нужно")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    # Режим дозаполнения: сайты уже обойдены, нужны только имена из ЕГРЮЛ.
    # Отдельный режим нужен потому, что ФНС ограничивает частоту запросов,
    # и прогон по именам приходится повторять, не трогая остальное.
    if args.enrich:
        rows = read_base(args.enrich)
        before = sum(1 for r in rows if r.get("person"))
        enrich_people(rows)
        after = sum(1 for r in rows if r.get("person"))
        write_base(rows, args.enrich)
        print(f"\nимён было {before}, стало {after} — обновлён {args.enrich}")
        return

    if not args.src or not args.dst:
        p.error("нужны --in и --out (либо --enrich для дозаполнения)")

    with args.src.open(encoding="utf-8-sig", newline="") as f:
        raw = list(csv.DictReader(f))
    print(f"на входе: {len(raw)}")

    # --- фильтр ICP ---
    kept, dropped_icp = [], []
    seen_domains: set[str] = set()
    for r in raw:
        if NOT_ICP.search(r.get("name", "")):
            dropped_icp.append(r["name"])
            continue
        dom = registrable(normalize_site(r.get("website", "")))
        if not dom or dom in seen_domains:
            continue          # дубль по домену: филиалы одной компании
        seen_domains.add(dom)
        kept.append(r)

    print(f"отсеяно как не-ICP (корпорации, банки, маркетплейсы): {len(dropped_icp)}")
    print(f"к обходу сайтов: {len(kept)}\n")

    scraper = SiteScraper()
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, r, scraper): r for r in kept}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if not res:
                continue
            rows.append(res)
            mark = {"valid": "+", "risky": "~"}.get(res.get("email_status"), "-")
            print(f"[{i}/{len(kept)}] {mark} {res.get('company','')[:30]:<32} "
                  f"{res.get('email','')[:32]:<34} {res.get('person','')[:24]}")

    good = [r for r in rows if r.get("email_status") in ("valid", "risky")]

    # ЕГРЮЛ дёргаем только для строк, которые реально попадут в базу:
    # запрашивать директора для компании без рабочей почты бессмысленно.
    enrich_people(good)

    good.sort(key=lambda r: (r["email_status"] != "valid", not r["person"]))

    for r in good:
        r["note"] = "; ".join(
            x for x in (r.get("note"), r.get("email_note")) if x
        )
    write_base(good, args.dst)

    valid = sum(1 for r in good if r["email_status"] == "valid")
    with_person = sum(1 for r in good if r["person"])
    print(f"\n{'='*66}")
    print(f"обработано сайтов : {len(rows)}")
    print(f"с рабочей почтой  : {len(good)}  (из них valid: {valid}, risky: {len(good)-valid})")
    print(f"с именем ЛПР      : {with_person}")
    print(f"сохранено         : {args.dst}")
    if len(good) < args.target:
        print(f"\n[!] нужно {args.target}, собрано {len(good)} — "
              f"добавьте страниц в hh_collect.py (--pages)")


if __name__ == "__main__":
    main()

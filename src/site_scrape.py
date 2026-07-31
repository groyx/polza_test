"""
сбор базы, шаг 2 + фактура для Задачи 2: обход сайта компании.

Достаём три вещи:
  1. почту       — из mailto:, из текста, из-под обфускации Cloudflare;
  2. ЛПР         — ФИО рядом со словом «директор» / «основатель»;
  3. фактуру     — описание компании и свежие новости, на которых потом
                   строится персонализация.

Принцип, который держит всю Задачу 2: персонализацию нельзя выдумывать.
Поэтому модуль возвращает только сырые цитаты со страниц и URL, откуда
они взяты. LLM на следующем шаге получает этот текст и физически не имеет
данных для фантазии — он их сжимает, а не сочиняет.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from curl_cffi import requests

# Пути, по которым на российских сайтах реально лежат контакты и «о компании».
# Порядок важен: сначала самое вероятное, обход прекращается по лимиту.
CONTACT_PATHS = [
    "/contacts", "/contacts/", "/kontakty", "/kontakty/", "/contact",
    "/about", "/about/", "/o-kompanii", "/o-kompanii/", "/company",
    "/o-nas", "/about-us", "/rukovodstvo", "/team", "/komanda",
]
NEWS_PATHS = ["/news", "/news/", "/novosti", "/novosti/", "/press", "/blog"]

EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", re.IGNORECASE
)

# Мусорные адреса, которые встречаются в шаблонах сайтов и в чужих скриптах.
EMAIL_JUNK = re.compile(
    r"(@(example|sentry|wixpress|domain|mail)\.|"
    r"\.(png|jpe?g|gif|svg|webp|css|js)$|"
    r"^(u003e|x[0-9a-f]{2}))",
    re.IGNORECASE,
)

# ФИО рядом с должностью. Ловим «Генеральный директор — Иванов Иван Иванович»
# и обратный порядок «Иванов И.И., генеральный директор».
#
# Флаг IGNORECASE здесь применять НЕЛЬЗЯ на весь шаблон: он обнуляет
# требование заглавной буквы в [А-ЯЁ], и тогда в ФИО попадает любой обрывок
# строчного текста. Поймано на живом сайте: из «...управляющего компьютера»
# извлекалось ФИО «го компьютера». Поэтому регистронезависимость включаем
# точечно, только на названии должности, через inline-флаг (?i:...).
ROLE_WORDS = (
    r"генеральн\w+ директор|исполнительн\w+ директор|коммерческ\w+ директор|"
    r"технически\w+ директор|финансов\w+ директор|директор по развитию|"
    r"управляющий партнёр|управляющий директор|основател\w+|владелец|"
    r"руководител\w+ отдела продаж|собственник|президент компании"
)
ROLE_CI = rf"(?i:{ROLE_WORDS})"
NAME_TOKEN = r"[А-ЯЁ][а-яё]{2,}"

FIO_AFTER_ROLE = re.compile(
    rf"{ROLE_CI}\s*[:\-–—]?\s*"
    rf"({NAME_TOKEN}\s+{NAME_TOKEN}(?:\s+{NAME_TOKEN})?)"
)
FIO_BEFORE_ROLE = re.compile(
    rf"({NAME_TOKEN}\s+{NAME_TOKEN}(?:\s+{NAME_TOKEN})?)\s*[,\-–—]\s*{ROLE_CI}"
)

# Отчество — самый надёжный признак, что это человек, а не название.
PATRONYMIC = re.compile(r"(ович|евич|ьич|овна|евна|ична|инична)$", re.IGNORECASE)
# Типичные окончания русских фамилий.
SURNAME = re.compile(
    r"(ов|ев|ёв|ин|ын|ский|цкий|ская|цкая|ова|ева|ёва|ина|ына|ко|ук|юк|ых|их)$",
    re.IGNORECASE,
)
# Слова, которые пишутся с заглавной, но человеком не являются.
NOT_A_NAME = re.compile(
    r"^(Москва|Россия|Санкт|Петербург|Екатеринбург|Компания|Общество|Групп\w*|"
    r"Холдинг|Завод|Центр|Торгов\w+|Дом|Премиум|Авто|Сервис|Систем\w*|"
    r"Проект|Строй\w*|Тех\w*|Отдел|Департамент|Управление|Наш\w*|Ваш\w*)$",
    re.IGNORECASE,
)


@dataclass
class SiteData:
    url: str
    reachable: bool = False
    final_url: str = ""
    title: str = ""
    meta_description: str = ""
    emails: list[str] = field(default_factory=list)
    person_name: str = ""
    person_role: str = ""
    person_source: str = ""
    about_text: str = ""       # сырой текст «о компании», до 1200 символов
    raw_text: str = ""         # больший объём текста — сырьё для добычи фактов
    headings: list[str] = field(default_factory=list)  # h1/h2 главной
    news: list[str] = field(default_factory=list)   # заголовки новостей
    pages_seen: list[str] = field(default_factory=list)
    error: str = ""


def _decode_cf_email(hex_str: str) -> str:
    """
    Cloudflare Email Obfuscation прячет адрес в data-cfemail.

    Формат: hex-строка, первый байт — ключ, остальные XOR-ятся с ним.
    Без этого на сайтах за Cloudflare почта просто не находится.
    """
    try:
        data = bytes.fromhex(hex_str)
        key = data[0]
        return "".join(chr(b ^ key) for b in data[1:])
    except (ValueError, IndexError):
        return ""


def _clean_emails(candidates: list[str]) -> list[str]:
    out, seen = [], set()
    for e in candidates:
        e = e.strip().strip(".,;:()<>\"'").lower()
        if not e or EMAIL_JUNK.search(e) or e in seen:
            continue
        # Отсекаем склейки вида "почтаinfo@site.ru", которые даёт
        # текстовый режим, когда в вёрстке нет пробела.
        m = EMAIL_RE.search(e)
        if not m or m.group(0) != e:
            continue
        seen.add(e)
        out.append(e)
    return out


def _extract_emails(soup: BeautifulSoup, raw_html: str) -> list[str]:
    found: list[str] = []

    for a in soup.select('a[href^="mailto:"]'):
        addr = a.get("href", "")[7:].split("?")[0]
        if addr:
            found.append(addr)

    for el in soup.select("[data-cfemail]"):
        decoded = _decode_cf_email(el.get("data-cfemail", ""))
        if decoded:
            found.append(decoded)

    # Текстом, а не по сырому HTML: в HTML ловятся адреса из
    # аналитических скриптов и структурированной разметки.
    found += EMAIL_RE.findall(soup.get_text(" ", strip=True))

    return _clean_emails(found)


def _looks_like_fio(name: str) -> bool:
    """
    Отсекает названия компаний, которые тоже пишутся с заглавных букв.

    Пропускаем, только если есть отчество (Иванов Иван Иванович) либо
    похожая на русскую фамилия (Скотников Виктор). Без этого в базу
    попадает «Авто Премиум Москва» в графе «генеральный директор» —
    поймано на живом сайте.
    """
    tokens = name.split()
    if not 2 <= len(tokens) <= 3:
        return False
    if any(NOT_A_NAME.match(t) for t in tokens):
        return False
    if len(tokens) == 3 and PATRONYMIC.search(tokens[2]):
        return True
    return any(SURNAME.search(t) for t in tokens)


def _extract_person(text: str) -> tuple[str, str]:
    """ФИО и должность ЛПР. Пустая строка, если уверенно не нашли."""
    for pattern in (FIO_AFTER_ROLE, FIO_BEFORE_ROLE):
        for m in pattern.finditer(text):
            name = " ".join(m.group(1).split())
            if not _looks_like_fio(name):
                continue
            role_m = re.search(ROLE_WORDS, m.group(0), re.IGNORECASE)
            return name, (role_m.group(0).lower() if role_m else "")
    return "", ""


class SiteScraper:
    def __init__(self, timeout: int = 20, max_pages: int = 5):
        self.session = requests.Session(impersonate="chrome110")
        self.timeout = timeout
        self.max_pages = max_pages

    def _fetch(self, url: str) -> tuple[str, str] | None:
        """Возвращает (html, final_url) либо None."""
        try:
            r = self.session.get(
                url, timeout=self.timeout, allow_redirects=True, verify=False
            )
        except Exception:
            return None
        if r.status_code != 200 or not r.text:
            return None
        return r.text, str(r.url)

    def scrape(self, website: str) -> SiteData:
        if not website.startswith(("http://", "https://")):
            website = "https://" + website
        data = SiteData(url=website)

        first = self._fetch(website)
        if first is None:
            # Часть российских сайтов живёт только на http или только на www.
            parsed = urlparse(website)
            for alt in (
                website.replace("https://", "http://"),
                f"{parsed.scheme}://www.{parsed.netloc}{parsed.path}",
            ):
                first = self._fetch(alt)
                if first:
                    break
        if first is None:
            data.error = "сайт не открылся"
            return data

        html, final = first
        data.reachable = True
        data.final_url = final
        soup = BeautifulSoup(html, "lxml")
        data.pages_seen.append(final)

        if soup.title and soup.title.string:
            data.title = soup.title.string.strip()[:200]
        md = soup.find("meta", attrs={"name": "description"})
        if md and md.get("content"):
            data.meta_description = md["content"].strip()[:400]

        emails = _extract_emails(soup, html)
        page_text = soup.get_text(" ", strip=True)
        name, role = _extract_person(page_text)
        if name:
            data.person_name, data.person_role, data.person_source = name, role, final

        # Заголовки главной — самый концентрированный источник конкретики
        # о продукте: там пишут «Листогибочные прессы до 400 тонн», а не
        # «мы динамично развивающаяся компания».
        for h in soup.select("h1, h2")[:12]:
            t = " ".join(h.get_text(" ", strip=True).split())
            if 8 <= len(t) <= 120:
                data.headings.append(t)

        about_chunks: list[str] = [page_text[:800]]
        raw_chunks: list[str] = [page_text[:4000]]

        # --- внутренние страницы ---
        base = f"{urlparse(final).scheme}://{urlparse(final).netloc}"
        visited = {final.rstrip("/")}

        for path in CONTACT_PATHS + NEWS_PATHS:
            if len(data.pages_seen) >= self.max_pages:
                break
            target = urljoin(base, path)
            if target.rstrip("/") in visited:
                continue
            got = self._fetch(target)
            if got is None:
                continue
            visited.add(target.rstrip("/"))
            sub_html, sub_url = got
            sub = BeautifulSoup(sub_html, "lxml")
            data.pages_seen.append(sub_url)

            emails += _extract_emails(sub, sub_html)
            sub_text = sub.get_text(" ", strip=True)
            raw_chunks.append(sub_text[:3000])

            if not data.person_name:
                name, role = _extract_person(sub_text)
                if name:
                    data.person_name = name
                    data.person_role = role
                    data.person_source = sub_url

            if path in NEWS_PATHS:
                # Заголовки новостей — лучший источник свежего повода.
                for h in sub.select("h1, h2, h3, .news-title, .news__title")[:8]:
                    t = h.get_text(" ", strip=True)
                    if 20 <= len(t) <= 200:
                        data.news.append(t)
            else:
                about_chunks.append(sub_text[:600])

        data.emails = _clean_emails(emails)
        data.about_text = " ".join(about_chunks)[:1200]
        data.raw_text = " ".join(raw_chunks)[:14000]
        data.headings = list(dict.fromkeys(data.headings))[:12]
        data.news = list(dict.fromkeys(data.news))[:5]
        return data


if __name__ == "__main__":
    targets = sys.argv[1:] or ["iskroline.ru", "kontur.ru"]
    sc = SiteScraper()
    for t in targets:
        d = sc.scrape(t)
        print(f"\n===== {t} =====")
        print(f"  доступен : {d.reachable}  {d.error}")
        print(f"  title    : {d.title[:90]}")
        print(f"  почты    : {d.emails[:5]}")
        print(f"  ЛПР      : {d.person_name or '—'} ({d.person_role or '—'})")
        print(f"  новости  : {d.news[:3]}")
        print(f"  страниц  : {len(d.pages_seen)}")

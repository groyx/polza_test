"""
Поиск ФИО руководителя в ЕГРЮЛ.

Зачем отдельный источник. Обход сайта находит ЛПР примерно у каждой
восьмой компании: большинство российских сайтов руководство не публикует.
А в ТЗ имя нужно для каждой строки.

ЕГРЮЛ — открытый государственный реестр, и у ФНС есть бесплатный поиск на
egrul.nalog.ru. Он отдаёт официальное название, ИНН и строку руководителя
вида «ДИРЕКТОР: Беляева Ольга Владимировна». Это первичный источник, а не
перепродажа данных, и он всегда актуальнее сайта.

Главный риск здесь — не «не найти», а «найти не того». В реестре три
разных ООО «Снабсталь» в трёх регионах, у каждого свой директор.
Подставить чужое имя в холодное письмо хуже, чем оставить поле пустым:
письмо «Здравствуйте, Роман» человеку по имени Кирилл — это провал
касания и потеря контакта навсегда.

Поэтому имя принимается только при выполнении обоих условий:
  - лучшее совпадение по названию не ниже STRONG_MATCH;
  - отрыв от второго кандидата не меньше MIN_GAP.
Если в реестре несколько одинаково похожих компаний — возвращаем пусто и
помечаем строку как неоднозначную. Пустое поле честнее угаданного.

Если на сайте компании удалось найти ИНН, идём по нему: это точный поиск,
неоднозначность исключена в принципе.
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from curl_cffi import requests
from rapidfuzz import fuzz

BASE = "https://egrul.nalog.ru"

STRONG_MATCH = 88.0   # ниже — не считаем совпадением вообще
MIN_GAP = 10.0        # отрыв от второго кандидата

# Юридические формы: в реестре они всегда есть, в названии с hh — не всегда.
# Без их вычистки «ООО Ромашка» и «ОБЩЕСТВО ... "РОМАШКА"» не совпадут.
LEGAL_FORM = re.compile(
    r"(общество с ограниченной ответственностью|"
    r"акционерное общество|публичное акционерное общество|"
    r"непубличное акционерное общество|закрытое акционерное общество|"
    r"открытое акционерное общество|"
    r"индивидуальный предприниматель|"
    r"\bооо\b|\bоао\b|\bзао\b|\bпао\b|\bнао\b|\bао\b|\bип\b|\bтд\b)",
    re.IGNORECASE,
)

# «ДИРЕКТОР: Иванов Иван», «Генеральный директор: Петров Пётр»
DIRECTOR_RE = re.compile(
    r"(?P<role>[А-ЯЁа-яё\s]*директор|управляющий|президент)\s*:\s*"
    r"(?P<fio>[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2})",
    re.IGNORECASE,
)


@dataclass
class EgrulResult:
    found: bool = False
    fio: str = ""
    role: str = ""
    official_name: str = ""
    inn: str = ""
    ambiguous: bool = False
    note: str = ""


def normalize_name(s: str) -> str:
    s = re.sub(r"[«»\"'`]", " ", (s or "").lower())
    s = LEGAL_FORM.sub(" ", s)
    s = re.sub(r"[^а-яёa-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_director(g: str) -> tuple[str, str]:
    """Из строки руководителя достаёт ФИО и должность."""
    if not g:
        return "", ""
    # Управляющая организация — это юрлицо, а не человек.
    if "управляющая организация" in g.lower():
        return "", ""
    m = DIRECTOR_RE.search(g)
    if not m:
        return "", ""
    return " ".join(m.group("fio").split()), m.group("role").strip().lower()


class Egrul:
    """
    Клиент к поиску ФНС с защитой от троттлинга.

    ФНС ограничивает частоту запросов, но делает это молча: после
    примерно десятка обращений подряд сервис продолжает отвечать 200 и
    отдаёт пустой список вместо результата. Поймано на прогоне базы —
    первые 11 компаний нашлись, следующие 56 «отсутствовали в ЕГРЮЛ»,
    включая те, что минутой позже находились вручную.

    Отсюда две меры: пауза между запросами и распознавание блокировки по
    серии пустых ответов. Серия пустых — это почти наверняка троттлинг, а
    не совпадение, поэтому ждём и пробуем ещё раз.
    """

    def __init__(
        self,
        delay: float = 0.9,
        timeout: int = 25,
        request_interval: float = 2.5,
        empty_streak_limit: int = 3,
    ):
        self.session = requests.Session(impersonate="chrome110")
        self.delay = delay
        self.timeout = timeout
        self.request_interval = request_interval
        self.empty_streak_limit = empty_streak_limit
        self._cache: dict[str, list[dict]] = {}
        self._last_request = 0.0
        self._empty_streak = 0

    def _pace(self) -> None:
        """Держит минимальный интервал между обращениями к ФНС."""
        wait = self.request_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _search_once(self, query: str) -> list[dict]:
        """
        Один цикл поиска. Протокол сервиса двухшаговый: POST кладёт запрос
        в очередь и отдаёт токен, затем GET по токену опрашивается, пока
        статус не перестанет быть "wait".
        """
        self._pace()
        try:
            r = self.session.post(
                f"{BASE}/",
                data={"query": query, "r": str(int(time.time() * 1000)), "vyp": "on"},
                timeout=self.timeout,
            )
            token = r.json().get("t")
        except Exception:
            return []
        if not token:
            return []

        for _ in range(10):
            time.sleep(self.delay)
            try:
                data = self.session.get(
                    f"{BASE}/search-result/{token}", timeout=self.timeout
                ).json()
            except Exception:
                continue
            if data.get("status") == "wait":
                continue
            return data.get("rows") or []
        return []

    def _search(self, query: str) -> list[dict]:
        if query in self._cache:
            return self._cache[query]

        rows = self._search_once(query)

        # Пустой ответ сам по себе нормален — компании может не быть.
        # Но серия пустых подряд означает, что нас притормозили: ждём и
        # повторяем последний запрос, чтобы не потерять данные молча.
        if rows:
            self._empty_streak = 0
        else:
            self._empty_streak += 1
            if self._empty_streak >= self.empty_streak_limit:
                backoff = min(60.0, 10.0 * (self._empty_streak - self.empty_streak_limit + 1))
                print(
                    f"    [ЕГРЮЛ] {self._empty_streak} пустых ответов подряд — "
                    f"похоже на ограничение частоты, пауза {backoff:.0f} с",
                    file=sys.stderr,
                )
                time.sleep(backoff)
                rows = self._search_once(query)
                if rows:
                    self._empty_streak = 0

        self._cache[query] = rows
        return rows

    def by_inn(self, inn: str) -> EgrulResult:
        """Точный поиск. Неоднозначности быть не может: ИНН уникален."""
        rows = self._search(inn)
        if not rows:
            return EgrulResult(note=f"ИНН {inn} не найден в ЕГРЮЛ")
        row = rows[0]
        fio, role = parse_director(row.get("g", ""))
        return EgrulResult(
            found=bool(fio),
            fio=fio,
            role=role,
            official_name=row.get("n", ""),
            inn=row.get("i", ""),
            note="" if fio else "в реестре не указан руководитель-физлицо",
        )

    def by_name(self, name: str, region_hint: str = "") -> EgrulResult:
        """
        Поиск по названию с защитой от однофамильцев-организаций.

        region_hint — город из карточки hh. Используется только чтобы
        развести одинаково названные компании, и только если он реально
        встречается в адресе из реестра.
        """
        rows = self._search(name)
        if not rows:
            return EgrulResult(note="в ЕГРЮЛ ничего не найдено")

        target = normalize_name(name)
        scored: list[tuple[float, dict]] = []
        for row in rows:
            cand = normalize_name(row.get("n", "") or row.get("c", ""))
            if not cand:
                continue
            score = fuzz.ratio(target, cand)
            # Совпадение города — слабый бонус, не решающий фактор.
            if region_hint:
                city = region_hint.split(",")[0].strip().lower()
                if city and city in (row.get("a", "") or "").lower():
                    score += 6
            scored.append((score, row))

        if not scored:
            return EgrulResult(note="пустая выдача")
        scored.sort(key=lambda t: t[0], reverse=True)

        top_score, top = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0

        if top_score < STRONG_MATCH:
            return EgrulResult(
                note=f"нет уверенного совпадения (лучшее {top_score:.0f})"
            )
        if top_score - second < MIN_GAP:
            return EgrulResult(
                ambiguous=True,
                note=(
                    f"в реестре {sum(1 for s, _ in scored if s >= STRONG_MATCH)} "
                    f"похожих компаний — имя не подставляю"
                ),
            )

        fio, role = parse_director(top.get("g", ""))
        return EgrulResult(
            found=bool(fio),
            fio=fio,
            role=role,
            official_name=top.get("n", ""),
            inn=top.get("i", ""),
            note="" if fio else "в реестре не указан руководитель-физлицо",
        )


if __name__ == "__main__":
    e = Egrul()
    queries = sys.argv[1:] or [
        "Белоярский Трубный Завод",   # однозначно
        "Снабсталь",                  # три одинаковых — должно отказать
        "6670484333",                 # точный поиск по ИНН
    ]
    for q in queries:
        res = e.by_inn(q) if q.isdigit() else e.by_name(q)
        status = "OK " if res.found else ("?? " if res.ambiguous else "-- ")
        print(f"{status} {q:<32} {res.fio or '—':<30} {res.note}")
        if res.official_name:
            print(f"     {res.official_name[:80]}  ИНН {res.inn}")

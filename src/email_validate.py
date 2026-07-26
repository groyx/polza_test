"""
Проверка почтовых адресов без платных сервисов.

Задача не «сказать да/нет», а разложить адрес по уровням риска. Для холодной
рассылки это принципиально: домен выгорает от bounce rate выше ~3-5%, поэтому
из базы нужно выкидывать невалидное, а к рискованному относиться осознанно.

Что проверяем и чего сознательно не делаем:

  syntax   — форма адреса. Дёшево, отсекает опечатки парсера.
  MX       — есть ли у домена почтовый обменник. Главная проверка: домен без MX
             почту не примет физически, это гарантированный hard bounce.
  role     — info@, sales@, zakaz@. Технически валидны и для B2B-аутрича в РФ
             часто единственное, что вообще опубликовано. Но читает их секретарь,
             а не ЛПР, и жалоба на спам с них прилетает чаще. Помечаем, не режем.
  free     — mail.ru, gmail, 126.com на строке, где заявлен корпоративный сайт.
             Сигнал, что данные собраны криво либо это перекуп/посредник.

SMTP-проба (RCPT TO без отправки) сознательно не реализована: в 2026 крупные
провайдеры её либо блокируют, либо отвечают accept-all, так что результат
недостоверен, а исходящие соединения на 25 порт с жилого IP быстро приводят
в блоклисты. Для реального проекта здесь ставится внешний валидатор.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache

# Практичная проверка формы. Полный RFC 5322 намеренно не реализуем: он
# допускает адреса, которые ни один реальный почтовик не примет.
EMAIL_RE = re.compile(
    r"^[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+"
    r"(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*"
    r"@(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,}$"
)

ROLE_PREFIXES = {
    "info", "sales", "office", "mail", "admin", "support", "contact",
    "zakaz", "order", "shop", "help", "hello", "team", "post", "secretary",
    "reception", "manager", "marketing", "pr", "hr", "job", "vacancy",
    "noreply", "no-reply", "webmaster", "abuse", "postmaster", "director",
}

FREE_MAILBOX_DOMAINS = {
    "gmail.com", "mail.ru", "bk.ru", "inbox.ru", "list.ru", "internet.ru",
    "yandex.ru", "ya.ru", "yandex.com", "rambler.ru", "outlook.com",
    "hotmail.com", "live.com", "icloud.com", "aol.com", "proton.me",
    "protonmail.com", "126.com", "163.com", "qq.com", "sina.com", "yeah.net",
}

DISPOSABLE_DOMAINS = {
    "mailinator.com", "10minutemail.com", "guerrillamail.com", "temp-mail.org",
    "throwaway.email", "yopmail.com", "trashmail.com", "sharklasers.com",
}


@dataclass
class EmailCheck:
    email: str
    domain: str = ""
    syntax_ok: bool = False
    has_mx: bool = False
    mx_hosts: list[str] = field(default_factory=list)
    is_role: bool = False
    is_free: bool = False
    is_disposable: bool = False

    @property
    def status(self) -> str:
        """valid — можно слать. risky — можно, но осознанно. invalid — выкинуть."""
        if not self.syntax_ok or self.is_disposable:
            return "invalid"
        if not self.has_mx:
            return "invalid"
        if self.is_free or self.is_role:
            return "risky"
        return "valid"

    @property
    def reason(self) -> str:
        bits = []
        if not self.syntax_ok:
            bits.append("некорректный формат")
        elif not self.has_mx:
            bits.append("у домена нет MX-записи — письмо не будет доставлено")
        if self.is_disposable:
            bits.append("одноразовый домен")
        if self.is_free:
            bits.append("бесплатный почтовый ящик, не корпоративный домен")
        if self.is_role:
            bits.append("ролевой адрес — читает не ЛПР")
        return "; ".join(bits) or "ок"


# Домашние и офисные роутеры часто проксируют DNS и молча роняют всё, кроме
# A/AAAA: MX-запрос уходит в таймаут. Поймано на реальной машине — системный
# резолвер 192.168.1.1 отвечал на A и висел на MX. Поэтому идём в публичные
# резолверы, а системный оставляем последним запасным вариантом.
_RESOLVER_POOL: tuple[tuple[str, ...] | None, ...] = (
    ("1.1.1.1", "1.0.0.1"),
    ("8.8.8.8", "8.8.4.4"),
    None,  # системный
)


def _resolvers():
    import dns.resolver

    for servers in _RESOLVER_POOL:
        r = dns.resolver.Resolver()
        if servers:
            r.nameservers = list(servers)
        r.lifetime = 4.0
        r.timeout = 4.0
        yield r


@lru_cache(maxsize=4096)
def _mx_lookup(domain: str) -> tuple[str, ...]:
    """
    MX-записи домена. Кэш на процесс: в базе много компаний на одном домене
    (например, филиалы), а DNS-запрос стоит ~50-200 мс.

    Если MX нет, но есть A-запись — по RFC 5321 почта может идти на A (implicit
    MX). Такое встречается у мелких российских хостингов, поэтому проверяем.
    """
    try:
        import dns.resolver  # noqa: F401
    except ImportError:
        # Без dnspython не притворяемся, что проверили: пусть вызывающий
        # код увидит пустой результат и не пометит адрес валидным ошибочно.
        return ()

    a_record_seen = False
    for resolver in _resolvers():
        try:
            answers = resolver.resolve(domain, "MX")
            hosts = sorted(str(r.exchange).rstrip(".") for r in answers)
            if hosts:
                return tuple(hosts)
        except Exception:
            pass

        if not a_record_seen:
            try:
                resolver.resolve(domain, "A")
                a_record_seen = True
            except Exception:
                pass

    return (f"{domain} (implicit MX по A-записи)",) if a_record_seen else ()


def check_email(email: str) -> EmailCheck:
    email = (email or "").strip().lower()
    res = EmailCheck(email=email)

    if not EMAIL_RE.match(email):
        return res
    res.syntax_ok = True

    local, _, domain = email.partition("@")
    res.domain = domain
    res.is_role = local.split("+")[0].split(".")[0] in ROLE_PREFIXES
    res.is_free = domain in FREE_MAILBOX_DOMAINS
    res.is_disposable = domain in DISPOSABLE_DOMAINS

    res.mx_hosts = list(_mx_lookup(domain))
    res.has_mx = bool(res.mx_hosts)
    return res


def check_many(emails: list[str], workers: int = 12) -> list[EmailCheck]:
    """
    Параллельная проверка. Упирается в DNS, не в CPU, поэтому потоки, а не
    процессы. 12 — эмпирический потолок, выше начинают отваливаться таймауты
    на публичных резолверах.
    """
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(check_email, emails))


if __name__ == "__main__":
    import sys

    samples = sys.argv[1:] or [
        "sales@iskroline.ru",        # корпоративный, ролевой
        "swyct@126.com",             # бесплатный китайский ящик
        "sales@technosphera.ru",
        "не почта",                  # мусор
        "test@domain-that-does-not-exist-xyz123.ru",
    ]
    for c in check_many(samples):
        print(f"{c.status:<8} {c.email:<45} {c.reason}")

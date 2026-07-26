"""
Тесты на чистые функции пайплайна.

Сознательно не тестируем то, что ходит в сеть. Проверка вида «скрапер
находит почту на сайте X» через месяц упадёт не из-за нашего изменения,
а потому что сайт переверстали. Такой тест не защищает код, а создаёт шум
и приучает игнорировать красный CI.

Поэтому здесь только детерминированная логика — нормализация, сравнение
названий, разбор форматов. Именно в ней живут ошибки, которые тихо портят
данные: неправильно свёрнутый домен или пропущенный дубль не роняют
скрипт, они просто дают кривую базу.

Почти каждый тест ниже — это баг, реально пойманный на живых данных.

Запуск:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest  # noqa: E402

from audit_base import coverage, domain_core, fold, name_core, similarity  # noqa: E402
from facts import Fact, _dedup_phrase, _is_company_name, compose  # noqa: E402
from build_base import normalize_site, pick_email, registrable  # noqa: E402
from egrul import normalize_name, parse_director  # noqa: E402
from email_validate import EMAIL_RE  # noqa: E402
from llm import _extract_json  # noqa: E402
from site_scrape import _decode_cf_email, _looks_like_fio  # noqa: E402


class TestFold:
    """Сведение кириллицы и латиницы к общему виду."""

    @pytest.mark.parametrize("company, domain", [
        ("ТД Ункомтех", "uncomtech.ru"),        # ц/c, х/ch
        ("РИЦ Техносфера", "technosphera.ru"),  # ph/f, ch/h
        ("Искролайн", "iskroline.ru"),          # й/y/i
    ])
    def test_cyrillic_matches_its_own_domain(self, company, domain):
        # Это ядро Задачи 4: без свёртки кириллица даёт 0 и подмену
        # столбцов поймать нельзя в принципе.
        assert similarity(company, domain) >= 75

    def test_фонетические_пары_совпадают(self):
        assert fold("Техносфера") == fold("technosphera")
        assert fold("Ункомтех") == fold("uncomtech")

    def test_domain_core_отбрасывает_зону_и_www(self):
        assert domain_core("https://www.jat-carbide.com/x") == fold("jat-carbide")
        assert domain_core("") == ""


class TestSimilarity:
    def test_не_склеивает_разные_компании_с_общим_куском(self):
        # Наивный SequenceMatcher давал здесь 0.90 из-за общего «techno»
        # и предлагал переставить домен не туда.
        assert similarity("Rogen Technologies", "technosphera.ru") < 62

    def test_покрытие_ловит_домен_с_пропущенным_словом(self):
        # ratio даёт всего ~71, потому что «cemented» в домен не попало,
        # но домен целиком собран из слов названия.
        assert coverage("JAT Cemented Carbide", "jat-carbide.com") == 100
        assert similarity("JAT Cemented Carbide", "jat-carbide.com") >= 75

    def test_короткие_слова_не_дают_ложного_покрытия(self):
        # Слова короче 3 букв совпадают случайно и покрытие не поднимают.
        assert coverage("ИП Ко", "kompaniya.ru") < 50

    def test_name_core_убирает_юрформу_и_родовые_слова(self):
        assert "ooo" not in name_core("ООО Ромашка")
        assert name_core("Ezhong Heavy Machinery") == fold("Ezhong")


class TestNormalizeSite:
    @pytest.mark.parametrize("raw, expected", [
        ("https://rabota.cdek.ru/", "https://cdek.ru"),
        ("http://job.megafon.ru", "https://megafon.ru"),
        ("https://hnrus.com/career/", "https://hnrus.com"),
        ("http://www.btz96.ru", "https://btz96.ru"),
    ])
    def test_hr_порталы_сводятся_к_основному_домену(self, raw, expected):
        # hh.ru часто отдаёт карьерный поддомен вместо сайта компании.
        assert normalize_site(raw) == expected

    def test_пустая_строка_не_ломает(self):
        assert normalize_site("") == ""

    @pytest.mark.parametrize("host, expected", [
        ("internor.com.cn", "internor.com.cn"),   # составная зона
        ("www.shop.example.ru", "example.ru"),
        ("mgwmachine.com", "mgwmachine.com"),
    ])
    def test_registrable(self, host, expected):
        assert registrable(host) == expected


class TestPickEmail:
    def test_предпочитает_адрес_на_домене_сайта(self):
        # Адрес на mail.ru при живом корпоративном сайте обычно
        # принадлежит подрядчику или дилеру, а не самой компании.
        email, note = pick_email(
            ["manager@mail.ru", "sales@mirtpa.ru"], "https://mirtpa.ru"
        )
        assert email == "sales@mirtpa.ru"
        assert note == ""

    def test_предпочитает_продажи_а_не_кадры(self):
        email, _ = pick_email(
            ["hr@x.ru", "sales@x.ru"], "https://x.ru"
        )
        assert email == "sales@x.ru"

    def test_помечает_чужой_домен(self):
        _, note = pick_email(["info@other.ru"], "https://x.ru")
        assert "не на домене" in note

    def test_пустой_список(self):
        email, note = pick_email([], "https://x.ru")
        assert email == "" and note


class TestPersonExtraction:
    @pytest.mark.parametrize("name", [
        "Скотников Виктор Александрович",   # отчество
        "Ольга Дорогина",                   # фамилия на -ина
    ])
    def test_принимает_настоящие_фио(self, name):
        assert _looks_like_fio(name)

    @pytest.mark.parametrize("name", [
        "Авто Премиум Москва",   # название компании, ловилось как ФИО
        "го компьютера",         # обрывок текста из-за IGNORECASE
        "Компания Центр",
        "Иван",                  # одно слово — не ФИО
    ])
    def test_отбрасывает_не_фио(self, name):
        assert not _looks_like_fio(name)


class TestCloudflareEmail:
    def test_расшифровка(self):
        # Собираем шифровку сами: первый байт ключ, дальше XOR.
        plain, key = "info@example.ru", 0x2a
        encoded = f"{key:02x}" + "".join(f"{ord(c) ^ key:02x}" for c in plain)
        assert _decode_cf_email(encoded) == plain

    def test_мусор_не_роняет(self):
        assert _decode_cf_email("нехекс") == ""
        assert _decode_cf_email("") == ""


class TestEgrulParsing:
    def test_достаёт_фио_директора(self):
        fio, role = parse_director("ДИРЕКТОР: Беляева Ольга Владимировна")
        assert fio == "Беляева Ольга Владимировна"
        assert "директор" in role

    def test_управляющая_организация_это_не_человек(self):
        # У части компаний руководитель — юрлицо. Подставлять его
        # название в обращение «Здравствуйте, ...» нельзя.
        fio, _ = parse_director(
            'Управляющая организация: ОБЩЕСТВО С ОГРАНИЧЕННОЙ '
            'ОТВЕТСТВЕННОСТЬЮ "КОРОНА ТЭХЕТ"'
        )
        assert fio == ""

    def test_normalize_name_снимает_юрформу(self):
        a = normalize_name('ОБЩЕСТВО С ОГРАНИЧЕННОЙ ОТВЕТСТВЕННОСТЬЮ "РОМАШКА"')
        b = normalize_name("ООО Ромашка")
        assert a == b == "ромашка"


class TestEmailRegex:
    @pytest.mark.parametrize("addr", [
        "sales@iskroline.ru", "a.b+c@sub.domain.co.uk", "swyct@126.com",
    ])
    def test_валидные(self, addr):
        assert EMAIL_RE.match(addr)

    @pytest.mark.parametrize("addr", [
        "не почта", "@nodomain.ru", "no-at.ru", "two@@at.ru", "trailing@dot.",
    ])
    def test_невалидные(self, addr):
        assert not EMAIL_RE.match(addr)


class TestFacts:
    @pytest.mark.parametrize("raw, expected", [
        # Вёрстка дублирует слоган во вложенных тегах — повтор не с начала.
        ("Сага Телеком — всегда надежное решение всегда надежное решение",
         "Сага Телеком — всегда надежное решение"),
        ("раз два три раз два три", "раз два три"),
        ("чистая строка без повторов", "чистая строка без повторов"),
    ])
    def test_dedup_phrase(self, raw, expected):
        assert _dedup_phrase(raw) == expected

    def test_заголовок_равный_названию_компании_отбрасывается(self):
        # «Заметил у вас направление "Белоярский трубный завод"» — это
        # зачитывание вывески, а не наблюдение.
        assert _is_company_name("Белоярский трубный завод",
                                "ООО Белоярский Трубный Завод")

    def test_настоящее_направление_не_отбрасывается(self):
        assert not _is_company_name("Поставка ЗИП для паровой турбины Siemens",
                                    "ООО Кронштадт")

    def test_compose_не_дублирует_вводное_слово(self):
        # Два факта подряд давали «Судя по сайту, X. Судя по сайту, Y».
        facts = [
            Fact("scale", 4, "Судя по сайту, производство в Москве", "q1", "u"),
            Fact("news", 1, "Судя по сайту, более 500 клиентов", "q2", "u"),
        ]
        text, _, _ = compose(facts)
        assert text.count("Судя по сайту") == 1

    def test_compose_не_берёт_вакансию_вторым_фактом(self):
        # Вакансия одинакова по всей базе и уже стоит в первом письме;
        # вторым факто́м она обнуляет различимость персонализации.
        facts = [
            Fact("product", 3, "Заметил у вас направление «Станки с ЧПУ»", "q", "u"),
            Fact("hiring", 9, "Увидел, что вы ищете менеджера", "q", "u"),
        ]
        text, _, _ = compose(facts)
        assert "менеджера" not in text

    def test_compose_на_пустом_списке(self):
        assert compose([]) == ("", "", "")


class TestExtractJson:
    def test_достаёт_из_markdown_забора(self):
        # Модели регулярно оборачивают JSON в ```json ... ``` и добавляют
        # пояснения, поэтому json.loads на сыром ответе падает.
        raw = 'Вот результат:\n```json\n{"personalization": "текст"}\n```\nГотово.'
        assert _extract_json(raw) == {"personalization": "текст"}

    def test_вложенные_объекты(self):
        assert _extract_json('{"a": {"b": 1}, "c": 2}') == {"a": {"b": 1}, "c": 2}

    def test_скобка_внутри_строки_не_сбивает_счётчик(self):
        assert _extract_json('{"t": "склеили } скобку"}') == {"t": "склеили } скобку"}

    def test_нет_json(self):
        assert _extract_json("совсем не json") is None

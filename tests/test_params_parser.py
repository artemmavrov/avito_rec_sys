from avito_rec_sys.data.params_parser import extract_whitelisted, parse_params


def test_doc_worked_example():
    # worked example: a raw params string before / after whitelist filtering
    raw = (
        "Вид услуги Компьютерная помощь Место оказания услуг пр-т Ленина "
        "Тип стоимости за услугу Начальная цена График работы от 34200 "
        "Услуга Диагностика компьютера Услуга Замена термопасты "
        "Услуга Ремонт после залития Услуга Чистка"
    )
    out = extract_whitelisted(raw)
    assert out.startswith("Вид услуги Компьютерная помощь")
    assert "пр-т Ленина" not in out
    assert "34200" not in out
    assert "Диагностика компьютера" in out
    assert "Замена термопасты" in out
    assert "Ремонт после залития" in out
    assert "Чистка" in out


def test_value_terminates_at_next_known_key_not_swallowed():
    raw = "Вид услуги Компьютерная помощь Место оказания услуг пр-т Ленина"
    pairs = dict(parse_params(raw))
    assert pairs["Вид услуги"] == "Компьютерная помощь"


def test_dedup_repeated_price_list_segments():
    raw = "Услуга Баня Стоимость 1600 Услуга Баня Стоимость 1700 Услуга Сауна Стоимость 1500"
    out = extract_whitelisted(raw)
    assert out.count("Баня") == 1
    assert "Сауна" in out


def test_empty_and_none():
    assert extract_whitelisted(None) == ""
    assert extract_whitelisted("") == ""
    assert parse_params(None) == []


def test_no_known_keys_returns_empty():
    assert extract_whitelisted("совершенно неизвестный текст без ключей") == ""

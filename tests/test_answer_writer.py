from avito_rec_sys.inference.answer_writer import write_answer
from avito_rec_sys.inference.validator import validate_answer

A, B, C = "a" * 16, "b" * 16, "0123456789abcdef"
QIDS = ["00WuFMaXSFZBxSzT", "03ztb1gtRFC4K4vP"]
CORPUS = {A, B, C}


def test_roundtrip_is_valid(tmp_path):
    p = write_answer(tmp_path / "answer.csv", QIDS, {QIDS[0]: [A, B], QIDS[1]: [C]})
    assert validate_answer(p, QIDS, CORPUS) == []
    # leading-zero-looking id survives as a string
    assert C in p.read_text(encoding="utf-8")


def test_writer_dedups_and_truncates(tmp_path):
    p = write_answer(tmp_path / "a.csv", QIDS, {QIDS[0]: [A, A, B, C], QIDS[1]: []}, max_len=2)
    text = p.read_text(encoding="utf-8").splitlines()
    assert text[1].endswith(f"{A} {B}")
    assert validate_answer(p, QIDS, CORPUS, max_len=2) == []


def test_validator_catches_problems(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text(
        "query_id,answer\n"
        f"{QIDS[0]},{A.upper()}\n"          # uppercase -> bad format
        f"{QIDS[0]},{A} {A}\n",             # duplicate query_id and repeated item
        encoding="utf-8",
    )
    problems = validate_answer(p, QIDS, CORPUS)
    joined = " | ".join(problems)
    assert "duplicate query_id" in joined
    assert "missing" in joined
    assert "not 16 lowercase hex" in joined
    assert "repeated item_ids" in joined


def test_validator_flags_item_not_in_corpus(tmp_path):
    p = write_answer(tmp_path / "a.csv", QIDS, {QIDS[0]: ["f" * 16], QIDS[1]: [A]})
    assert any("do not exist in the corpus" in x for x in validate_answer(p, QIDS, CORPUS))


def test_validator_rejects_wrong_header(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("id,answer\n", encoding="utf-8")
    assert validate_answer(p, QIDS, CORPUS)

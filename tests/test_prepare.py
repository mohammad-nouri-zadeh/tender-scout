import pandas as pd

from tender_scout.prepare import build_text, clean, label_names, parse_split


def test_clean_keeps_one_prevalent_row_per_lot(raw_month):
    lots, stats = clean(raw_month)
    assert lots["cig"].is_unique
    assert set(lots["cpv2"]) == {"33", "45", "79", "09"}
    # The orphan and the 3 division-name rows have no prevalent CPV.
    assert stats["cigs_without_prevalent_cpv"] == 4
    assert stats["lots"] == 51  # 50 + the code without check digit
    assert stats["cpv_not_available_dropped"] == 1
    assert stats["invalid_cpv_dropped"] == 0
    assert "99" not in set(lots["cpv2"])


def test_leading_zero_survives(raw_month):
    lots, _ = clean(raw_month)
    assert "09" in set(lots["cpv2"])


def test_build_text_adds_tender_title_only_when_different():
    df = pd.DataFrame({
        "oggetto_lotto": ["LOTTO 3 - SERBATOIO", "STRADE", "X  Y"],
        "oggetto_gara": ["GARA EUROPEA", "STRADE", None],
    })
    assert build_text(df).tolist() == ["LOTTO 3 - SERBATOIO | GARA EUROPEA", "STRADE", "X Y"]


def test_label_names_prefer_division_level_code(raw_month):
    names = label_names(raw_month)
    assert names["45"] == "LAVORI DI COSTRUZIONE"
    assert names["09"] == "GAS PROPANO LIQUEFATTO"  # no division-level row: most frequent


def test_parse_split_rejects_unknown_names():
    assert parse_split(["2025_01=train"]) == {"2025_01": "train"}
    try:
        parse_split(["2025_01=training"])
    except SystemExit:
        return
    raise AssertionError("expected SystemExit")

"""Synthetic ANAC-shaped data that reproduces the quirks found in the real files."""

import random

import pandas as pd
import pytest

from tender_scout.config import COLUMNS

VOCAB = {
    "33141600-6": ("RECIPIENTI SACCHE DRENAGGIO", ["fornitura", "dispositivi", "medici", "siringhe", "cateteri"]),
    "45233220-7": ("LAVORI DI SUPERFICIE PER STRADE", ["lavori", "manutenzione", "strade", "asfalto", "marciapiedi"]),
    "79940000-5": ("SERVIZI DI AGENZIE DI RISCOSSIONE", ["servizio", "riscossione", "tributi", "canone", "gestione"]),
    "09122110-4": ("GAS PROPANO LIQUEFATTO", ["fornitura", "gas", "propano", "gpl", "riscaldamento"]),
}
DIVISION_NAMES = {
    "33000000-0": "APPARECCHIATURE MEDICHE",
    "45000000-7": "LAVORI DI COSTRUZIONE",
    "79000000-4": "SERVIZI PER LE IMPRESE",
}


def make_month(key: str, n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        cpv = rng.choice(list(VOCAB))
        desc, words = VOCAB[cpv]
        text = " ".join(rng.choices(words, k=5)).upper()
        rows.append({
            "cig": f"{key}-{i:05d}", "numero_gara": f"G{key}-{i:05d}",
            "oggetto_gara": text, "oggetto_lotto": text,
            "oggetto_principale_contratto": "FORNITURE", "n_lotti_componenti": "1",
            "importo_lotto": "1000.0", "data_pubblicazione": f"{key.replace('_', '-')}-15",
            "cod_cpv": cpv, "descrizione_cpv": desc, "flag_prevalente": "1",
        })
    df = pd.DataFrame(rows)
    # A lot with a secondary CPV in another division (as in the real data).
    extra = df.iloc[[0]].assign(cod_cpv="45200000-9", descrizione_cpv="LAVORI", flag_prevalente="0")
    # A lot whose only row is secondary: it has no prevalent CPV.
    orphan = df.iloc[[1]].assign(cig=f"{key}-orphan", flag_prevalente="0")
    # Division-level codes, used only to name the labels.
    named = [df.iloc[[2]].assign(cig=f"{key}-div{j}", cod_cpv=code, descrizione_cpv=name, flag_prevalente="0")
             for j, (code, name) in enumerate(DIVISION_NAMES.items())]
    # Real files also carry codes without the check digit, and a "not available" code.
    no_check = df.iloc[[3]].assign(cig=f"{key}-nocheck", cod_cpv="45233220")
    unknown = df.iloc[[4]].assign(cig=f"{key}-unknown", cod_cpv="99999999",
                                  descrizione_cpv="Cpv prevalente non disponibile")
    return pd.concat([df, extra, orphan, *named, no_check, unknown], ignore_index=True)[COLUMNS]


@pytest.fixture
def raw_month():
    return make_month("2025_01", 50, seed=1).assign(file_month="2025_01")


@pytest.fixture
def project(tmp_path):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    for seed, key in enumerate(["2025_01", "2025_02", "2025_03"]):
        make_month(key, 400, seed).to_csv(raw / f"cig_csv_{key}.csv", sep=";", index=False)
    return tmp_path

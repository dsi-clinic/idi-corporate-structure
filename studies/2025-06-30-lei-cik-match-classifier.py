import json
import os
import itertools
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyarrow as pa
import pyarrow.parquet as pq
from thefuzz import fuzz

registry = pd.read_csv(
    "20250630-0800-gleif-goldencopy-lei2-golden-copy.csv",
    dtype=str,
    usecols=[
        "LEI",
        "Entity.LegalName",
        "Entity.LegalAddress.FirstAddressLine",
        "Entity.LegalAddress.City",
        "Entity.LegalAddress.Region",
        "Entity.LegalAddress.Country",
        "Entity.LegalAddress.PostalCode",
        "Entity.HeadquartersAddress.FirstAddressLine",
        "Entity.HeadquartersAddress.City",
        "Entity.HeadquartersAddress.Region",
        "Entity.HeadquartersAddress.Country",
        "Entity.HeadquartersAddress.PostalCode",
    ],
)

registry["LegalAddress-region"] = np.where(
    registry["Entity.LegalAddress.Region"].notna(),
    registry["Entity.LegalAddress.Region"],
    registry["Entity.LegalAddress.Country"],
)
registry["HeadquartersAddress-region"] = np.where(
    registry["Entity.HeadquartersAddress.Region"].notna(),
    registry["Entity.HeadquartersAddress.Region"],
    registry["Entity.HeadquartersAddress.Country"],
)

us_or_canada = registry["Entity.LegalAddress.Country"].isin(["US", "CA"]) | registry[
    "Entity.HeadquartersAddress.Country"
].isin(["US", "CA"])

sec2023 = pd.read_csv("data/sec-2023.csv", dtype={"cik": str})

address_replacement = {
    "STREET": "ST",
    "AVENUE": "AVE",
    "BOULEVARD": "BLVD",
    "LANE": "LN",
    "DRIVE": "DR",
    "ROAD": "RD",
    "CRESCENT": "CRES",
    "PLACE": "PL",
    "TERRACE": "TER",
    "COURT": "CT",
    "CIRCLE": "CIR",
    "SQUARE": "SQ",
    "ALLEY": "ALY",
    "MOUNT": "MT",
    "HILL": "HL",
    "HILLS": "HLS",
    "ESTATE": "EST",
    "ESTATES": "ESTS",
    "GARDEN": "GDN",
    "GARDENS": "GDNS",
    "GREEN": "GRN",
    "GROVE": "GRV",
    "PARKWAY": "PKWY",
    "PARK": "PK",
    "PARKS": "PKS",
    "PARKLAND": "PKLD",
    "MARKET": "MKT",
    "HIGHWAY": "HWY",
    "TOLLWAY": "TLWY",
    "FLAT": "FLT",
    "SUITE": "STE",
    "TOWER": "TWR",
    "BUILDING": "BLDG",
    "BLOCK": "BLK",
    "APARTMENT": "APT",
    "FLOOR": "FLR",
    "INDUSTRIAL": "IND",
    "CENTER": "CTR",
    "COMPLEX": "CMPLX",
    "UNIVERSITY": "UNIV",
    "INSTITUTE": "INST",
    "PLAZA": "PLZ",
    "TRAIL": "TRL",
    "BRIDGE": "BRG",
    "EAST": "E",
    "WEST": "W",
    "SOUTH": "S",
    "NORTH": "N",
    "POINT": "PT",
    "PENTHOUSE": "PH",
    "SAINT": "ST",
    "SAINTS": "STS",
    "JUNCTION": "JCT",
    "CROSSING": "XING",
    "EXPRESSWAY": "EXPY",
    "FREEWAY": "FWY",
    "EXTENSION": "EXT",
    "MEADOWS": "MDWS",
    "FIELDS": "FLDS",
    "FIELD": "FLD",
    "WOODS": "WDS",
    "FOREST": "FRST",
    "ROOM": "RM",
    "FIRST": "1ST",
    "SECOND": "2ND",
    "THIRD": "3RD",
    "FOURTH": "4TH",
    "FIFTH": "5TH",
    "SIXTH": "6TH",
    "SEVENTH": "7TH",
    "EIGHTH": "8TH",
    "NINTH": "9TH",
    "TENTH": "10TH",
    "ELEVENTH": "11TH",
    "TWELVTH": "12TH",
    "ONE": "1",
    "TWO": "2",
    "THREE": "3",
    "FOUR": "4",
    "FIVE": "5",
    "SIX": "6",
    "SEVEN": "7",
    "EIGHT": "8",
    "NINE": "9",
    "TEN": "10",
    "ELEVEN": "11",
    "TWELVE": "12",
    "THIRTEEN": "13",
    "FOURTEEN": "14",
    "FIFTEEN": "15",
    "SIXTEEN": "16",
    "SEVENTEEN": "17",
    "EIGHTEEN": "18",
    "NINTEEN": "19",
    "TWENTY": "20",
}


def normalize_address(address):
    if not isinstance(address, str):
        return ""
    return " ".join(
        address_replacement.get(word, word)
        for word in address.upper()
        .replace("POST OFFICE", "PO")
        .replace("P.O", "PO")
        .replace(".", " ")
        .replace(",", " ")
        .replace("-", " ")
        .split()
        if word != "C/O"
    )


us_codes = {
    "ALABAMA": "AL",
    "KENTUCKY": "KY",
    "OHIO": "OH",
    "ALASKA": "AK",
    "LOUISIANA": "LA",
    "OKLAHOMA": "OK",
    "ARIZONA": "AZ",
    "MAINE": "ME",
    "OREGON": "OR",
    "ARKANSAS": "AR",
    "MARYLAND": "MD",
    "PENNSYLVANIA": "PA",
    "AMERICAN SAMOA": "AS",
    "MASSACHUSETTS": "MA",
    "PUERTO RICO": "PR",
    "CALIFORNIA": "CA",
    "MICHIGAN": "MI",
    "RHODE ISLAND": "RI",
    "COLORADO": "CO",
    "MINNESOTA": "MN",
    "SOUTH CAROLINA": "SC",
    "CONNECTICUT": "CT",
    "MISSISSIPPI": "MS",
    "SOUTH DAKOTA": "SD",
    "DELAWARE": "DE",
    "MISSOURI": "MO",
    "TENNESSEE": "TN",
    "DISTRICT OF COLUMBIA": "DC",
    "MONTANA": "MT",
    "TEXAS": "TX",
    "FLORIDA": "FL",
    "NEBRASKA": "NE",
    "TRUST TERRITORIES": "TT",
    "GEORGIA": "GA",
    "NEVADA": "NV",
    "UTAH": "UT",
    "GUAM": "GU",
    "NEW HAMPSHIRE": "NH",
    "VERMONT": "VT",
    "HAWAII": "HI",
    "NEW JERSEY": "NJ",
    "VIRGINIA": "VA",
    "IDAHO": "ID",
    "NEW MEXICO": "NM",
    "VIRGIN ISLANDS": "VI",
    "ILLINOIS": "IL",
    "NEW YORK": "NY",
    "WASHINGTON": "WA",
    "INDIANA": "IN",
    "NORTH CAROLINA": "NC",
    "WEST VIRGINIA": "WV",
    "IOWA": "IA",
    "NORTH DAKOTA": "ND",
    "WISCONSIN": "WI",
    "KANSAS": "KS",
    "NORTHERN MARIANA ISLANDS": "MP",
    "WYOMING": "WY",
}
ca_codes = {
    "ALBERTA": "AB",
    "BRITISH COLUMBIA": "BC",
    "MANITOBA": "MB",
    "NEW BRUNSWICK": "NB",
    "NEWFOUNDLAND AND LABRADOR": "NL",
    "NORTHWEST TERRITORIES": "NT",
    "NOVA SCOTIA": "NS",
    "NUNAVUT": "NU",
    "ONTARIO": "ON",
    "PRINCE EDWARD ISLAND": "PE",
    "QUEBEC": "QC",
    "QUÉBEC": "QC",
    "SASKATCHEWAN": "SK",
    "YUKON": "YT",
}


def states_to_codes(x):
    if not isinstance(x, str):
        return ""
    y = " ".join(x.upper().replace(".", " ").replace(",", " ").split())
    if y in us_codes.values():
        return f"US-{y}"
    if y in ca_codes.values():
        return f"CA-{y}"
    z = us_codes.get(y)
    if z is not None:
        return f"US-{z}"
    z = ca_codes.get(y)
    if z is not None:
        return f"CA-{z}"
    if y == "CAYMAN ISLANDS":
        return "KY"
    assert False


columns = {
    "LEI": "LEI",
    "Entity.LegalName": "name",
    "Entity.LegalAddress.FirstAddressLine": "legal_address",
    "Entity.LegalAddress.City": "legal_city",
    "LegalAddress-region": "legal_region",
    "Entity.LegalAddress.PostalCode": "legal_zip",
    "Entity.HeadquartersAddress.FirstAddressLine": "hq_address",
    "Entity.HeadquartersAddress.City": "hq_city",
    "HeadquartersAddress-region": "hq_region",
    "Entity.HeadquartersAddress.PostalCode": "hq_zip",
}
gleif = registry[us_or_canada][columns.keys()].rename(columns=columns)
gleif["name"] = gleif["name"].str.upper()
gleif["legal_address"] = gleif["legal_address"].apply(normalize_address)
gleif["legal_city"] = gleif["legal_city"].str.upper()
gleif["legal_region"] = gleif["legal_region"].str.upper()
gleif["legal_zip"] = gleif["legal_zip"].str.upper().fillna("")
gleif["hq_address"] = gleif["hq_address"].apply(normalize_address)
gleif["hq_city"] = gleif["hq_city"].str.upper()
gleif["hq_region"] = gleif["hq_region"].str.upper()
gleif["hq_zip"] = gleif["hq_zip"].str.upper().fillna("")

sec = pd.DataFrame(
    {
        "CIK": sec2023["cik"].fillna(""),
        "name": sec2023["name"].str.upper(),
        "address": sec2023["address"].apply(normalize_address),
        "city": sec2023["city"].apply(
            lambda x: (
                " ".join(x.upper().replace(".", " ").replace(",", " ").split())
                if isinstance(x, str)
                else ""
            )
        ),
        "state": sec2023["state"].apply(states_to_codes),
        "zip": sec2023["zip"].str.upper().fillna(""),
    }
)

del registry
del sec2023

with open("/net/projects2/spun-hyper/idi-corporate-structure/comparisons.csv", "w") as file:
    file.write("LEI,CIK,name,legal_address,legal_city,legal_state,legal_zip,hq_address,hq_city,hq_state,hq_zip,is_us\n")

    for i, (grow, srow) in (
        enumerate(itertools.product(gleif.itertuples(), sec.itertuples()))
        # total=len(gleif) * len(sec),
    ):
        if i % 1000 == 0:
            print(i / (len(gleif) * len(sec)))

        LEI = grow.LEI
        CIK = srow.CIK
        name = fuzz.ratio(grow.name, srow.name)
        legal_address = fuzz.ratio(grow.legal_address, srow.address)
        legal_city = fuzz.ratio(grow.legal_city, srow.city)
        legal_state = fuzz.ratio(grow.legal_region, srow.state)
        legal_zip = len(os.path.commonprefix([grow.legal_zip, srow.zip]))
        hq_address = fuzz.ratio(grow.hq_address, srow.address)
        hq_city = fuzz.ratio(grow.hq_city, srow.city)
        hq_state = fuzz.ratio(grow.hq_region, srow.state)
        hq_zip = len(os.path.commonprefix([grow.hq_zip, srow.zip]))
        yes = srow.state.startswith("US-")
        is_us = (1 if grow.legal_region.startswith("US-") and yes else 0) + (
            2 if grow.hq_region.startswith("US-") and yes else 0
        )

        file.write(f"{LEI},{CIK},{name},{legal_address},{legal_city},{legal_state},{legal_zip},{hq_address},{hq_city},{hq_state},{hq_zip},{is_us}\n")

print("DONE")

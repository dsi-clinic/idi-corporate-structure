import os
import sys
import itertools

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from thefuzz import fuzz
from tqdm import tqdm

which = sys.argv[1]

gleif = pd.read_parquet(f"gleif_{which}.parquet")
sec = pd.read_parquet("sec.parquet")

LEI = np.empty(len(gleif) * len(sec), dtype=object)
CIK = np.empty(len(gleif) * len(sec), dtype=object)
name = np.full(len(gleif) * len(sec), -1, dtype=np.int8)
legal_address = np.full(len(gleif) * len(sec), -1, dtype=np.int8)
legal_city = np.full(len(gleif) * len(sec), -1, dtype=np.int8)
legal_state = np.full(len(gleif) * len(sec), -1, dtype=np.int8)
legal_zip = np.full(len(gleif) * len(sec), -1, dtype=np.int8)
hq_address = np.full(len(gleif) * len(sec), -1, dtype=np.int8)
hq_city = np.full(len(gleif) * len(sec), -1, dtype=np.int8)
hq_state = np.full(len(gleif) * len(sec), -1, dtype=np.int8)
hq_zip = np.full(len(gleif) * len(sec), -1, dtype=np.int8)
is_us = np.full(len(gleif) * len(sec), -1, dtype=np.int8)

for i, (grow, srow) in tqdm(
    enumerate(itertools.product(gleif.itertuples(), sec.itertuples())),
    total=len(gleif) * len(sec),
):
    LEI[i] = grow.LEI
    CIK[i] = srow.CIK
    name[i] = fuzz.ratio(grow.name, srow.name)
    legal_address[i] = fuzz.ratio(grow.legal_address, srow.address)
    legal_city[i] = fuzz.ratio(grow.legal_city, srow.city)
    legal_state[i] = fuzz.ratio(grow.legal_region, srow.state)
    legal_zip[i] = len(os.path.commonprefix([grow.legal_zip, srow.zip]))
    hq_address[i] = fuzz.ratio(grow.hq_address, srow.address)
    hq_city[i] = fuzz.ratio(grow.hq_city, srow.city)
    hq_state[i] = fuzz.ratio(grow.hq_region, srow.state)
    hq_zip[i] = len(os.path.commonprefix([grow.hq_zip, srow.zip]))
    yes = srow.state.startswith("US-")
    is_us[i] = (1 if grow.legal_region.startswith("US-") and yes else 0) + (
        2 if grow.hq_region.startswith("US-") and yes else 0
    )

def make_table(start, stop):
    return pa.Table.from_arrays(
        [
            pa.array(LEI[start:stop]),
            pa.array(CIK[start:stop]),
            pa.array(name[start:stop]),
            pa.array(legal_address[start:stop]),
            pa.array(legal_city[start:stop]),
            pa.array(legal_state[start:stop]),
            pa.array(legal_zip[start:stop]),
            pa.array(hq_address[start:stop]),
            pa.array(hq_city[start:stop]),
            pa.array(hq_state[start:stop]),
            pa.array(hq_zip[start:stop]),
            pa.array(is_us[start:stop]),
        ],
        names=[
            "LEI",
            "CIK",
            "name",
            "legal_address",
            "legal_city",
            "legal_state",
            "legal_zip",
            "hq_address",
            "hq_city",
            "hq_state",
            "hq_zip",
            "is_us",
        ],
    )

writer = pq.ParquetWriter(
    f"cartesian-product-{which}.parquet",
    make_table(0, 1).schema,
    compression="gzip",
    compression_level=4,
)

BATCH_SIZE = 1000000

for start in (range(0, len(LEI), BATCH_SIZE)):
    stop = min(len(LEI), start + BATCH_SIZE)
    print(f"writing {100*stop/len(LEI):.0f}")
    for batch in make_table(start, stop).to_batches():
        writer.write_batch(batch)

writer.close()

print(f"done with {which}")

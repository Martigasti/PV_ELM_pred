"""
Preprocessing dispatcher: builds the dataset's 30-min (or native-step) .npy
cache, routing to the right parser based on the dataset's PARSER (set/inferred
in dataset_config.py). One entry point for every dataset:

    DATASET=Palaiseau python scripts/preprocessing.py   # CSV  -> preprocessing_csv
    DATASET=solete    python scripts/preprocessing.py   # NC   -> preprocessing_nc
    DATASET=meteo_vignola_temperature python scripts/preprocessing.py  # -> preprocessing_meteo

The CSV/NetCDF/meteo parsers stay importable and runnable on their own; this
dispatcher just calls the matching `main()`, so adding a CSV dataset needs no new
script (only an entry in dataset_config.py). run_full.py calls this dispatcher
automatically when the cache is missing.
"""

import time

from dataset_config import DATASET, PARSER


def main():
    if PARSER == "nc":
        import preprocessing_nc as mod
    elif PARSER == "meteo":
        import preprocessing_meteo as mod
    else:  # "csv"
        import preprocessing_csv as mod
    print(f"[dispatcher] DATASET={DATASET} -> preprocessing_{PARSER}")
    mod.main()


if __name__ == "__main__":
    t0 = time.time()
    main()
    dt = time.time() - t0
    print(f"\n[chrono] preprocessing.py ({DATASET}) : {dt:.2f} s  ({dt/60:.2f} min)")

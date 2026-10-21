"""Stage `featurize`: fit the feature pipeline (see `build.py`) on the train split."""

import joblib
import pandas as pd

from fjd.data.io import REPORTS, read_split, write_json
from fjd.features.build import build_featurizer
from fjd.utils.config import PROJECT_ROOT, load_params
from fjd.utils.seed import set_seed

OUT = PROJECT_ROOT / "models" / "featurizer.joblib"
REPORT = REPORTS / "features.json"


def main() -> None:
    params = load_params()
    set_seed(params["seed"])
    featurizer = build_featurizer(params["features"]).fit(read_split("train"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(featurizer, OUT)
    names = featurizer.get_feature_names_out()
    blocks = pd.Series(names).str.split("__").str[0].value_counts().sort_index()
    write_json({"n_features": len(names), "per_block": blocks.to_dict()}, REPORT)


if __name__ == "__main__":
    main()

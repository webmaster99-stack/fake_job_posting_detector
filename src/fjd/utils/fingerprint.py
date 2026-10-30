"""Fingerprint the featurizer's output (`python -m fjd.utils.fingerprint`), to tell whether two
machines compute identical features. CI prints it after `dvc repro --force` (pipeline-pr.yml).

md5 of the sparse val matrix (values, indices, row pointers) from `models/featurizer.joblib`, and
of the χ² scores that the tree models' feature selection ranks (ties there flip selections).
"""

import hashlib

import joblib
import numpy as np
from sklearn.feature_selection import chi2

from fjd.data.io import read_split
from fjd.data.schema import TARGET
from fjd.models.train import FEATURIZER


def md5(*arrays: np.ndarray) -> str:
    digest = hashlib.md5()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def main() -> None:
    featurizer = joblib.load(FEATURIZER)
    val = featurizer.transform(read_split("val")).tocsr()
    train = read_split("train")
    scores, _ = chi2(featurizer.transform(train), train[TARGET])
    print(f"val features: {md5(val.data, val.indices, val.indptr)}")
    print(f"train chi2:   {md5(scores)}")


if __name__ == "__main__":
    main()

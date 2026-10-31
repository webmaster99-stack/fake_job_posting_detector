"""Stage `drift_reference`: the training split's drift columns, which live predictions are
compared with (`fjd.monitoring.drift`, `.github/workflows/drift.yml`)."""

from fjd.data.io import PROCESSED, read_split, write_parquet
from fjd.monitoring.features import monitoring_frame

OUT = PROCESSED / "drift_reference.parquet"


def main() -> None:
    write_parquet(monitoring_frame(read_split("train")), OUT)


if __name__ == "__main__":
    main()

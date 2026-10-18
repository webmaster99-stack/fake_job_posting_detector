"""Stage `validate`: fail the pipeline if the ingested data breaks the schema."""

from fjd.data.ingest import OUT as INGESTED
from fjd.data.io import REPORTS, read_parquet, write_json
from fjd.data.schema import SCHEMA, TARGET

OUT = REPORTS / "data_validation.json"


def main() -> None:
    df = read_parquet(INGESTED)
    SCHEMA.validate(df, lazy=True)  # raises SchemaErrors listing every failure
    write_json(
        {
            "rows": len(df),
            "fraud_rate": round(float(df[TARGET].mean()), 6),
            "rows_per_source": df["source"].value_counts().sort_index().to_dict(),
        },
        OUT,
    )


if __name__ == "__main__":
    main()

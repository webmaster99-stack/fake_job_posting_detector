"""predictions and feedback

Revision ID: 0001
Revises:
Create Date: 2026-10-27 11:31:20.837283
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_name", sa.String(length=64), nullable=False),
        sa.Column("model_version", sa.String(length=32), nullable=True),
        sa.Column(
            "posting",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column("probability", sa.Double(), nullable=False),
        sa.Column("fraud", sa.Boolean(), nullable=False),
        sa.Column("threshold", sa.Double(), nullable=False),
        sa.Column("latency_ms", sa.Double(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_predictions_created_at"), "predictions", ["created_at"], unique=False)
    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("prediction_id", sa.Uuid(), nullable=False),
        sa.Column("correct", sa.Boolean(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["prediction_id"], ["predictions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("prediction_id"),
    )


def downgrade() -> None:
    op.drop_table("feedback")
    op.drop_index(op.f("ix_predictions_created_at"), table_name="predictions")
    op.drop_table("predictions")

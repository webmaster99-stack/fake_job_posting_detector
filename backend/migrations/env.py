"""Alembic environment: migrates the database in `DATABASE_URL` (see backend.settings).

A caller may pass its own URL as `config.attributes["url"]` (the tests do).
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine

from backend.db import Base
from backend.settings import Settings

config = context.config
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)
url = config.attributes.get("url") or Settings.from_env().database_url


def run_migrations_offline() -> None:
    context.configure(url=url, target_metadata=Base.metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    with create_engine(url).connect() as connection:
        context.configure(connection=connection, target_metadata=Base.metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

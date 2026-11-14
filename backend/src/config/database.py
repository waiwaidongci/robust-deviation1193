from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from .settings import DATABASE_URL
class Base(DeclarativeBase):
    pass
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
def upgrade_schema():
    """Idempotent upgrade: historical single-peak runs get mode='single'."""
    inspector=inspect(engine)
    if 'deviation_runs' not in inspector.get_table_names():return
    columns={column['name'] for column in inspector.get_columns('deviation_runs')}
    if 'mode' not in columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE deviation_runs ADD COLUMN mode VARCHAR(16) NOT NULL DEFAULT 'single'"))
    with engine.begin() as connection:
        connection.execute(text("UPDATE deviation_runs SET mode='single' WHERE mode IS NULL OR mode=''"))
def get_db():
    with SessionLocal() as db:
        yield db

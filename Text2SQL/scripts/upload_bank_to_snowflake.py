import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from db.snowflake_connector import snowflake_connection


CSV_PATH = ROOT_DIR / "data" / "bank.csv"
DEFAULT_DATABASE = "TEXT2SQL_DB"
DEFAULT_SCHEMA = "PUBLIC"
DEFAULT_TABLE = "BANK"
CHUNK_SIZE = 10_000

COLUMN_TYPES = {
    "age": "NUMBER(3,0)",
    "job": "VARCHAR(50)",
    "marital": "VARCHAR(20)",
    "education": "VARCHAR(20)",
    "default": "VARCHAR(10)",
    "balance": "NUMBER(12,0)",
    "housing": "VARCHAR(10)",
    "loan": "VARCHAR(10)",
    "contact": "VARCHAR(20)",
    "day": "NUMBER(2,0)",
    "month": "VARCHAR(10)",
    "duration": "NUMBER(10,0)",
    "campaign": "NUMBER(10,0)",
    "pdays": "NUMBER(10,0)",
    "previous": "NUMBER(10,0)",
    "poutcome": "VARCHAR(20)",
    "y": "VARCHAR(10)",
}


def quote_identifier(identifier: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
        raise ValueError(f"Invalid Snowflake identifier: {identifier}")
    return f'"{identifier.upper()}"'


def qualified_name(database: str, schema: str, table: str) -> str:
    return ".".join(
        [
            quote_identifier(database),
            quote_identifier(schema),
            quote_identifier(table),
        ]
    )


def create_table_sql(database: str, schema: str, table: str, columns: list[str]) -> str:
    column_defs = []
    for column in columns:
        snowflake_type = COLUMN_TYPES.get(column)
        if snowflake_type is None:
            raise ValueError(f"Unexpected CSV column: {column}")
        column_defs.append(f"{quote_identifier(column)} {snowflake_type}")

    return f"""
CREATE TABLE IF NOT EXISTS {qualified_name(database, schema, table)} (
    {", ".join(column_defs)}
)
""".strip()


def truncate_table_sql(database: str, schema: str, table: str) -> str:
    return f"TRUNCATE TABLE {qualified_name(database, schema, table)}"


def insert_sql(database: str, schema: str, table: str, columns: list[str]) -> str:
    column_names = ", ".join(quote_identifier(column) for column in columns)
    placeholders = ", ".join(["%s"] * len(columns))
    return f"""
INSERT INTO {qualified_name(database, schema, table)} ({column_names})
VALUES ({placeholders})
""".strip()


def upload_csv(
    csv_path: Path,
    database: str,
    schema: str,
    table: str,
    replace: bool,
    chunk_size: int,
) -> int:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    load_dotenv(ROOT_DIR / ".env")
    total_rows = 0
    reader = pd.read_csv(csv_path, sep=";", chunksize=chunk_size)

    with snowflake_connection({"database": database, "schema": schema}) as conn:
        cursor = conn.cursor()
        try:
            first_chunk = True
            for chunk in reader:
                columns = list(chunk.columns)
                if first_chunk:
                    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {quote_identifier(database)}")
                    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {quote_identifier(database)}.{quote_identifier(schema)}")
                    cursor.execute(create_table_sql(database, schema, table, columns))
                    if replace:
                        cursor.execute(truncate_table_sql(database, schema, table))
                    first_chunk = False

                records = [tuple(row) for row in chunk.itertuples(index=False, name=None)]
                if records:
                    cursor.executemany(insert_sql(database, schema, table, columns), records)
                    total_rows += len(records)

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    return total_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload data/bank.csv to Snowflake database TEXT2SQL_DB."
    )
    parser.add_argument("--csv", type=Path, default=CSV_PATH, help="Path to bank.csv.")
    parser.add_argument("--database", default=DEFAULT_DATABASE, help="Snowflake database name.")
    parser.add_argument("--schema", default=DEFAULT_SCHEMA, help="Snowflake schema name.")
    parser.add_argument("--table", default=DEFAULT_TABLE, help="Snowflake table name.")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Truncate the target table before uploading rows.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Number of CSV rows to insert per batch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = upload_csv(
        csv_path=args.csv,
        database=args.database,
        schema=args.schema,
        table=args.table,
        replace=args.replace,
        chunk_size=args.chunk_size,
    )
    print(f"Uploaded {rows} rows to {args.database}.{args.schema}.{args.table}")


if __name__ == "__main__":
    main()

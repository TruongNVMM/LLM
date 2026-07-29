import os
from typing import Dict, Optional
from contextlib import contextmanager

from dotenv import load_dotenv
import snowflake.connector

load_dotenv()


def _conn_kwargs(overrides: Optional[Dict[str, str]] = None) -> Dict[str, Optional[str]]:
    """Build connection kwargs from environment with optional overrides."""
    overrides = overrides or {}
    return {
        "account": overrides.get("account") or os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": overrides.get("user") or os.getenv("SNOWFLAKE_USER"),
        "authenticator": overrides.get("authenticator") or os.getenv("SNOWFLAKE_AUTHENTICATOR") or "SNOWFLAKE_JWT",
        "private_key_file": overrides.get("private_key_file") or os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH"),
        "private_key_file_pwd": overrides.get("private_key_file_pwd") or os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE"),
        "warehouse": overrides.get("warehouse") or os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": overrides.get("database") or os.getenv("SNOWFLAKE_DATABASE"),
        "schema": overrides.get("schema") or os.getenv("SNOWFLAKE_SCHEMA"),
        "role": overrides.get("role") or os.getenv("SNOWFLAKE_ROLE"),
    }

def get_snowflake_connection(overrides: Optional[Dict[str, str]] = None) -> snowflake.connector.SnowflakeConnection:
    """Return a new Snowflake connection using env vars or provided overrides.

    Example:
        conn = get_snowflake_connection()
        cur = conn.cursor()
        ...
        cur.close()
        conn.close()
    """
    kwargs = {k: v for k, v in _conn_kwargs(overrides).items() if v is not None}
    return snowflake.connector.connect(**kwargs)


@contextmanager
def snowflake_connection(overrides: Optional[Dict[str, str]] = None):
    """Context manager that yields a Snowflake connection and ensures it is closed.

    Usage:
        with snowflake_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT CURRENT_USER()")
            ...
    """
    conn = get_snowflake_connection(overrides)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass

if __name__ == "__main__":
    with snowflake_connection() as conn:
        cur = conn.cursor()

        try:
            cur.execute("SELECT CURRENT_USER(), CURRENT_ROLE(), CURRENT_WAREHOUSE(), CURRENT_DATABASE()")
            result = cur.fetchone()

            print("User:", result[0])
            print("Role:", result[1])
            print("Warehouse:", result[2])
            print("Database:", result[3])

        finally:
            cur.close()
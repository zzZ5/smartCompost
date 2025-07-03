import sqlite3
import os

DB_PATH = "compost_data.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS compost_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            T1 REAL,
            T2 REAL,
            T3 REAL,
            O2 REAL,
            CO2 REAL,
            H2O REAL,
            fan TEXT,
            p_value REAL
        )
    """)
    conn.commit()
    conn.close()
    print("✅ 数据库初始化完成")


def save_data(data):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO compost_data (time, T1, T2, T3, O2, CO2, H2O, fan, p_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                data.get("time"),
                data.get("T1"),
                data.get("T2"),
                data.get("T3"),
                data.get("O2"),
                data.get("CO2"),
                data.get("H2O"),
                data.get("fan"),
                data.get("p_value"),
            ),
        )
        conn.commit()
        conn.close()
        print("📝 数据已写入数据库")
    except Exception as e:
        print(f"❌ 数据库写入失败: {e}")

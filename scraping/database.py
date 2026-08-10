import sqlite3
import pandas as pd
from pathlib import Path


DATABASE_PATH = "database/books.db"


def create_database():

    Path("database").mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    # Enable foreign key enforcement
    cursor.execute("PRAGMA foreign_keys = ON")

    # Categories table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        )
    """)

    # Books table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,

            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
    """)

    connection.commit()

    return connection


def load_data(df):

    connection = create_database()

    cursor = connection.cursor()

    # Clear existing data so the pipeline can be rerun
    cursor.execute("DELETE FROM books")
    cursor.execute("DELETE FROM categories")

    # Insert unique categories
    categories = df["category"].drop_duplicates()

    for category in categories:

        cursor.execute(
            """
            INSERT INTO categories (category_name)
            VALUES (?)
            """,
            (category,)
        )

    # Insert books
    for _, row in df.iterrows():

        cursor.execute(
            """
            SELECT category_id
            FROM categories
            WHERE category_name = ?
            """,
            (row["category"],)
        )

        category_id = cursor.fetchone()[0]

        cursor.execute(
            """
            INSERT INTO books
            (
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["title"],
                float(row["price_gbp"]),
                float(row["price_inr"]),
                int(row["rating"]),
                int(bool(row["in_stock"])),
                category_id
            )
        )

    connection.commit()

    print(
        f"Database created successfully: {DATABASE_PATH}"
    )

    print(
        "Books inserted:",
        cursor.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    )

    print(
        "Categories inserted:",
        cursor.execute(
            "SELECT COUNT(*) FROM categories"
        ).fetchone()[0]
    )

    connection.close()


if __name__ == "__main__":

    df = pd.read_csv("data/cleaned_books.csv")

    load_data(df)
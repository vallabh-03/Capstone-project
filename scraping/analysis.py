import sqlite3
from pathlib import Path

import pandas as pd


DATABASE_PATH = "database/books.db"
OUTPUT_DIR = Path("outputs")


def run_sql_queries():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(DATABASE_PATH)

    queries = {

        "Query 1 - SELECT WHERE": """
            SELECT
                title,
                price_gbp,
                rating,
                in_stock
            FROM books
            WHERE rating = 5
              AND in_stock = 1
        """,

        "Query 2 - ORDER BY LIMIT": """
            SELECT
                title,
                price_gbp,
                price_inr,
                rating
            FROM books
            ORDER BY price_gbp DESC
            LIMIT 10
        """,

        "Query 3 - DISTINCT": """
            SELECT DISTINCT
                category_name
            FROM categories
            ORDER BY category_name
        """,

        "Query 4 - BETWEEN": """
            SELECT
                title,
                price_gbp,
                price_inr,
                rating
            FROM books
            WHERE price_gbp BETWEEN 20 AND 40
            ORDER BY price_gbp
        """,

        "Query 5 - IN": """
            SELECT
                title,
                price_gbp,
                rating,
                category_id
            FROM books
            WHERE category_id IN (1, 2, 3)
            ORDER BY rating DESC
        """,

        "Query 6 - JOIN": """
            SELECT
                c.category_name,
                b.title,
                b.price_gbp,
                b.price_inr,
                b.rating,
                b.in_stock
            FROM books AS b
            JOIN categories AS c
                ON b.category_id = c.category_id
            ORDER BY
                b.rating DESC,
                b.price_gbp DESC
            LIMIT 10
        """
    }

    output_text = []

    for number, (name, query) in enumerate(
        queries.items(),
        start=1
    ):

        df = pd.read_sql(query, connection)

        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)

        print(df.to_string(index=False))

        output_file = OUTPUT_DIR / f"query_{number}.csv"

        df.to_csv(
            output_file,
            index=False
        )

        output_text.append(
            "\n"
            + "=" * 70
            + "\n"
            + name
            + "\n"
            + "=" * 70
            + "\n"
            + query.strip()
            + "\n\n"
            + df.to_string(index=False)
            + "\n"
        )

    with open(
        OUTPUT_DIR / "query_outputs.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(output_text)
        )

    connection.close()


def compare_sql_and_pandas_join():

    connection = sqlite3.connect(DATABASE_PATH)

    # ---------------------------------------------------------
    # SQL JOIN using pd.read_sql()
    # ---------------------------------------------------------

    sql_join = """
        SELECT
            c.category_name,
            b.title,
            b.price_gbp,
            b.price_inr,
            b.rating,
            b.in_stock
        FROM books AS b
        JOIN categories AS c
            ON b.category_id = c.category_id
        ORDER BY
            b.rating DESC,
            b.price_gbp DESC
        LIMIT 10
    """

    sql_df = pd.read_sql(
        sql_join,
        connection
    )

    # ---------------------------------------------------------
    # Read original tables separately
    # ---------------------------------------------------------

    books_df = pd.read_sql(
        """
        SELECT
            book_id,
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        FROM books
        """,
        connection
    )

    categories_df = pd.read_sql(
        """
        SELECT
            category_id,
            category_name
        FROM categories
        """,
        connection
    )

    connection.close()

    # ---------------------------------------------------------
    # pandas merge
    # ---------------------------------------------------------

    pandas_join = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner"
    )

    pandas_join = pandas_join[
        [
            "category_name",
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock"
        ]
    ]

    pandas_join = pandas_join.sort_values(
        by=["rating", "price_gbp"],
        ascending=[False, False]
    ).head(10)

    pandas_join = pandas_join.reset_index(drop=True)

    sql_df = sql_df.reset_index(drop=True)

    # ---------------------------------------------------------
    # Compare results
    # ---------------------------------------------------------

    comparison = sql_df.copy()

    comparison["pandas_match"] = (
        sql_df.astype(str).values
        == pandas_join.astype(str).values
    ).all(axis=1)

    print("\n" + "=" * 70)
    print("SQL JOIN RESULT")
    print("=" * 70)

    print(sql_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("PANDAS MERGE RESULT")
    print("=" * 70)

    print(pandas_join.to_string(index=False))

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(comparison.to_string(index=False))

    print(
        "\nAll rows equivalent:",
        comparison["pandas_match"].all()
    )

    comparison.to_csv(
        OUTPUT_DIR / "sql_vs_pandas_join.csv",
        index=False
    )


if __name__ == "__main__":

    run_sql_queries()

    compare_sql_and_pandas_join()
from pathlib import Path

from scrape import scrape_books
from clean import clean_books
from transform import add_inr_price
from database import load_data
from analysis import (
    run_sql_queries,
    compare_sql_and_pandas_join
)


RAW_FILE = Path("data/raw_books.csv")
CLEANED_FILE = Path("data/cleaned_books.csv")


def main():

    print("\n")
    print("=" * 70)
    print("ZEpto DATA PIPELINE")
    print("=" * 70)

    # ---------------------------------------------------------
    # STEP 1: SCRAPE
    # ---------------------------------------------------------

    print("\n[1/6] Scraping books...")

    raw_df = scrape_books()

    RAW_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    raw_df.to_csv(
        RAW_FILE,
        index=False
    )

    print(
        f"Raw records saved: {len(raw_df)}"
    )

    # ---------------------------------------------------------
    # STEP 2: CLEAN
    # ---------------------------------------------------------

    print("\n[2/6] Cleaning data...")

    cleaned_df = clean_books(raw_df)

    print(
        f"Cleaned records: {len(cleaned_df)}"
    )

    # ---------------------------------------------------------
    # STEP 3: CURRENCY CONVERSION
    # ---------------------------------------------------------

    print("\n[3/6] Converting GBP to INR...")

    transformed_df = add_inr_price(
        cleaned_df
    )

    transformed_df.to_csv(
        CLEANED_FILE,
        index=False
    )

    print(
        "Fixed conversion rate: "
        "1 GBP = 105.50 INR"
    )

    # ---------------------------------------------------------
    # STEP 4: DATABASE
    # ---------------------------------------------------------

    print("\n[4/6] Loading SQLite database...")

    load_data(transformed_df)

    # ---------------------------------------------------------
    # STEP 5: SQL QUERIES
    # ---------------------------------------------------------

    print("\n[5/6] Running SQL queries...")

    run_sql_queries()

    # ---------------------------------------------------------
    # STEP 6: PANDAS MERGE
    # ---------------------------------------------------------

    print("\n[6/6] Comparing SQL JOIN with pandas.merge()...")

    compare_sql_and_pandas_join()

    # ---------------------------------------------------------
    # COMPLETE
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nGenerated files:")
    print("- data/raw_books.csv")
    print("- data/cleaned_books.csv")
    print("- database/books.db")
    print("- outputs/query_outputs.txt")
    print("- outputs/query_1.csv ... query_6.csv")
    print("- outputs/sql_vs_pandas_join.csv")


if __name__ == "__main__":
    main()
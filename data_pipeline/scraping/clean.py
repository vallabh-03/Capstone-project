import pandas as pd


RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}


def clean_price(value):
    """
    Convert price text such as £51.77 into float 51.77.
    """

    if pd.isna(value):
        return None

    try:
        value = str(value).strip()
        value = value.replace("£", "").replace(",", "")

        return float(value)

    except (ValueError, TypeError):
        return None


def clean_rating(value):
    """
    Convert text rating into integer.
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    return RATING_MAP.get(value)


def clean_stock(value):
    """
    Convert availability text into Boolean.
    """

    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if "in stock" in value:
        return True

    if "out of stock" in value:
        return False

    return None


def clean_books(df):
    """
    Clean and validate scraped book data.
    """

    df = df.copy()

    # Clean numeric fields
    df["price_gbp"] = df["price"].apply(clean_price)
    df["rating"] = df["star_rating"].apply(clean_rating)

    # Clean availability
    df["in_stock"] = df["availability"].apply(clean_stock)

    # Median imputation for numeric fields
    price_median = df["price_gbp"].median()
    rating_median = df["rating"].median()

    if pd.isna(price_median):
        price_median = 0.0

    if pd.isna(rating_median):
        rating_median = 3

    df["price_gbp"] = df["price_gbp"].fillna(price_median)
    df["rating"] = df["rating"].fillna(round(rating_median))

    # Rows with missing non-numeric information are removed.
    before = len(df)

    df = df.dropna(
        subset=[
            "title",
            "category",
            "in_stock"
        ]
    )

    dropped_rows = before - len(df)

    print(f"Rows dropped because of unparseable required fields: {dropped_rows}")

    # Ensure correct data types
    df["price_gbp"] = df["price_gbp"].astype(float)
    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)

    # Keep only the required columns
    df = df[
        [
            "title",
            "price_gbp",
            "rating",
            "in_stock",
            "category"
        ]
    ]

    # Validate rating
    df = df[df["rating"].between(1, 5)]

    return df.reset_index(drop=True)


if __name__ == "__main__":

    input_file = "data/raw_books.csv"
    output_file = "data/cleaned_books.csv"

    df = pd.read_csv(input_file)

    cleaned_df = clean_books(df)

    cleaned_df.to_csv(output_file, index=False)

    print("\nCleaned data:")
    print(cleaned_df.head())

    print("\nData types:")
    print(cleaned_df.dtypes)

    print("\nTotal cleaned rows:", len(cleaned_df))
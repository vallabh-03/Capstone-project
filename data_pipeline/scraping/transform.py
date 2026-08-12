import pandas as pd


GBP_TO_INR = 105.50


def add_inr_price(df):
    """
    Convert GBP prices to INR using the project's
    required fixed conversion rate.
    """

    df = df.copy()

    df["price_inr"] = (
        df["price_gbp"] * GBP_TO_INR
    ).round(2)

    return df


if __name__ == "__main__":

    input_file = "data/cleaned_books.csv"

    df = pd.read_csv(input_file)

    df = add_inr_price(df)

    df.to_csv(input_file, index=False)

    print(df.head())

    print(
        f"\nConversion rate used: "
        f"1 GBP = {GBP_TO_INR} INR"
    )
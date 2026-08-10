import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path


BASE_URL = "https://books.toscrape.com/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DataPipelineProject/1.0)"
}


def get_soup(url):
    """
    Download a webpage and return its BeautifulSoup object.
    """
    response = requests.get(url, headers=HEADERS, timeout=15)

    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch {url}. "
            f"Status code: {response.status_code}"
        )

    return BeautifulSoup(response.text, "html.parser")


def get_categories():
    """
    Get at least three book categories from the website.
    """

    soup = get_soup(BASE_URL)

    categories = []

    category_links = soup.select(".side_categories ul li ul li a")

    for link in category_links:
        category_name = link.get_text(strip=True)
        category_url = link.get("href")

        if category_name and category_url:
            full_url = BASE_URL + category_url

            categories.append({
                "category": category_name,
                "url": full_url
            })

    if len(categories) < 3:
        raise Exception("Less than 3 categories were found.")

    # We only need at least 3 categories.
    return categories[:3]


def parse_book(book, category):
    """
    Extract raw fields from a book card.
    """

    title_tag = book.select_one("h3 a")
    price_tag = book.select_one(".price_color")
    rating_tag = book.select_one(".star-rating")
    availability_tag = book.select_one(".availability")

    title = title_tag.get("title") if title_tag else None
    price = price_tag.get_text(strip=True) if price_tag else None

    rating = None

    if rating_tag:
        classes = rating_tag.get("class", [])

        for rating_class in classes:
            if rating_class != "star-rating":
                rating = rating_class
                break

    availability = (
        availability_tag.get_text(" ", strip=True)
        if availability_tag
        else None
    )

    return {
        "title": title,
        "price": price,
        "star_rating": rating,
        "availability": availability,
        "category": category
    }


def scrape_category(category_name, category_url):
    """
    Scrape all pages for one category.
    """

    books = []

    page_url = category_url

    while page_url:

        print(f"Scraping: {page_url}")

        soup = get_soup(page_url)

        book_cards = soup.select("article.product_pod")

        for book in book_cards:
            book_data = parse_book(book, category_name)
            books.append(book_data)

        next_button = soup.select_one("li.next a")

        if next_button:
            next_url = next_button.get("href")

            current_page_url = page_url.rsplit("/", 1)[0] + "/"

            page_url = current_page_url + next_url
        else:
            page_url = None

    return books


def scrape_books():
    """
    Scrape books from at least three categories.
    """

    categories = get_categories()

    print("\nCategories selected:")
    for category in categories:
        print("-", category["category"])

    all_books = []

    for category in categories:

        category_books = scrape_category(
            category["category"],
            category["url"]
        )

        all_books.extend(category_books)

    df = pd.DataFrame(all_books)

    print("\nTotal books scraped:", len(df))
    print("Categories:", df["category"].nunique())

    if len(df) < 60:
        raise Exception(
            f"Only {len(df)} books were scraped. "
            "At least 60 books are required."
        )

    if df["category"].nunique() < 3:
        raise Exception(
            "Less than 3 categories were scraped."
        )

    return df


if __name__ == "__main__":

    output_path = Path("data/raw_books.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = scrape_books()

    df.to_csv(output_path, index=False)

    print(f"\nRaw data saved to: {output_path}")
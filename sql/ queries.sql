-- ============================================================
-- Query 1: SELECT + WHERE
-- List books with a rating of 5 that are currently in stock.
-- ============================================================

SELECT
    title,
    price_gbp,
    rating,
    in_stock
FROM books
WHERE rating = 5
  AND in_stock = 1;


-- ============================================================
-- Query 2: ORDER BY + LIMIT
-- Find the 10 most expensive books.
-- ============================================================

SELECT
    title,
    price_gbp,
    price_inr,
    rating
FROM books
ORDER BY price_gbp DESC
LIMIT 10;


-- ============================================================
-- Query 3: DISTINCT
-- List all available book categories.
-- ============================================================

SELECT DISTINCT
    category_name
FROM categories
ORDER BY category_name;


-- ============================================================
-- Query 4: BETWEEN
-- Find books priced between £20 and £40.
-- ============================================================

SELECT
    title,
    price_gbp,
    price_inr,
    rating
FROM books
WHERE price_gbp BETWEEN 20 AND 40
ORDER BY price_gbp;


-- ============================================================
-- Query 5: IN
-- Find books belonging to selected categories.
-- ============================================================

SELECT
    title,
    price_gbp,
    rating,
    category_id
FROM books
WHERE category_id IN (1, 2, 3)
ORDER BY rating DESC;


-- ============================================================
-- Query 6: JOIN
-- Display books together with their category names.
-- Top 10 highest-rated books per category.
-- ============================================================

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
LIMIT 10;
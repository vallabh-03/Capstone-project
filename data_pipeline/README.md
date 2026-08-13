# Data Pipeline

## Overview
The Data Pipeline module collects book catalog data, cleans and transforms it, enriches it, and prepares it for relational/SQL analysis.

## Structure
```text
data_pipeline/
├── scraping/
├── data/
│   ├── raw_books.csv
│   └── cleaned_books.csv
├── outputs/
├── sql/
└── README.md
```

## Technologies
- Python
- Requests
- BeautifulSoup
- Pandas
- SQL / relational database

## Pipeline
```text
Web Source → Scraping → Raw Data → Cleaning → Enrichment → Cleaned Data → SQL/Database → Analysis
```

## Main Tasks
- Scrape book catalog data from the required categories.
- Clean and transform the raw data.
- Perform the required baseline enrichment, including currency conversion where applicable.
- Store/process the cleaned data for relational and SQL analysis.
- Run SQL queries for analysis.

## Important Files
```text
data/raw_books.csv
data/cleaned_books.csv
sql/
outputs/
```

## Running
From the repository root, activate the virtual environment and run the scripts in `data_pipeline` in their intended order:
1. Run scraping.
2. Run cleaning/transformation.
3. Run enrichment/data preparation.
4. Run SQL/database steps.

## Design Decisions
Scraping, cleaning, storage, and SQL analysis are kept as separate stages so each stage can be tested independently. Raw data is preserved separately from cleaned data to keep the transformation process traceable.

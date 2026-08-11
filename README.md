# Capstone-project
A capstone project for showcasing my skills on Machine learning and python

# Module 1 — Data Pipeline

## Overview

This module implements a complete catalog data pipeline:

Scrape → Clean → Convert Currency → Normalize → Store → Query → Validate

The data source is Books to Scrape:

https://books.toscrape.com/

The website is a public scraping-practice website and does not require
authentication or an API key.

## Requirements

- Python 3.9+
- requests
- beautifulsoup4
- pandas
- sqlite3 

## Pipeline Status

The data pipeline has been successfully tested end-to-end, including scraping, cleaning, currency conversion, SQLite loading, SQL analysis, and pandas validation.

## Pipeline Validation

The pipeline was tested successfully from scraping through data cleaning, transformation, SQLite database loading, SQL querying, and pandas analysis.

## Repository Status

The Data Pipeline module is organized into separate components for scraping, cleaning, transformation, database operations, and analysis.

## Installation

From the repository root:

```bash
pip install -r requirements.txt
python scraping/run_pipeline.py


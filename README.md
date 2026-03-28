# Project 4 — Netflix Content Analysis 🎬

## What Is This Project?

An exploratory data analysis of a Netflix-style content library with 6,000 titles.
The focus is real-world data wrangling: parsing messy comma-separated genre
strings, handling non-standard date formats, extracting numbers from mixed
text columns, and presenting results in a visually striking dark dashboard
that matches Netflix brand identity.

**Difficulty:** Intermediate
**Time:** 3–4 days
**Tools:** Python, Pandas, Matplotlib, Seaborn, Collections, WordCloud (optional)
**No external dataset needed** — but swapping in the real Netflix CSV requires
zero code changes

---

## What Problem Does It Solve?

A content strategist at a streaming company wants to know:
- Are we a movies-first or shows-first platform?
- Which countries produce the most content on our platform?
- Which genres dominate our library?
- How has our content library grown over the years?
- What does a typical movie length look like?

This project answers all of those questions visually.

---

## What You Will Build

| File | Description |
|------|-------------|
| `netflix_dashboard.png` | 8-panel dark-themed Netflix dashboard |
| `netflix_wordcloud.png` | Genre word cloud sized by frequency (optional) |
| `netflix_clean.csv` | 6,000 rows of parsed and cleaned content data |

### The 8 Dashboard Panels

| Panel | Chart Type | What It Shows |
|-------|-----------|---------------|
| 1 | Donut chart | Movies vs TV Shows split |
| 2 | Horizontal bar | Top 10 content-producing countries |
| 3 | Bar chart | Content rating distribution (G, PG, R, TV-MA, etc.) |
| 4 | Line chart | Content added per year, split by type |
| 5 | Bar chart | Release decade distribution (1960s–2020s) |
| 6 | Horizontal bar | Top 10 genres by frequency |
| 7 | Histogram | Movie duration distribution with mean line |
| 8 | Bar chart | TV show seasons distribution |

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn
pip install wordcloud   # optional — for the genre word cloud panel

python netflix_analysis.py
```

---

## Full Step-by-Step Explanation

### Step 1 — Generate the Dataset

Creates 6,000 titles that match the exact format of the real Netflix Kaggle
dataset — so swapping in the real data requires only changing one line:

```python
# Synthetic data (what the script uses now)
df = pd.DataFrame({ ... })

# Real data (swap in with zero other changes)
# df = pd.read_csv("netflix_titles.csv")
```

The generated data includes:
- Genre strings like `"Dramas, Thrillers, International Movies"` — multiple
  values in one field, separated by commas
- Dates in the format `"January 14, 2020"` — not standard ISO format
- Duration strings like `"127 min"` or `"3 Seasons"` — numbers mixed with text
- Intentional NaN values in country and director columns

### Step 2 — Parse Genre Strings

This is the most important data wrangling step. Each title has multiple genres
in a single column separated by commas. The goal is to count how often each
genre appears across all titles.

```python
from collections import Counter

# Split every genre string and flatten into one big list
all_genres = []
for g in df["listed_in"]:
    all_genres.extend([x.strip() for x in g.split(",")])

# Count occurrences of each genre
genre_counts = Counter(all_genres)
# Result: {"Dramas": 1823, "Comedies": 1456, ...}
```

`Counter` is the standard Python tool for this pattern. It is faster and
cleaner than writing a manual loop with a dictionary.

### Step 3 — Parse Non-Standard Dates

The date format `"January 14, 2020"` is not in ISO format, so Pandas needs
an explicit format string to parse it:

```python
df["date_added_parsed"] = pd.to_datetime(
    df["date_added"],
    format="%B %d, %Y",   # %B = full month name, %d = day, %Y = 4-digit year
    errors="coerce"        # invalid dates become NaN instead of crashing
)

df["year_added"]  = df["date_added_parsed"].dt.year
df["month_added"] = df["date_added_parsed"].dt.month
```

Non-standard date formats are the most common data cleaning challenge in
real-world analyst work.

### Step 4 — Extract Numeric Duration

The duration column contains strings like `"127 min"` and `"3 Seasons"`.
Use regex to extract just the number:

```python
# str.extract() with a regex pattern pulls the first number from each string
df["duration_val"] = df["duration"].str.extract(r"(\d+)").astype(float)

# Now you can compute averages and plot distributions
avg_movie_length = df[df["type"] == "Movie"]["duration_val"].mean()
```

`r"(\d+)"` means: capture one or more digit characters. The parentheses
create a capture group that `str.extract()` returns as a column.

### Step 5 — Build the Netflix Dashboard

The entire figure uses a black `#141414` background with Netflix red `#E50914`
chart elements. This is intentional — it demonstrates brand awareness and
design thinking, not just technical ability.

```python
NETFLIX_RED = "#E50914"
DARK_BG     = "#141414"

fig = plt.figure(figsize=(22, 16), facecolor=DARK_BG)
fig.suptitle("Netflix Content Analysis", color=NETFLIX_RED, fontsize=26)
```

Every axis gets its own black background with red titles.

### Step 6 — Word Cloud (Optional)

If `wordcloud` is installed, a genre word cloud is generated where genres
that appear more frequently are displayed in larger text:

```python
from wordcloud import WordCloud

wc = WordCloud(
    width=800, height=400,
    background_color="black",
    colormap="Reds",
    max_words=80
).generate_from_frequencies(genre_counts)
```

### Step 7 — Monthly Seasonality Analysis

The final analysis checks which month Netflix adds the most content —
useful for understanding platform strategy:

```python
monthly_adds = df["month_added"].value_counts().sort_index()
peak_month   = monthly_adds.idxmax()
```

---

## Key Concepts Explained

| Concept | Plain English |
|---------|--------------|
| `str.split(",")` | Splits a comma-separated string into a Python list |
| `list.extend()` | Adds all items from one list into another — used to flatten genre lists |
| `Counter(list)` | Counts occurrences of each unique value — like `value_counts()` for lists |
| `str.extract(r"(\d+)")` | Regex: pulls the first number from each string in a Series |
| `pd.to_datetime(format=...)` | Parses dates written in non-standard formats |
| `dt.year`, `dt.month` | Extracts the year or month component from a datetime column |
| `groupby().size().unstack()` | Counts rows per two-column group, then pivots into wide format |
| `figure.facecolor` | Sets the background colour of the entire figure |
| `WordCloud.generate_from_frequencies()` | Sizes words by the provided count dictionary |

---

## Key Insights You Will Discover

- Movies make up roughly 68% of the library — Netflix is still primarily a
  movie platform despite the rise of original TV shows
- The United States produces the most content, followed by India and the UK
- TV-MA is the most common rating — the library skews toward adult audiences
- Content additions grew sharply between 2016 and 2019, then levelled off
- Most TV shows run for only 1–2 seasons
- Dramas and Comedies are by far the most common genres

---

## Bonus Challenges

1. Download the real **Netflix Movies and TV Shows dataset** from Kaggle
   (8,800 titles up to 2021) — the code works with zero changes
2. Add a **choropleth world map** using Plotly to show content by country
3. Analyse which directors appear across the most genres
4. Track how Netflix shifted from licensed content to originals (2015–2021)
5. Compare Netflix vs Disney Plus by finding both datasets on Kaggle

---

## Real Dataset

**Kaggle:** Search "Netflix Movies and TV Shows" by Shivamb
8,800 titles with the exact same column names and format as this project.
Replace the data generation block with:

```python
df = pd.read_csv("netflix_titles.csv")
```

Every other line of code works unchanged.

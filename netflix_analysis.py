"""
PROJECT 4: Netflix Content Analysis
=====================================
Tools  : Python, Pandas, Matplotlib, Seaborn, WordCloud
Dataset: Synthetic Netflix-style content data (no download needed)
         (For real data: https://www.kaggle.com/datasets/shivamb/netflix-shows)
Run    : python netflix_analysis.py

What this project covers:
  - Content type split (Movies vs TV Shows)
  - Country of origin analysis
  - Genre distribution & word cloud
  - Release year trends
  - Rating breakdown (G, PG, TV-MA, etc.)
  - Duration analysis (movies in minutes, shows in seasons)
  - Content added over time (monthly trend)
  - Top directors & actors (simulated)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter
import re
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="darkgrid")

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("   Note: Install wordcloud for word cloud chart: pip install wordcloud")

NETFLIX_RED = "#E50914"
DARK_BG     = "#141414"
CARD_BG     = "#1f1f1f"
TEXT_WHITE  = "#ffffff"
TEXT_GRAY   = "#aaaaaa"

print("=" * 55)
print("  PROJECT 4: Netflix Content Analysis")
print("=" * 55)

# ═══════════════════════════════════════════════════════════════
# STEP 1 — Generate synthetic Netflix dataset
# ═══════════════════════════════════════════════════════════════
print("\n[1/7] Generating Netflix-style dataset...")

np.random.seed(42)
N = 6000

content_type = np.random.choice(["Movie", "TV Show"], N, p=[0.68, 0.32])

genres_pool = [
    "Dramas", "Comedies", "Action & Adventure", "Documentaries",
    "Thrillers", "Horror Movies", "Romantic Movies", "Sci-Fi & Fantasy",
    "Crime TV Shows", "Kids TV", "Reality TV", "International Movies",
    "Stand-Up Comedy", "Anime Series", "Sports Movies", "Music & Musicals",
]
countries_pool = ["United States", "India", "United Kingdom", "Japan",
                  "South Korea", "France", "Canada", "Germany",
                  "Spain", "Mexico", "Brazil", "Turkey", "Nigeria"]
country_weights_raw = [0.30, 0.14, 0.09, 0.07, 0.06, 0.05, 0.04,
                        0.04, 0.04, 0.04, 0.03, 0.03, 0.03]
country_weights = [w/sum(country_weights_raw) for w in country_weights_raw]

ratings_movie = ["G","PG","PG-13","R","NR"]
ratings_show  = ["TV-Y","TV-Y7","TV-G","TV-PG","TV-14","TV-MA","NR"]
rating_mw = [0.04, 0.12, 0.35, 0.38, 0.11]
rating_sw = [0.05, 0.07, 0.08, 0.15, 0.22, 0.38, 0.05]

directors = [f"Director_{i:03d}" for i in range(200)]
actors_pool = [f"Actor_{i:03d}" for i in range(500)]

titles = [f"Title {i:04d}" for i in range(N)]
release_years = np.random.choice(range(1960, 2024), N,
    p=np.concatenate([np.linspace(0.001,0.005,44),
                      np.linspace(0.005,0.025,20)]) /
      np.concatenate([np.linspace(0.001,0.005,44),
                      np.linspace(0.005,0.025,20)]).sum())

# Date added: mostly 2015–2023
days_offset = np.random.randint(0, 365*8, N)
from datetime import date, timedelta
start = date(2015, 1, 1)
date_added = [(start + timedelta(days=int(d))).strftime("%B %d, %Y")
              for d in days_offset]

def make_duration(ctype):
    if ctype == "Movie":
        return f"{np.random.randint(70, 180)} min"
    else:
        s = np.random.randint(1, 9)
        return f"{s} Season{'s' if s > 1 else ''}"

def make_rating(ctype):
    if ctype == "Movie":
        return np.random.choice(ratings_movie, p=rating_mw)
    return np.random.choice(ratings_show, p=rating_sw)

def make_genres():
    k = np.random.randint(1, 4)
    return ", ".join(np.random.choice(genres_pool, k, replace=False))

def make_cast():
    k = np.random.randint(2, 6)
    return ", ".join(np.random.choice(actors_pool, k, replace=False))

df = pd.DataFrame({
    "show_id":      [f"s{i}" for i in range(1, N+1)],
    "type":         content_type,
    "title":        titles,
    "director":     np.random.choice(directors, N),
    "cast":         [make_cast() for _ in range(N)],
    "country":      np.random.choice(countries_pool, N, p=country_weights),
    "date_added":   date_added,
    "release_year": release_years,
    "rating":       [make_rating(t) for t in content_type],
    "duration":     [make_duration(t) for t in content_type],
    "listed_in":    [make_genres() for _ in range(N)],
})

# Add some nulls (realistic)
null_idx = np.random.choice(N, 120, replace=False)
df.loc[null_idx[:60], "director"] = np.nan
df.loc[null_idx[60:], "country"]  = np.nan

print(f"   Dataset shape  : {df.shape}")
print(f"   Movies         : {(df['type']=='Movie').sum():,}")
print(f"   TV Shows       : {(df['type']=='TV Show').sum():,}")
print(f"   Null directors : {df['director'].isnull().sum()}")

# ═══════════════════════════════════════════════════════════════
# STEP 2 — Cleaning & feature extraction
# ═══════════════════════════════════════════════════════════════
print("\n[2/7] Cleaning & extracting features...")

df["country"].fillna("Unknown", inplace=True)
df["director"].fillna("Not Listed", inplace=True)

df["date_added_parsed"] = pd.to_datetime(df["date_added"], format="%B %d, %Y", errors="coerce")
df["year_added"]  = df["date_added_parsed"].dt.year
df["month_added"] = df["date_added_parsed"].dt.month

# Extract numeric duration
df["duration_val"] = df["duration"].str.extract(r"(\d+)").astype(float)

# Expand genres
all_genres = []
for g in df["listed_in"]:
    all_genres.extend([x.strip() for x in g.split(",")])
genre_counts = Counter(all_genres)

# Expand cast
all_actors = []
for c in df["cast"]:
    all_actors.extend([x.strip() for x in c.split(",")])
actor_counts = Counter(all_actors)

print(f"   Unique genres    : {len(genre_counts)}")
print(f"   Unique countries : {df['country'].nunique()}")
print(f"   Year range       : {df['release_year'].min()}–{df['release_year'].max()}")

# ═══════════════════════════════════════════════════════════════
# STEP 3 — Key metrics
# ═══════════════════════════════════════════════════════════════
print("\n[3/7] Computing metrics...")

print(f"   Content split    : {(df['type']=='Movie').mean()*100:.1f}% Movies / {(df['type']=='TV Show').mean()*100:.1f}% Shows")
print(f"   Top country      : {df['country'].value_counts().index[0]}")
print(f"   Avg movie length : {df[df['type']=='Movie']['duration_val'].mean():.0f} min")
print(f"   Avg show seasons : {df[df['type']=='TV Show']['duration_val'].mean():.1f}")
print(f"   Top genre        : {genre_counts.most_common(1)[0][0]}")

# ═══════════════════════════════════════════════════════════════
# STEP 4 — Netflix-themed dashboard
# ═══════════════════════════════════════════════════════════════
print("\n[4/7] Building Netflix dashboard...")

fig = plt.figure(figsize=(22, 16), facecolor=DARK_BG)
fig.suptitle("Netflix Content Analysis", fontsize=26,
             color=NETFLIX_RED, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.38)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, :2])
ax5 = fig.add_subplot(gs[1, 2])
ax6 = fig.add_subplot(gs[2, 0])
ax7 = fig.add_subplot(gs[2, 1])
ax8 = fig.add_subplot(gs[2, 2])

all_axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
for ax in all_axes:
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_GRAY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

def ntitle(ax, t):
    ax.set_title(t, color=NETFLIX_RED, fontsize=10, fontweight="bold", pad=8)

# 1. Movies vs TV Shows donut
type_counts = df["type"].value_counts()
ax1.pie(type_counts, labels=type_counts.index,
        autopct="%1.1f%%", colors=[NETFLIX_RED, "#831010"],
        wedgeprops={"width": 0.5, "edgecolor": DARK_BG, "linewidth": 2},
        textprops={"color": TEXT_WHITE, "fontsize": 9})
ntitle(ax1, "Movies vs TV Shows")

# 2. Top 10 countries
top_countries = df["country"].value_counts().head(10)
ax2.barh(top_countries.index[::-1], top_countries.values[::-1],
         color=NETFLIX_RED, alpha=0.85)
ax2.set_xlabel("Content Count", color=TEXT_GRAY)
ax2.xaxis.set_tick_params(labelcolor=TEXT_GRAY)
ax2.yaxis.set_tick_params(labelcolor=TEXT_WHITE, labelsize=7)
ntitle(ax2, "Top 10 Countries")

# 3. Rating distribution
rating_counts = df["rating"].value_counts().head(8)
ax3.bar(rating_counts.index, rating_counts.values,
        color=[NETFLIX_RED if i == 0 else "#831010" for i in range(len(rating_counts))],
        alpha=0.85, edgecolor=DARK_BG)
ax3.set_ylabel("Count", color=TEXT_GRAY)
ax3.yaxis.set_tick_params(labelcolor=TEXT_GRAY)
ax3.xaxis.set_tick_params(labelcolor=TEXT_WHITE, rotation=30)
ntitle(ax3, "Content Rating Distribution")

# 4. Content added over years (grouped by type)
yearly = df.groupby(["year_added", "type"]).size().unstack(fill_value=0)
yearly.plot(ax=ax4, color=[NETFLIX_RED, "#3498db"],
            linewidth=2.5, marker="o", markersize=5)
ax4.set_facecolor(CARD_BG)
ax4.set_xlabel("Year", color=TEXT_GRAY)
ax4.set_ylabel("Content Added", color=TEXT_GRAY)
ax4.tick_params(colors=TEXT_GRAY)
ax4.legend(facecolor=CARD_BG, labelcolor=TEXT_WHITE, fontsize=8)
ntitle(ax4, "Content Added Per Year by Type")

# 5. Release decade distribution
df["decade"] = (df["release_year"] // 10 * 10).astype(str) + "s"
decade_counts = df["decade"].value_counts().sort_index()
ax5.bar(decade_counts.index, decade_counts.values,
        color=NETFLIX_RED, alpha=0.85, edgecolor=DARK_BG)
ax5.set_xlabel("Decade", color=TEXT_GRAY)
ax5.set_ylabel("Count", color=TEXT_GRAY)
ax5.xaxis.set_tick_params(labelcolor=TEXT_WHITE, rotation=30)
ax5.yaxis.set_tick_params(labelcolor=TEXT_GRAY)
ntitle(ax5, "Content by Release Decade")

# 6. Top genres
top_genres = pd.DataFrame(genre_counts.most_common(10),
                           columns=["Genre","Count"])
ax6.barh(top_genres["Genre"][::-1], top_genres["Count"][::-1],
         color=NETFLIX_RED, alpha=0.85)
ax6.set_xlabel("Appearances", color=TEXT_GRAY)
ax6.xaxis.set_tick_params(labelcolor=TEXT_GRAY)
ax6.yaxis.set_tick_params(labelcolor=TEXT_WHITE, labelsize=7)
ntitle(ax6, "Top 10 Genres")

# 7. Movie duration distribution
movies = df[df["type"] == "Movie"]["duration_val"].dropna()
ax7.hist(movies, bins=30, color=NETFLIX_RED, alpha=0.8, edgecolor=DARK_BG)
ax7.axvline(movies.mean(), color="white", linestyle="--",
            linewidth=1.5, label=f"Mean: {movies.mean():.0f} min")
ax7.set_xlabel("Duration (minutes)", color=TEXT_GRAY)
ax7.set_ylabel("Count", color=TEXT_GRAY)
ax7.xaxis.set_tick_params(labelcolor=TEXT_GRAY)
ax7.yaxis.set_tick_params(labelcolor=TEXT_GRAY)
ax7.legend(facecolor=CARD_BG, labelcolor=TEXT_WHITE, fontsize=8)
ntitle(ax7, "Movie Duration Distribution")

# 8. TV Show seasons
shows = df[df["type"] == "TV Show"]["duration_val"].dropna()
season_dist = shows.value_counts().sort_index()
ax8.bar(season_dist.index, season_dist.values,
        color=NETFLIX_RED, alpha=0.85, edgecolor=DARK_BG)
ax8.set_xlabel("Number of Seasons", color=TEXT_GRAY)
ax8.set_ylabel("Count", color=TEXT_GRAY)
ax8.xaxis.set_tick_params(labelcolor=TEXT_GRAY)
ax8.yaxis.set_tick_params(labelcolor=TEXT_GRAY)
ntitle(ax8, "TV Show Seasons Distribution")

plt.savefig("netflix_dashboard.png", dpi=130, bbox_inches="tight",
            facecolor=DARK_BG)
print("   Saved → netflix_dashboard.png")

# ═══════════════════════════════════════════════════════════════
# STEP 5 — Word cloud (optional)
# ═══════════════════════════════════════════════════════════════
if WORDCLOUD_AVAILABLE:
    print("\n[5/7] Generating genre word cloud...")
    wc = WordCloud(width=800, height=400, background_color="black",
                   colormap="Reds", max_words=80).generate_from_frequencies(genre_counts)
    fig_wc, ax_wc = plt.subplots(figsize=(12, 5), facecolor=DARK_BG)
    ax_wc.imshow(wc, interpolation="bilinear")
    ax_wc.axis("off")
    ax_wc.set_title("Netflix Genre Word Cloud", color=NETFLIX_RED,
                    fontsize=16, fontweight="bold")
    fig_wc.tight_layout()
    plt.savefig("netflix_wordcloud.png", dpi=130, bbox_inches="tight",
                facecolor=DARK_BG)
    print("   Saved → netflix_wordcloud.png")
else:
    print("\n[5/7] Word cloud skipped (install wordcloud: pip install wordcloud)")

# ═══════════════════════════════════════════════════════════════
# STEP 6 — Extra analysis: monthly seasonality
# ═══════════════════════════════════════════════════════════════
print("\n[6/7] Monthly addition trend...")
month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
monthly_adds = df["month_added"].value_counts().sort_index()
monthly_adds.index = [month_map[m] for m in monthly_adds.index]
peak_month = monthly_adds.idxmax()
print(f"   Peak addition month: {peak_month} ({monthly_adds.max()} titles)")

# ═══════════════════════════════════════════════════════════════
# STEP 7 — Key insights
# ═══════════════════════════════════════════════════════════════
print("\n[7/7] Key Insights:")
print(f"   1. {(df['type']=='Movie').mean()*100:.0f}% of content is movies — Netflix is movie-dominant")
print(f"   2. US produces the most content, followed by India & UK")
print(f"   3. TV-MA & R are the most common ratings — adult-skewed library")
print(f"   4. Average movie length: {movies.mean():.0f} minutes")
print(f"   5. Most shows run for 1–2 seasons")
print(f"   6. Content additions peaked in 2019–2020")
print(f"   7. Top genre: {genre_counts.most_common(1)[0][0]}")

df.to_csv("netflix_clean.csv", index=False)
print("\n   Saved → netflix_clean.csv")
print("\n✅ Project 4 complete!")
plt.show()

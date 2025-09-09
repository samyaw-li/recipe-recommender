import streamlit as st
import pandas as pd
import numpy as np
import urllib.parse
import re, ast
from pathlib import Path
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Global tools/constants
lemmatizer = WordNetLemmatizer()
units = {"cup","cups","tablespoon","tablespoons","tbsp",
         "teaspoon","teaspoons","tsp","gram","grams","kg","ml","oz"}


# Helper functions

def normalize_tokens(str_list):
    cleaned = []
    for text in str_list:
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text) 
        words = [lemmatizer.lemmatize(w) for w in text.split() if w not in units]
        if words:
            cleaned.extend(words)
    return cleaned

def overlap_count(pantry, recipe_ing):
    return len(set(pantry) & set(recipe_ing))


# Data Loading + Preprocessing

DATA = Path("data")

@st.cache_resource
def load_data():
    recipes = pd.read_csv(DATA / "RAW_recipes.csv")
    recipes["ingredients_clean"] = recipes["ingredients"].apply(ast.literal_eval).apply(normalize_tokens)
    recipes["ingredients_str"] = recipes["ingredients_clean"].apply(lambda lst: " ".join(lst))
    recipes["tags_list"] = recipes["tags"].apply(ast.literal_eval)
    recipes["steps_list"] = recipes["steps"].apply(ast.literal_eval)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(recipes["ingredients_str"])
    return recipes, vectorizer, tfidf_matrix

recipes, vectorizer, tfidf_matrix = load_data()


# Recipe recommendation
def recommend_recipes(pantry, top_k=5, min_overlap=2, max_minutes=None, include_tags=None, exclude_tags=None, sort_by="similarity"):
    pantry_clean = normalize_tokens(pantry)
    pantry_str = " ".join(pantry_clean)
    pantry_vec = vectorizer.transform([pantry_str])
    sims = cosine_similarity(pantry_vec, tfidf_matrix)[0]
    sorted_idx = sims.argsort()[::-1]

    results = []
    for idx in sorted_idx:
        recipe = recipes.iloc[idx]
        recipe_ing = recipe["ingredients_clean"]
        recipe_tags = recipe["tags_list"]
        overlap = set(pantry_clean) & set(recipe_ing)

        if overlap_count(pantry_clean, recipe_ing) < min_overlap:
            continue
        if max_minutes and recipe["minutes"] > max_minutes:
            continue
        if include_tags and not any(tag in recipe_tags for tag in include_tags):
            continue
        if exclude_tags and any(tag in recipe_tags for tag in exclude_tags):
            continue

        results.append({
            "name": recipe["name"],
            "minutes": recipe["minutes"],
            "ingredients": recipe_ing,
            "similarity": round(float(sims[idx]), 3),
            "tags": ", ".join(recipe_tags[:5]),
            "matches": ", ".join(overlap)
        })

        if len(results) >= top_k:
            break
    
    df = pd.DataFrame(results)
    if sort_by == "similarity":
        df = df.sort_values(by = "similarity", ascending = False)
    elif sort_by == "minutes":
        df = df.sort_values(by = "minutes", ascending = True)
    elif sort_by == "matches":
        df["match_count"] = df["matches"].apply(lambda x: len(x.split(", ")))
        df = df.sort_values(by="match_count", ascending=False)
    return df


# Streamlit UI

st.title("🍲 Recipe Recommender")
st.write("Find recipes based on what’s in your pantry.")

# Pantry input
pantry_input = st.text_area("Enter your pantry items (comma separated)", "onion, rice, chicken")
pantry = [item.strip() for item in pantry_input.split(",") if item.strip()]

# Filters
max_minutes = st.slider("Maximum cook time (minutes)", 0, 240, 60)
min_overlap = st.slider("Minimum ingredient matches", 1, 5, 2)
top_k = st.slider("Number of recipes to return", 1, 20, 5)

# Tag filters
tag_counts = Counter(tag for tags in recipes["tags_list"] for tag in tags)
top_tags = [tag for tag, _ in tag_counts.most_common(100)]
include_tags = st.multiselect("Include recipes with these tags:", top_tags)
exclude_tags = st.multiselect("Exclude recipes with these tags:", top_tags)

#Sorting
sort_by = st.selectbox(
    "Sort results by:",
    ["similarity", "minutes", "matches"]
)

# Run recommender
if st.button("Recommend Recipes"):
    recommendations = recommend_recipes(
        pantry=pantry,
        top_k=top_k,
        min_overlap=min_overlap,
        max_minutes=max_minutes,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
        sort_by = sort_by
    )

    if not recommendations.empty:

        st.write(f"✅ Found {len(recommendations)} recipes")

        for _, row in recommendations.iterrows():
            st.subheader(f"{row['name']} ({row['minutes']} min)")
            st.write(f"**Similarity:** {row['similarity']} | **Matches:** {row['matches']}")
            st.write(f"**Tags:** {row['tags']}")
            st.write("**Ingredients:** " + ", ".join(row["ingredients"]))

            # 🔽 Expandable steps
            with st.expander("Show Steps"):
                steps = recipes.loc[recipes["name"] == row["name"], "steps_list"].values[0]
                for i, step in enumerate(steps, 1):
                    st.write(f"{i}. {step}")

            # 🔗 Google search link
            query = urllib.parse.quote(row["name"] + " recipe")
            url = f"https://www.google.com/search?q={query}"
            st.markdown(f"[🔍 Search Online: {url}]")

            st.markdown("---")

    else:
        st.write("⚠️ No recipes found. Try adjusting filters or adding more ingredients.")

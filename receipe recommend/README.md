# 🌶️ Masala Match — Smart Recipe & Nutrition Recommender

A Streamlit app built from your `recipeex001.csv` dataset and the logic in your
`smart_receipe_and_nutritional_rec` notebook (TF-IDF + cosine similarity recommender,
fuzzy-matched search, ingredient substitutes, and nutrition-goal filtering).

## What's inside

- **🔍 Find Similar Recipes** — type any recipe name (typos okay, powered by RapidFuzz)
  and get the top-N most similar recipes based on TF-IDF over name + cuisine +
  ingredients + instructions.
- **🔄 Ingredient Substitutes** — look up a recipe and see healthier swaps for its
  ingredients (e.g. paneer ➜ tofu, maida ➜ whole wheat flour).
- **🥗 Nutrition Goals** — filter the whole dataset by goal: High Protein, High Fiber,
  Low Carb, or Healthy Fat.
- **📖 Browse All** — search/filter the full recipe collection by cuisine or keyword.

## Run it locally

```bash
cd recipe_app
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Notes on changes from the notebook

- Instead of precomputing one giant NxN similarity matrix (which used a lot of memory
  for 5,938 recipes × 5,938 recipes), the app computes cosine similarity on-demand
  between the query recipe and all others. Same math, much lighter on memory, and it
  keeps the app fast to start.
- The recipe data and TF-IDF model are cached (`@st.cache_data` / `@st.cache_resource`)
  so they're only built once per session, not on every click.
- All three notebook features (recommend, suggest_substitutes, nutrition_filter) are
  preserved as-is, just wired up to a UI instead of `print()` statements.

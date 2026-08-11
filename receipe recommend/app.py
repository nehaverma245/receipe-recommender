import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz
import os

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Masala Match | Smart Recipe Recommender",
    page_icon="🌶️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# THEME / STYLING
# ----------------------------------------------------------------------------
# Palette pulled from an Indian spice rack:
#   Turmeric   #E0A458   (primary accent)
#   Chilli     #C1440E   (secondary / strong accent)
#   Cumin      #4A3728   (deep brown text)
#   Curry leaf #4B6043   (green, success/tag)
#   Cream      #FBF3E7   (background)
#   Paper      #FFFFFF

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Karla:wght@400;500;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Karla', sans-serif;
}

.stApp {
    background-color: #FBF3E7;
}

h1, h2, h3 {
    font-family: 'Fraunces', serif !important;
    color: #4A3728 !important;
    letter-spacing: -0.01em;
}

/* Hero banner */
.hero {
    background: linear-gradient(120deg, #C1440E 0%, #E0A458 100%);
    padding: 2.2rem 2.4rem;
    border-radius: 18px;
    color: #FBF3E7;
    margin-bottom: 1.6rem;
    box-shadow: 0 8px 24px rgba(193, 68, 14, 0.18);
}
.hero h1 {
    color: #FBF3E7 !important;
    font-size: 2.4rem;
    margin-bottom: 0.3rem;
}
.hero p {
    color: #FBEADD;
    font-size: 1.05rem;
    margin: 0;
}

/* Recipe card */
.recipe-card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
    border: 1px solid #EDE0CC;
    box-shadow: 0 2px 10px rgba(74, 55, 40, 0.06);
}
.recipe-card h4 {
    font-family: 'Fraunces', serif;
    color: #C1440E;
    margin-bottom: 0.2rem;
    font-size: 1.25rem;
}
.tag {
    display: inline-block;
    background: #F1E6D3;
    color: #4A3728;
    border-radius: 999px;
    padding: 0.15rem 0.7rem;
    font-size: 0.78rem;
    margin-right: 0.4rem;
    font-weight: 700;
}
.tag-green {
    background: #E4EADF;
    color: #4B6043;
}
.tag-score {
    background: #4A3728;
    color: #FBF3E7;
}
.sub-arrow {
    color: #C1440E;
    font-weight: 700;
}
hr.spice-divider {
    border: none;
    border-top: 2px dashed #E0A458;
    margin: 1.4rem 0;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #F1E6D3;
    border-radius: 10px 10px 0 0;
    padding: 8px 18px;
    font-weight: 700;
    color: #4A3728;
}
.stTabs [aria-selected="true"] {
    background-color: #C1440E !important;
    color: #FBF3E7 !important;
}
[data-testid="stSidebar"] {
    background-color: #F1E6D3;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DATA_PATH = os.path.join(os.path.dirname(__file__), "recipeex001.csv")

# ----------------------------------------------------------------------------
# DATA LOADING + FEATURE ENGINEERING (mirrors the notebook's cleaning steps)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path):
    df = pd.read_csv(path)
    df = df.fillna("")
    keep_cols = [
        "Name", "Cuisine", "Ingredients", "TranslatedInstructions",
        "TotalTimeInMins", "Ingredient-count",
    ]
    # img_url may or may not exist depending on the source file
    if "img_url" in df.columns:
        keep_cols.append("img_url")
    df = df[keep_cols].copy()

    df["Name_display"] = df["Name"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str).str.lower().str.strip()
    df["Cuisine_display"] = df["Cuisine"].astype(str).str.strip()
    df["Cuisine"] = df["Cuisine"].astype(str).str.lower()
    df["Ingredients"] = df["Ingredients"].astype(str).str.lower()
    df["TranslatedInstructions"] = df["TranslatedInstructions"].astype(str).str.lower()

    df["combined_features"] = (
        df["Name"] + " " + df["Cuisine"] + " " + df["Ingredients"] + " " + df["TranslatedInstructions"]
    )
    df = df.reset_index(drop=True)
    return df


@st.cache_resource(show_spinner=False)
def build_model(df):
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    recipe_vectors = tfidf.fit_transform(df["combined_features"])
    return tfidf, recipe_vectors


df = load_data(DATA_PATH)
tfidf, recipe_vectors = build_model(df)
recipe_index = pd.Series(df.index, index=df["Name"]).drop_duplicates()
all_names = df["Name"].tolist()

# ----------------------------------------------------------------------------
# INGREDIENT SUBSTITUTES DICTIONARY (from the notebook)
# ----------------------------------------------------------------------------
ingredient_substitutes = {
    "sugar": "Jaggery", "brown sugar": "Coconut Sugar", "powdered sugar": "Stevia",
    "honey": "Dates Syrup", "corn syrup": "Maple Syrup",
    "salt": "Pink Himalayan Salt", "table salt": "Rock Salt", "black salt": "Pink Himalayan Salt",
    "butter": "Olive Oil", "ghee": "Cold-Pressed Coconut Oil", "vegetable oil": "Olive Oil",
    "refined oil": "Cold-Pressed Mustard Oil", "sunflower oil": "Olive Oil",
    "soybean oil": "Sesame Oil", "palm oil": "Avocado Oil", "oil": "Cold-Pressed Mustard Oil",
    "milk": "Almond Milk", "whole milk": "Oat Milk", "cream": "Greek Yogurt",
    "fresh cream": "Hung Curd", "cheese": "Low Fat Cheese", "paneer": "Tofu",
    "condensed milk": "Coconut Milk",
    "maida": "Whole Wheat Flour", "all purpose flour": "Whole Wheat Flour",
    "refined flour": "Millet Flour", "corn flour": "Oat Flour", "bread crumbs": "Oats Powder",
    "white rice": "Brown Rice", "rice": "Brown Rice", "basmati rice": "Quinoa",
    "poha": "Red Rice Poha", "semolina": "Millet Rava",
    "white bread": "Whole Wheat Bread", "pasta": "Whole Wheat Pasta",
    "noodles": "Millet Noodles", "vermicelli": "Whole Wheat Vermicelli",
    "potato": "Sweet Potato", "fried potato": "Boiled Sweet Potato", "yam": "Pumpkin",
    "tofu": "Tempeh", "egg": "Chickpea Flour", "chicken": "Soya Chunks",
    "mutton": "Mushrooms", "fish": "Tofu",
    "mayonnaise": "Greek Yogurt", "cream cheese": "Hung Curd", "whipping cream": "Cashew Cream",
    "chips": "Roasted Makhana", "nachos": "Baked Multigrain Chips", "croutons": "Roasted Chickpeas",
    "milk chocolate": "Dark Chocolate", "chocolate syrup": "Cocoa Powder",
    "peanuts": "Almonds", "cashews": "Walnuts", "raisins": "Dates",
    "banana": "Apple", "mango": "Papaya",
    "jam": "Fruit Compote", "peanut butter": "Almond Butter",
    "red chilli powder": "Kashmiri Red Chilli", "garam masala": "Homemade Spice Mix",
    "black pepper": "White Pepper", "desiccated coconut": "Fresh Coconut",
    "rajma": "Black Beans", "chickpeas": "Green Gram", "black gram": "Horse Gram",
    "vinegar": "Lemon Juice", "soy sauce": "Coconut Aminos", "tomato ketchup": "Homemade Tomato Puree",
    "msg": "Natural Herbs", "food color": "Turmeric", "gelatin": "Agar Agar",
    "custard powder": "Cornstarch", "ice cream": "Frozen Yogurt",
    "soft drink": "Fresh Coconut Water", "energy drink": "Lemon Honey Water",
}

# ----------------------------------------------------------------------------
# NUTRITION KEYWORDS (from the notebook)
# ----------------------------------------------------------------------------
nutrition_keywords = {
    "High Protein": ["paneer", "tofu", "soy", "lentil", "dal", "rajma", "chickpea", "egg", "chicken", "fish"],
    "High Fiber": ["spinach", "broccoli", "beans", "oats", "brown rice", "apple", "carrot"],
    "Low Carb": ["cauliflower", "spinach", "cabbage", "mushroom", "paneer"],
    "Healthy Fat": ["almond", "walnut", "olive oil", "avocado", "flaxseed"],
}

# ----------------------------------------------------------------------------
# CORE LOGIC (ported from the notebook, adapted for the app)
# ----------------------------------------------------------------------------
def find_best_match(query):
    """Returns (matched_lowercase_name, score) or (None, 0)."""
    query = query.lower().strip()
    if query in recipe_index:
        return query, 100.0
    match = process.extractOne(query, all_names, scorer=fuzz.WRatio)
    if not match or match[1] < 70:
        return None, 0
    return match[0], match[1]


def get_recommendations(recipe_name, top_n=5):
    matched_name, score = find_best_match(recipe_name)
    if matched_name is None:
        return None, None, []

    idx = recipe_index[matched_name]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    query_vec = recipe_vectors[idx]
    sims = cosine_similarity(query_vec, recipe_vectors).flatten()
    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    ranked = [r for r in ranked if r[0] != idx][:top_n]

    results = []
    for i, s in ranked:
        row = df.iloc[i]
        results.append({
            "name": row["Name_display"],
            "cuisine": row["Cuisine_display"],
            "time": row["TotalTimeInMins"],
            "ingredients": row["Ingredients"],
            "instructions": row["TranslatedInstructions"],
            "img_url": row.get("img_url", ""),
            "score": s,
        })
    return matched_name, score, results


def get_substitutes(recipe_name):
    matched_name, score = find_best_match(recipe_name)
    if matched_name is None:
        return None, None, [], ""

    recipe_row = df[df["Name"] == matched_name].iloc[0]
    ingredients = recipe_row["Ingredients"]
    subs = [
        (ing.title(), sub) for ing, sub in ingredient_substitutes.items() if ing in ingredients
    ]
    return matched_name, score, subs, recipe_row["Name_display"]


def filter_by_nutrition(goal, top_n=10):
    keywords = nutrition_keywords.get(goal, [])
    scored = []
    for _, row in df.iterrows():
        ingredients = row["Ingredients"]
        score = sum(1 for kw in keywords if kw in ingredients)
        if score > 0:
            scored.append((row, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def render_recipe_card(name, cuisine, time_mins, ingredients, instructions, img_url=None, badge=None):
    col_img, col_txt = st.columns([1, 3]) if img_url else (None, None)
    with st.container():
        st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
        inner_img, inner_txt = st.columns([1, 3]) if img_url else (st.container(), st.container())
        if img_url:
            with inner_img:
                try:
                    st.image(img_url, use_container_width=True)
                except Exception:
                    pass
            with inner_txt:
                _render_card_text(name, cuisine, time_mins, ingredients, instructions, badge)
        else:
            _render_card_text(name, cuisine, time_mins, ingredients, instructions, badge)
        st.markdown('</div>', unsafe_allow_html=True)


def _render_card_text(name, cuisine, time_mins, ingredients, instructions, badge):
    badge_html = f'<span class="tag tag-score">{badge}</span>' if badge else ""
    st.markdown(f"#### {name.title()} {badge_html}", unsafe_allow_html=True)
    st.markdown(
        f'<span class="tag">{cuisine.title() if cuisine else "Unknown Cuisine"}</span>'
        f'<span class="tag tag-green">⏱ {time_mins} mins</span>',
        unsafe_allow_html=True,
    )
    with st.expander("Ingredients & Instructions"):
        st.markdown(f"**Ingredients:** {ingredients.capitalize()}")
        st.markdown(f"**Instructions:** {instructions.capitalize()[:900]}{'…' if len(instructions) > 900 else ''}")


# ----------------------------------------------------------------------------
# HERO
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🌶️ Masala Match</h1>
        <p>Your smart recipe & nutrition companion — find similar dishes, healthier swaps,
        and meals that match your goals, from a kitchen of {} recipes.</p>
    </div>
    """.format(len(df)),
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🍛 About")
    st.write(
        "Masala Match uses TF-IDF + cosine similarity over recipe names, "
        "cuisines, ingredients, and instructions to recommend similar dishes, "
        "along with fuzzy matching so typos don't get in your way."
    )
    st.markdown("---")
    st.markdown("### 📊 Dataset Snapshot")
    st.metric("Total Recipes", len(df))
    st.metric("Cuisines", df["Cuisine_display"].nunique())
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, scikit-learn & RapidFuzz")

# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Find Similar Recipes", "🔄 Ingredient Substitutes", "🥗 Nutrition Goals", "📖 Browse All",
])

# --- TAB 1: RECOMMENDER ------------------------------------------------------
with tab1:
    st.markdown("### Discover recipes similar to one you love")
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Enter a recipe name", placeholder="e.g. Masala Karela Recipe, Palak Paneer...",
            key="reco_query",
        )
    with col2:
        top_n = st.slider("Results", min_value=3, max_value=15, value=5, key="reco_topn")

    if st.button("Find Similar Recipes 🌶️", type="primary", key="reco_btn"):
        if not query.strip():
            st.warning("Please enter a recipe name.")
        else:
            matched_name, score, results = get_recommendations(query, top_n)
            if matched_name is None:
                st.error("❌ No close match found. Try a different recipe name.")
            else:
                if score < 100:
                    st.info(f"Using closest match: **{matched_name.title()}** ({score:.1f}% match)")
                st.markdown(f"#### Recipes similar to *{matched_name.title()}*")
                st.markdown('<hr class="spice-divider">', unsafe_allow_html=True)
                for r in results:
                    render_recipe_card(
                        r["name"], r["cuisine"], r["time"], r["ingredients"], r["instructions"],
                        img_url=r["img_url"] if r["img_url"] else None,
                        badge=f"{r['score']*100:.1f}% match",
                    )

# --- TAB 2: SUBSTITUTES -------------------------------------------------------
with tab2:
    st.markdown("### Find healthier ingredient swaps for a recipe")
    sub_query = st.text_input(
        "Enter a recipe name", placeholder="e.g. Palak Paneer, Masala Karela Recipe...",
        key="sub_query",
    )
    if st.button("Suggest Substitutes 🔄", type="primary", key="sub_btn"):
        if not sub_query.strip():
            st.warning("Please enter a recipe name.")
        else:
            matched_name, score, subs, display_name = get_substitutes(sub_query)
            if matched_name is None:
                st.error("❌ No close match found. Try a different recipe name.")
            else:
                if score < 100:
                    st.info(f"Using closest match: **{display_name}** ({score:.1f}% match)")
                st.markdown(f"#### Substitutions for *{display_name}*")
                if not subs:
                    st.write("No substitutions available for the ingredients in this recipe.")
                else:
                    for ing, sub in subs:
                        st.markdown(
                            f'<div class="recipe-card">{ing} <span class="sub-arrow">➜</span> {sub}</div>',
                            unsafe_allow_html=True,
                        )

# --- TAB 3: NUTRITION FILTER --------------------------------------------------
with tab3:
    st.markdown("### Find recipes that match your nutrition goal")
    goal = st.selectbox("Choose a goal", list(nutrition_keywords.keys()), key="nutrition_goal")
    nutrition_topn = st.slider("Results", min_value=5, max_value=30, value=10, key="nutrition_topn")

    if st.button("Filter Recipes 🥗", type="primary", key="nutrition_btn"):
        results = filter_by_nutrition(goal, nutrition_topn)
        if not results:
            st.error("No recipes found for this goal.")
        else:
            st.markdown(f"#### Top recipes for **{goal}**")
            st.caption(f"Matched on keywords: {', '.join(nutrition_keywords[goal])}")
            st.markdown('<hr class="spice-divider">', unsafe_allow_html=True)
            for row, score in results:
                render_recipe_card(
                    row["Name_display"], row["Cuisine_display"], row["TotalTimeInMins"],
                    row["Ingredients"], row["TranslatedInstructions"],
                    img_url=row.get("img_url", "") if row.get("img_url", "") else None,
                    badge=f"{score} keyword match{'es' if score != 1 else ''}",
                )

# --- TAB 4: BROWSE -------------------------------------------------------------
with tab4:
    st.markdown("### Browse the full recipe collection")
    colf1, colf2 = st.columns(2)
    with colf1:
        cuisine_filter = st.selectbox(
            "Filter by cuisine", ["All"] + sorted(df["Cuisine_display"].unique().tolist()),
            key="browse_cuisine",
        )
    with colf2:
        search_text = st.text_input("Search by name or ingredient", key="browse_search")

    filtered = df
    if cuisine_filter != "All":
        filtered = filtered[filtered["Cuisine_display"] == cuisine_filter]
    if search_text.strip():
        s = search_text.lower().strip()
        filtered = filtered[
            filtered["Name"].str.contains(s, na=False) | filtered["Ingredients"].str.contains(s, na=False)
        ]

    st.caption(f"Showing {min(len(filtered), 50)} of {len(filtered)} matching recipes")
    for _, row in filtered.head(50).iterrows():
        render_recipe_card(
            row["Name_display"], row["Cuisine_display"], row["TotalTimeInMins"],
            row["Ingredients"], row["TranslatedInstructions"],
            img_url=row.get("img_url", "") if row.get("img_url", "") else None,
        )

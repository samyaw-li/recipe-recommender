# 🍲 Recipe Recommender System

A **Streamlit app** that suggests recipes based on the ingredients you already have in your pantry.  
Built using the [Food.com Recipes & Interactions Dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions/data).

---

## 🚀 Features
- Ingredient-based recipe recommendations  
- Interactive filters:
  - Maximum cook time
  - Minimum ingredient overlap
  - Include / exclude tags
  - Sort by similarity, cooking time, or ingredient matches
- Expandable recipe steps (click to reveal)  
- Quick Google search links for the same recipe online

---

## 📊 How It Works
1. **Ingredient normalization**  
   - Lowercases, removes units and numbers, and lemmatizes words  
   - Example: `"1 Cup Potatoes"` → `"potato"`

2. **TF–IDF Representation**  
   - Each recipe’s ingredient list is transformed into a weighted vector  
   - Captures importance of rarer ingredients

3. **Cosine Similarity**  
   - Compares user pantry vector with all recipes  
   - Ranks recipes by similarity to what you have

4. **Filters & Metadata**  
   - Cooking time, tags, and ingredient overlap ensure relevant results

---

## 📸 Demo

![App Demo](assets/recipe-recommender-demo.gif)

---

## 🛠️ Built With
- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [NLTK](https://www.nltk.org/)
- [pandas](https://pandas.pydata.org/)

---

## 🖥️ Running the App

### 1. Clone the repo and install dependencies
```bash
git clone https://github.com/samyaw-li/recipe-recommender.git
cd recipe-recommender
pip install -r requirements.txt

### 2. Run the app
```bash
streamlit run app.py
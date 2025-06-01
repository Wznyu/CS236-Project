import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import base64

# Path to your SQLite database
DB_PATH = "final_results.db"

# List of available tables
TABLES = [
    "review_sentiment", "df_price_rating", "review_length_summary",
    "df_recommend_sentiment", "brand_stats", "category_stats",
    "monthly_reviews", "yearly_reviews"
]

TABLE_DESCRIPTIONS = {
    "review_sentiment": "This table evaluates customer feedback text and compares these sentiment scores with the numerical ratings – this approach effectively reveals which brands consistently deliver positive experiences while flagging potential reputation concerns where sentiment and ratings don't align.",
    "df_price_rating": "This table evaluates whether higher-priced products tend to receive better ratings, providing insight into the relationship between price and perceived value.",
    "monthly_reviews": "This table helps identify monthly trends in customer engagement and satisfaction, monitor product popularity, and support business decisions such as inventory planning, marketing strategies, and product development.",
    "yearly_reviews": "This table helps identify yearly trends in customer engagement and satisfaction, monitor product popularity, and support business decisions such as inventory planning, marketing strategies, and product development.",
    "review_length_summary": "This table helps us understand whether longer reviews tend to receive more community engagement, as reflected by vote counts.",
    "brand_stats": "This table evaluates brand-level performance by calculating the average rating and average price of all products associated with each brand. We can identify how customer satisfaction (reflected in ratings) correlates with pricing strategies across different brands. This helps in understanding brand positioning in the market.",
    "category_stats": "This table helps identify how different categories perform in terms of customer satisfaction and pricing trends, enabling better decision-making for inventory management, marketing focus, and product development based on category-level insights.",
    "df_recommend_sentiment": "This table helps us to understand the relationship between whether users recommend a product and the sentiment of their reviews. We can assess how product recommendations align with users' actual feelings and experiences. This helps identify products with strong positive or negative reception."
}

def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
# setup background
set_background("image.jpeg")

# Function to get columns for a table
def get_columns(table_name):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        return [row[1] for row in cursor.fetchall()]

# Function to query data from table
def query_table(table_name, selected_cols):
    with sqlite3.connect(DB_PATH) as conn:
        cols_str = ", ".join(selected_cols)
        query = f"SELECT {cols_str} FROM {table_name} LIMIT 1000"
        return pd.read_sql_query(query, conn)

# --- Streamlit UI ---
st.title("Electronic Products Analysis")

# Select table
selected_table = st.selectbox("Select a table:", TABLES)

# Show dynamic table description
if selected_table:
    st.markdown(f"### Description for `{selected_table}`")
    st.markdown(TABLE_DESCRIPTIONS.get(selected_table, "No description available."))

# Fetch columns for selected table
available_columns = get_columns(selected_table)

with st.expander("Column Selection", expanded=False):
    # Select columns to display
    selected_columns = st.multiselect("Select columns to display:", available_columns, default=available_columns)

    # Display result
    if selected_columns:
        df = query_table(selected_table, selected_columns)
        st.dataframe(df)
    else:
        st.warning("Please select at least one column to display.")

    # Chart rendering logic
    st.subheader("📊 Chart Preview")

    if selected_table == "review_sentiment":
        st.bar_chart(df["predicted_sentiment"].value_counts())

    elif selected_table == "df_price_rating":
        st.bar_chart(df.set_index("price_tier")["avg_rating"])

    elif selected_table == "review_length_summary":
        fig, ax = plt.subplots()
        ax.pie(df["review_count"], labels=df["length_category"], autopct='%1.1f%%')
        st.pyplot(fig)

    elif selected_table == "df_recommend_sentiment":
        fig, ax = plt.subplots()
        ax.bar(df["Recommendation"], df["avg_sentiment"])
        ax.set_ylabel("Avg Sentiment")
        ax.set_title("Sentiment by Recommendation")
        st.pyplot(fig)

    elif selected_table == "brand_stats":
        st.bar_chart(df.set_index("brand")[["review_count", "avg_stars"]])

    elif selected_table == "category_stats":
        st.bar_chart(df.set_index("category")[["review_count", "avg_stars"]])

    elif selected_table == "monthly_reviews":
        st.line_chart(df.set_index("review_month")[["num_reviews", "avg_stars"]].sort_index())

    elif selected_table == "yearly_reviews":
        st.line_chart(df.set_index("review_year")[["num_reviews", "avg_stars"]].sort_index())

# --- Dynamic Filters ---
# st.markdown("### 🔍 Filter Conditions")
with st.expander("🔍 Filter Conditions", expanded=False):
    filters = {}
    for col in available_columns:
        unique_vals = df[col].dropna().unique()

        if len(unique_vals) > 50:
            val = st.text_input(f"Filter `{col}` (contains):")
            if val:
                df = df[df[col].astype(str).str.contains(val, case=False, na=False)]
        else:
            selected = st.multiselect(f"Select `{col}` values:", sorted(unique_vals))
            # ✅ Apply filter only if selection is not empty AND user made a choice
            if selected:
                df = df[df[col].isin(selected)]
    # Display filtered results
    st.markdown("### 📋 Filtered Table")
    st.dataframe(df)

import pandas as pd

def printFromDB(conn, query):
  output = pd.read_sql_query(query, conn)
  print("sqlite output: \n", output)

  
import sqlite3
conn = sqlite3.connect("final_results.db")

printFromDB(conn, "SELECT * FROM sqlite_master WHERE type='table';")

printFromDB(conn, "SELECT * FROM  review_sentiment;")
printFromDB(conn, "SELECT * FROM df_price_rating;")
printFromDB(conn, "SELECT * FROM review_length_summary;")
printFromDB(conn, "SELECT * FROM df_recommend_sentiment;")
printFromDB(conn, "SELECT * FROM   brand_stats;")
printFromDB(conn, "SELECT * FROM category_stats;")
printFromDB(conn, "SELECT * FROM  monthly_reviews;")
printFromDB(conn, "SELECT * FROM   yearly_reviews;")
conn.close()
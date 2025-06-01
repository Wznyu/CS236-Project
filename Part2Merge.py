import os
import sqlite3
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, length, when, avg, udf, count, from_unixtime, date_format,
    round, explode, from_json, flatten
)
from pyspark.sql.types import StringType, ArrayType
import sparknlp
from sparknlp.base import DocumentAssembler, Pipeline
from sparknlp.annotator import Tokenizer, ViveknSentimentModel

# Set Spark environment
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
os.environ['SPARK_DRIVER_HOST'] = '127.0.0.1'
os.environ['SPARK_DRIVER_BIND_ADDRESS'] = '127.0.0.1'

# Initialize Spark
spark = sparknlp.start()

# path of datasets
reviews_path = "Datasets/Electronics_5.json"

if os.path.exists("Datasets/ElectronicsMetadata.csv"):
    metadata_path = "Datasets/ElectronicsMetadata.csv"
else:
    metadata_path = "Datasets/Amazon Electronics Metadata.csv"
    
if os.path.exists("Datasets/ElectronicsProductData.csv"):
    ab_reviews_path = "Datasets/ElectronicsProductData.csv"
else:
    ab_reviews_path = "Datasets/DatafinitiElectronicsProductData.csv"

# review sentiment
def review_sentiment(meta_df, reviews_df, conn):
    # Merge
    df_merged = reviews_df.join(meta_df.select("asin", "brand"), on="asin", how="inner")

    # NLP pipeline
    document_assembler = DocumentAssembler().setInputCol("reviewText").setOutputCol("document")
    tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token")
    sentiment_model = ViveknSentimentModel.pretrained().setInputCols(["document", "token"]).setOutputCol("predicted_sentiment")
    nlp_pipeline = Pipeline(stages=[document_assembler, tokenizer, sentiment_model])
    nlp_model = nlp_pipeline.fit(df_merged)
    df_nlp_result = nlp_model.transform(df_merged)

    # Extract sentiment
    get_sent = udf(lambda x: x[0] if x else "Unknown", StringType())
    df_final = df_nlp_result.withColumn("predicted_sentiment", get_sent(col("predicted_sentiment.result")))

    # Add columns
    df_final = (
        df_final
        .withColumn("vote_count", col("helpful").getItem(1).cast("int"))
        .withColumn("review_length", length(col("reviewText")))
        .withColumn(
            "sentiment_index",
            when(col("predicted_sentiment") == "negative", 0)
            .when(col("predicted_sentiment") == "neutral", 1)
            .otherwise(2)
        )
    )

    # Cast types for analysis
    df_final = df_final.withColumn("vote_count", col("vote_count").cast("int")) \
                    .withColumn("review_length", col("review_length").cast("int")) \
                    .withColumn("sentiment_index", col("sentiment_index").cast("int")) \
                    .withColumn("overall", col("overall").cast("double"))

    df_final.select("asin", "brand", "reviewText", "predicted_sentiment", "sentiment_index", "vote_count", "review_length", "overall")

    # Write to SQLite
    saveToDB(conn, df_final, "review_sentiment")
    return df_final

def printStats(df_final):
    # Sentiment count
    print("=== Spark NLP Predicted Sentiment Class Counts ===")
    df_final.groupBy("predicted_sentiment").count().show(truncate=False)

    # Correlations
    corr_len = df_final.stat.corr("vote_count", "review_length")
    corr_rating = df_final.stat.corr("vote_count", "overall")
    corr_sent = df_final.stat.corr("vote_count", "sentiment_index")
    print(f"Correlation(vote_count vs. review_length): {corr_len:.4f}")
    print(f"Correlation(vote_count vs. star_rating):   {corr_rating:.4f}")
    print(f"Correlation(vote_count vs. sentiment):     {corr_sent:.4f}")

    # Avg vote by sentiment
    print("=== Average Vote Count by Predicted Sentiment ===")
    df_final.groupBy("predicted_sentiment").agg(count("*").alias("count"), avg("vote_count")).show(truncate=False)

# Price vs rating
def price_rating(meta_df, df_final, conn):
    df_metadata_cleaned = meta_df.withColumn(
        "price_cleaned",
        when(col("price").rlike("^[0-9]+\\.?[0-9]*$"), col("price").cast("double"))
    ).dropna(subset=["price_cleaned"])

    df_price_rating = df_final.join(
        df_metadata_cleaned.select("asin", "price_cleaned"), on="asin", how="inner"
    )

    df_price_rating = df_price_rating.withColumn(
        "price_tier",
        when(col("price_cleaned") < 50, "Low")
        .when(col("price_cleaned") <= 150, "Medium")
        .otherwise("High")
    )

    print("=== Average Ratings by Price Tier ===")
    df_price_rating = df_price_rating.groupBy("price_tier") \
                .agg(avg("overall").alias("avg_rating")) \
                .orderBy("price_tier") 
                
    # Write df_price_rating to SQLite
    saveToDB(conn, df_price_rating, "df_price_rating")

# Review length vs. vote
def review_length_summary(df_final, conn):
    print("=== Average Vote Count and Count by Review Length ===")
    df_review_length_summary = df_final.withColumn(
        "length_category",
        when(col("review_length") > 500, "Long").otherwise("Short")
    ).groupBy("length_category") \
    .agg(
        avg("vote_count").alias("avg_votes"),
        count("*").alias("review_count")
    )
    saveToDB(conn, df_review_length_summary, "review_length_summary")
    df_review_length_summary.show()


# Sentiment and Recommendation Analysis
def recommend_sentiment(df_final, df_dofin, conn):
    df_combined = df_final.join(df_dofin, on="asin", how="inner")

    df_re = df_combined.filter(col("`reviews.doRecommend`").isNotNull()) \
        .groupBy("`reviews.doRecommend`") \
        .agg(
            col("`reviews.doRecommend`").alias("Recommendation"),
            avg("sentiment_index").alias("avg_sentiment"),
            avg("vote_count").alias("avg_votes"),
            count("*").alias("count")
        )
    df_re = df_re.drop(col("`reviews.doRecommend`"))
    df_re.show()
    saveToDB(conn, df_re, "df_recommend_sentiment")

# merge query
def monthly_review(meta_df, reviews_df, conn):
  temp_reviews_df = reviews_df.withColumn("review_month", date_format(from_unixtime(col("unixReviewTime")), "yyyy-MM"))
  monthly_reviews_joined = temp_reviews_df.join(meta_df.select("asin"), on="asin", how="inner")

  monthly_reviews = monthly_reviews_joined.groupBy("asin", "review_month").agg(
      count("*").alias("num_reviews"),
      round(avg("overall"), 2).alias("avg_stars")
  )
  saveToDB(conn, monthly_reviews, "monthly_reviews")
  monthly_reviews.show(20, truncate=False)
  
def yearly_review(meta_df, reviews_df, conn):
  temp_reviews_df = reviews_df.withColumn("review_year", date_format(from_unixtime(col("unixReviewTime")), "yyyy"))
  yearly_reviews_joined = temp_reviews_df.join(meta_df.select("asin"), on="asin", how="inner")

  yearly_reviews = yearly_reviews_joined.groupBy("asin", "review_year").agg(
      count("*").alias("num_reviews"),
      round(avg("overall"), 2).alias("avg_stars")
  )
  saveToDB(conn, yearly_reviews, "yearly_reviews")
  yearly_reviews.show(20, truncate=False)

# Brand statistics
def brand_stat(meta_df, reviews_df, conn):
  brands_df = meta_df.select("asin", "brand", "price") \
            .filter((col("brand").isNotNull()) & (col("brand") != "Unknown"))
  joined_df = reviews_df.join(brands_df, on="asin", how="right")
  brand_stats = joined_df.groupBy("brand").agg(
      count("*").alias("review_count"),
      round(avg("overall"), 1).alias("avg_stars"),
      round(avg("price"), 2).alias("avg_price")
  ).orderBy(col("review_count").desc())
  saveToDB(conn, brand_stats, "brand_stats")

  brand_stats.show(truncate=False)

# Category statistics
def category_stat(meta_df, reviews_df, conn):
  nested_array_schema = ArrayType(ArrayType(StringType()))
  exploded_df = meta_df.withColumn("parsed_categories", from_json("categories", nested_array_schema))\
            .withColumn("flat_categories", flatten(col("parsed_categories"))) \
            .withColumn("category", explode("flat_categories"))   
  joined = reviews_df.join(exploded_df.select("asin", "category", "price"), on="asin", how="inner")

  category_stats = joined.groupBy("category").agg(
      count("*").alias("review_count"),
      round(avg("overall"), 2).alias("avg_stars"),
      round(avg("price"), 2).alias("avg_price")
  ).filter(col("avg_stars").isNotNull()).orderBy(col("avg_stars").desc())

  saveToDB(conn, category_stats, "category_stats")
  category_stats.show(truncate=False)
  
# save dataframe to db
def saveToDB(conn, df, tableName):
  df_pd = df.toPandas()
  df_pd.to_sql(tableName, conn, if_exists="replace", index=False)
  conn.commit()
  

def printFromDB(conn, query):
  output = pd.read_sql_query(query, conn)
  print("sqlite output: \n", output)
    

if __name__=='main':
    # connect db
    sqlite_path = "final_results.db"
    conn = sqlite3.connect(sqlite_path)
    
    # loading datasets
    meta_df = spark.read.option("header", True)\
            .option("multiLine", True)\
            .option("escape", "\"")\
            .option("quote", "\"")\
            .csv(metadata_path)
    
    reviews_df = spark.read.json(reviews_path)
    df_dofin = spark.read.option("header", True)\
            .option("multiLine", True)\
            .option("escape", "\"")\
            .option("quote", "\"")\
            .csv(ab_reviews_path)

    # Clean data
    reviews_df = reviews_df.dropna(subset=["asin", "reviewText", "overall"]).dropDuplicates(["asin", "reviewText"])
    meta_df = meta_df.dropna(subset=["asin", "brand"]).dropDuplicates(["asin", "brand"])
    df_dofin = df_dofin.selectExpr("explode(split(asins, ',')) as asin", "`reviews.doRecommend`")

    df_final = review_sentiment(meta_df, reviews_df, conn)
    printStats(df_final)
    
    price_rating(meta_df, df_final, conn)
    review_length_summary(df_final, conn)
    recommend_sentiment(df_final, df_dofin, conn)
    brand_stat(meta_df, reviews_df, conn)
    category_stat(meta_df, reviews_df, conn)
    monthly_review(meta_df, reviews_df, conn)
    yearly_review(meta_df, reviews_df, conn)

    printFromDB(conn, "SELECT * FROM sqlite_master WHERE type='table';")

    conn.close()
    
spark.stop()

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, length, col

os.environ['SPARK_LOCAL_IP']       = '127.0.0.1'
os.environ['SPARK_DRIVER_HOST']     = '127.0.0.1'
os.environ['SPARK_DRIVER_BIND_ADDRESS'] = '127.0.0.1'

spark = (
    SparkSession.builder
        .appName("CS236-Part2")
        .master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
)
sc = spark.sparkContext

reviews_path = r"/home/wzhen033/CS236/CS236-Project/Datasets/Electronics_5.json"
df_reviews = spark.read.json(reviews_path)

df_labeled = df_reviews.withColumn(
    "sentiment",
    when(col("overall").between(1, 2), "Negative")
     .when(col("overall") == 3,    "Neutral")
     .otherwise(                  "Positive")
)

print("=== Sentiment Class Counts ===")
df_labeled.groupBy("sentiment") \
          .count() \
          .show(truncate=False)

df_analysis = (
    df_labeled
    .withColumn("vote_count",      col("helpful").getItem(1).cast("int"))
    .withColumn("review_length",   length(col("reviewText")))
    .withColumn(
        "sentiment_index",
        when(col("sentiment") == "Negative", 0)
         .when(col("sentiment") == "Neutral",  1)
         .otherwise(                          2)
    )
)

corr_length  = df_analysis.stat.corr("vote_count", "review_length")
corr_rating  = df_analysis.stat.corr("vote_count", "overall")
corr_sent    = df_analysis.stat.corr("vote_count", "sentiment_index")

print(f"Correlation(vote_count vs. review_length): {corr_length:.4f}")
print(f"Correlation(vote_count vs. star_rating):   {corr_rating:.4f}")
print(f"Correlation(vote_count vs. sentiment):     {corr_sent:.4f}\n")

print("=== Average Vote Count by Sentiment ===")
df_analysis.groupBy("sentiment") \
           .avg("vote_count") \
           .show(truncate=False)

spark.stop()

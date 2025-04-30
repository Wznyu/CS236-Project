from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("EletronicsProductData").getOrCreate()
df = spark.read.option("header", "true").csv("DatafinitiElectronicsProductData.csv")
df.printSchema()
df.select("`reviews.title`", "`reviews.text`").show(5)





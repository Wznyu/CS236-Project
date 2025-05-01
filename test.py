from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("EletronicsProductData").getOrCreate()
df = spark.read.option("header", "true").csv("./Datasets/ElectronicsMetaData.csv")
print("The dataset of ElectronicsMetaData")
df.printSchema()
df.show(5)
print("The number of rows: " , df.count())
print("The number of columns: " , len(df.columns))

df = spark.read.option("header", "true").json("./Datasets/Electronics_5.json")
print("\n\nThe dataset of Electronics_5")
df.printSchema()
df.show(5)
print("The number of rows: " , df.count())
print("The number of columns: " , len(df.columns))

df = spark.read.option("header", "true").csv("./Datasets/ElectronicsProductData.csv")
print("\nThe dataset of ElectronicsProductData")
df.printSchema()
df.show(2)
print("The number of rows: " , df.count())
print("The number of columns: " , len(df.columns))





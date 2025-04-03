import sys
sys.stdout.reconfigure(encoding='utf-8')
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp, hour, dayofweek, month
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import os

# ---- 1. Khởi tạo SparkSession ----
spark_master_url = "spark://172.20.10.2:7077"

spark = SparkSession.builder \
    .appName("TaxiFarePrediction_Cluster") \
    .master(spark_master_url) \
    .config("spark.executor.memory", "12g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.cores", "16") \
    .config("spark.cores.max", "32") \
    .getOrCreate()

print(f"SparkSession đã kết nối tới {spark_master_url}. Spark version: {spark.version}")

# ---- 2. Tải dữ liệu ----
data_path = "D:/DOAN/DOAN/data/yellow_tripdata_2016-01.csv"  # Update path to your CSV file
if not os.path.exists(data_path):
    print(f"Lỗi: File '{data_path}' không tồn tại!")
    spark.stop()
    exit()

df = spark.read.csv(data_path, header=True, inferSchema=True)
print(f"Dữ liệu ban đầu: {df.count()} bản ghi")
df.printSchema()

# ---- 3. Làm sạch dữ liệu ----
required_columns = [
    "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count",
    "trip_distance", "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude", "fare_amount"
]

df = df.select(*required_columns) \
    .withColumn("pickup_time", col("tpep_pickup_datetime").cast("timestamp")) \
    .withColumn("dropoff_time", col("tpep_dropoff_datetime").cast("timestamp")) \
    .dropna()

df = df.filter(
    (col("fare_amount") > 2) & (col("fare_amount") < 500) &
    (col("trip_distance") > 0) & (col("trip_distance") < 50) &
    (col("passenger_count") > 0) & (col("passenger_count") <= 6)
)
print(f"Sau khi lọc dữ liệu: {df.count()} bản ghi")

# ---- 4. Kỹ thuật đặc trưng ----
df = df.withColumn("trip_duration_seconds", unix_timestamp(col("dropoff_time")) - unix_timestamp(col("pickup_time"))) \
    .filter(col("trip_duration_seconds") > 0)
df = df.withColumn("hour_of_day", hour(col("pickup_time"))) \
    .withColumn("day_of_week", dayofweek(col("pickup_time"))) \
    .withColumn("month", month(col("pickup_time")))

feature_cols = [
    "passenger_count", "trip_distance",
    "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
    "trip_duration_seconds", "hour_of_day", "day_of_week", "month"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)

# ---- 5. Chia dữ liệu ----
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
print(f"Tập train: {train_data.count()} bản ghi")
print(f"Tập test: {test_data.count()} bản ghi")

# ---- 6. Huấn luyện mô hình ----
lr = LinearRegression(featuresCol="scaled_features", labelCol="fare_amount")

paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
    .build()

pipeline = Pipeline(stages=[assembler, scaler, lr])
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(labelCol="fare_amount"),
                          numFolds=3)

print("Đang huấn luyện mô hình...")
cv_model = crossval.fit(train_data)
print("Huấn luyện thành công!")

# ---- 7. Đánh giá mô hình ----
predictions = cv_model.transform(test_data)

rmse = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="rmse").evaluate(predictions)
mae = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="mae").evaluate(predictions)
r2 = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="r2").evaluate(predictions)

print("\nKết quả đánh giá mô hình:")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE: {mae:.4f}")
print(f"  R²: {r2:.4f}")


# ---- 9. Dừng SparkSession ----
spark.stop()
print("Hoàn thành.")
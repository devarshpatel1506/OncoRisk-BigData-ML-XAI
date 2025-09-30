from pyspark.sql import SparkSession
import shap
import pickle
import joblib
import json
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col
import time
import psutil
import os

# Load Bloom Filter
print('Loading the Bloom Filter')
with open('bloom_filter.pkl', 'rb') as f:
    bloom_filter = pickle.load(f)

# Record start time and initial memory before streaming starts
global_start_time = time.time()
process = psutil.Process(os.getpid())
mem_before_streaming = process.memory_info().rss

# Start Spark session with Kafka package
spark = SparkSession.builder.appName('Kafka_EHR_Session').config(
    'spark.jars.packages', 'org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5').getOrCreate()

# Read stream data from Kafka topic
df = spark.readStream.format('kafka') \
    .option('kafka.bootstrap.servers', 'localhost:9092') \
    .option('subscribe', 'ehr_data_topic').load()

# Cast Kafka value as a string
df = df.selectExpr('CAST(value as STRING)')


# Function to check Bloom filter
def bloom(value):
    row_dict = json.loads(value)
    print("Row going into the model: ", row_dict)

    unique_key = f"{row_dict['menopause']}_{row_dict['age_group']}_{row_dict['density']}_{row_dict['race']}_{row_dict['Hispanic']}_{row_dict['BMI']}_{row_dict['Age_First']}_{row_dict['NRELBC']}_{row_dict['BRSTPROC']}_{row_dict['LASTMAMM']}_{row_dict['SURGMENO']}_{row_dict['HRT']}"

    if unique_key not in bloom_filter:
        return 0
    else:
        return 1

# Register UDF
predicted_udf = udf(bloom, IntegerType())

# Apply UDF
df = df.withColumn('final_prediction', predicted_udf(col("value")))

# Write to console
query = df.select("value", "final_prediction") \
    .writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", "false") \
    .option("numRows", 10000).start()

timeout_secs = 60  # Change this to how long you want the stream to run
query.awaitTermination(timeout=timeout_secs)

if query.isActive:
    query.stop()
# After streaming ends
global_end_time = time.time()
mem_after_streaming = process.memory_info().rss

total_time = global_end_time - global_start_time
total_mem_used = (mem_after_streaming - mem_before_streaming) / (1024 * 1024)

print(f"\n=== STREAMING STATS ===")
print(f"Total Time Taken     : {total_time:.2f} seconds")
print(f"Total Memory Used    : {total_mem_used:.4f} MB")

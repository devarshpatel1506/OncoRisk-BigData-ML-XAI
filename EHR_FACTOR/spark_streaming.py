from pyspark.sql import SparkSession
import pickle
import joblib
import json
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col

with open('bloom_filter.pkl','rb') as f:
    filter=pickle.load(f)


model=joblib.load('model.pkl')

spark=SparkSession.builder.appName('EHR_RISK_DATA_BF_ML').config(
    'spark.jars.packages', 'org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5').getOrCreate()

df=spark.readStream.format('kafka').option('kafka.bootstrap.servers','localhost:9092').option('subscribe','ehr_risk_topic').load()

df=df.selectExpr('CAST(value as STRING)')

def bloom_ml_model(value):
    rows=json.loads(value)
    unique_val=f"{rows['age_group_5_years']}_{rows['race_eth']}_{rows['first_degree_hx']}_{rows['age_menarche']}_{rows['age_first_birth']}_{rows['BIRADS_breast_density']}_{rows['current_hrt']}_{rows['menopaus']}_{rows['bmi_group']}_{rows['biophx']}"

    if unique_val not in filter:
        return 0
    
    features=[
        rows['age_group_5_years'],rows['race_eth'],rows['first_degree_hx'],rows['age_menarche'],rows['age_first_birth'],rows['BIRADS_breast_density'],rows['current_hrt'],rows['menopaus'],rows['bmi_group'],rows['biophx']
    ]
    print("Running The Model PRediction")
    prediction=model.predict([features])
    return int(prediction[0])

# Register UDF to use in DataFrame transformations
predicted_udf = udf(bloom_ml_model, IntegerType())

# Apply UDF to the Kafka DataFrame
df = df.withColumn('final_prediction', predicted_udf(col("value")))

# Write the results to the console
query = df.select("value", "final_prediction") \
    .writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", "false") \
    .option("numRows", 10000).start()


query.awaitTermination()


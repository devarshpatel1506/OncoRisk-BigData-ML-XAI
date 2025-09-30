from pyspark.sql import SparkSession
import shap
import pickle
import joblib
import json
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col

# Load Bloom Filter and ML Model
print('Loading the Bloom Filter')
with open('bloom_filter.pkl', 'rb') as f:
    bloom_filter = pickle.load(f)

# Load the ML model (Assuming it's a classification model)
best_model = joblib.load('model.pkl')

# Start Spark session with Kafka package
spark = SparkSession.builder.appName('Kafka_EHR_Session').config(
    'spark.jars.packages', 'org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5').getOrCreate()

# Read stream data from Kafka topic
df = spark.readStream.format('kafka') \
    .option('kafka.bootstrap.servers', 'localhost:9092') \
    .option('subscribe', 'ehr_data_topic').load()

# Cast Kafka value as a string
df = df.selectExpr('CAST(value as STRING)')

explainer = shap.TreeExplainer(best_model)


# Function to check Bloom filter and apply ML model
def bloom_and_ml_model(value):
    row_dict = json.loads(value)
    print("Row going into the model: ", row_dict)  # Print the row data
    # Create unique key based on features in the row
    unique_key = f"{row_dict['menopause']}_{row_dict['age_group']}_{row_dict['density']}_{row_dict['race']}_{row_dict['Hispanic']}_{row_dict['BMI']}_{row_dict['Age_First']}_{row_dict['NRELBC']}_{row_dict['BRSTPROC']}_{row_dict['LASTMAMM']}_{row_dict['SURGMENO']}_{row_dict['HRT']}"
    
    # Check if the key is in the Bloom filter
    if unique_key not in bloom_filter:
        return 0  # If not found, return 0
    
    # Otherwise, apply the ML model (assuming it's a classification task)
    # Extract the relevant features (you may need to modify this based on your model)
    features = [
        row_dict['menopause'], row_dict['age_group'], row_dict['density'], 
        row_dict['race'], row_dict['Hispanic'], row_dict['BMI'], 
        row_dict['Age_First'], row_dict['NRELBC'], row_dict['BRSTPROC'], 
        row_dict['LASTMAMM'], row_dict['SURGMENO'], row_dict['HRT']
    ]
    
    prediction = best_model.predict([features])  # Assuming your model expects this structure
    print("Going into the prediction model")
    shap_values = explainer.shap_values([features])
    return int(prediction[0]) # Return the prediction as an integer

# Register UDF to use in DataFrame transformations
predicted_udf = udf(bloom_and_ml_model, IntegerType())

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

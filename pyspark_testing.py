from pyspark.sql import SparkSession
from pyspark import SparkFiles
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import time
from sklearn.metrics import accuracy_score
import psutil
import os
import requests

# Initialize Spark
spark = SparkSession.builder \
    .appName("MammographyImageProcessing") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

sc = spark.sparkContext

# Configure HDFS access
HDFS_HOST = "10.0.0.41"
HDFS_PORT = "9870"
BASE_PATH = "/mammography_data/test"

def get_memory_usage():
    """Return current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

def get_hdfs_filelist(folder):
    """Get list of image files from HDFS directory"""
    url = f"http://{HDFS_HOST}:{HDFS_PORT}/webhdfs/v1{BASE_PATH}/{folder}/{folder}?op=LISTSTATUS"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return [
            f['pathSuffix'] for f in response.json()['FileStatuses']['FileStatus'] 
            if f['pathSuffix'].lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
    except Exception as e:
        print(f"Error listing {folder}: {str(e)}")
        return []

def process_image(url_info):
    """Process single image and return prediction"""
    folder, filename = url_info
    file_path = f"{BASE_PATH}/{folder}/{folder}/{filename}"
    url = f"http://{HDFS_HOST}:{HDFS_PORT}/webhdfs/v1{file_path}?op=OPEN"
    
    try:
        # Fetch and process image
        resp = requests.get(url, stream=True, timeout=15)
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        pred = model.predict(img_array, verbose=0)[0][0]
        true_label = 1 if folder == 'malignant' else 0
        
        return (filename, true_label, float(pred))
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return (filename, -1, -1.0)  # Return invalid values for failed processing

# Record initial memory usage
initial_mem = get_memory_usage()
print(f"Initial memory usage: {initial_mem:.2f} MB")

# Load model (broadcast to workers)
print("Loading model...")
model_load_start = time.time()
model = tf.keras.models.load_model('breast_cancer_model.h5')
model_load_time = time.time() - model_load_start
model_load_mem = get_memory_usage()
print(f"Model loaded in {model_load_time:.2f} seconds")
print(f"Memory after model load: {model_load_mem:.2f} MB")

# Broadcast model to workers
model_bc = sc.broadcast(model)
loss_fn = tf.keras.losses.BinaryCrossentropy()

# Get list of images to process
image_urls = []
for folder in ['benign', 'malignant']:
    files = get_hdfs_filelist(folder)
    image_urls.extend([(folder, f) for f in files])

# Create RDD and process images in parallel
start_time = time.time()
results_rdd = sc.parallelize(image_urls, numSlices=100).map(process_image)
results = results_rdd.collect()  # Trigger computation
total_time = time.time() - start_time

# Filter out failed processing attempts
valid_results = [r for r in results if r[1] != -1]
total_images = len(valid_results)

# Calculate metrics
if total_images > 0:
    true_labels = np.array([r[1] for r in valid_results])
    pred_probs = np.array([r[2] for r in valid_results])
    accuracy = accuracy_score(true_labels, pred_probs > 0.5)
    avg_loss = loss_fn(true_labels, pred_probs).numpy()
    final_memory = get_memory_usage()
    
    print("\n=== Test Results ===")
    print(f"Total Images Tested: {total_images}")
    print(f"Total Processing Time: {total_time:.2f} seconds")
    print(f"Model Loading Time: {model_load_time:.2f} seconds")
    print(f"Average Accuracy: {accuracy:.4f}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Time per Image: {total_time/total_images:.4f} seconds")
    print("\n=== Memory Usage ===")
    print(f"Initial Memory: {initial_mem:.2f} MB")
    print(f"After Model Load: {model_load_mem:.2f} MB")
    print(f"Final Memory Usage: {final_memory:.2f} MB")
    print(f"Total Memory Increase: {final_memory - initial_mem:.2f} MB")
    


# Stop Spark
spark.stop()
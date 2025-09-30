import requests
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import time
from sklearn.metrics import accuracy_score
import psutil
import os

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

# Record initial memory usage
initial_mem = get_memory_usage()
print(f"Initial memory usage: {initial_mem:.2f} MB")

# Load model
print("Loading model...")
model_load_start = time.time()
model = tf.keras.models.load_model('breast_cancer_model.h5')
model_load_time = time.time() - model_load_start
model_load_mem = get_memory_usage()
print(f"Model loaded in {model_load_time:.2f} seconds")
print(f"Memory after model load: {model_load_mem:.2f} MB")
loss_fn = tf.keras.losses.BinaryCrossentropy()

# Initialize metrics
total_images = 0
all_true_labels = []
all_pred_probs = []
start_time = time.time()
peak_memory = model_load_mem

# Process all images
for folder in ['benign', 'malignant']:
    image_files = get_hdfs_filelist(folder)
    if not image_files:
        continue
        
    for filename in image_files:
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
            print(f"Processed {filename}: Prediction = {pred:.4f}")
            
            # Store results
            true_label = 1 if folder == 'malignant' else 0
            all_true_labels.append(true_label)
            all_pred_probs.append(pred)
            total_images += 1
            
            # Update peak memory
            current_mem = get_memory_usage()
            peak_memory = max(peak_memory, current_mem)
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

# Calculate metrics
if total_images > 0:
    total_time = time.time() - start_time
    accuracy = accuracy_score(all_true_labels, np.array(all_pred_probs) > 0.5)
    avg_loss = loss_fn(all_true_labels, all_pred_probs).numpy()
    final_memory = get_memory_usage()
    
    print("\n=== Test Results ===")
    print(f"Total Images Tested: {total_images}")
    print(f"Total Testing Time: {total_time:.2f} seconds")
    print(f"Model Loading Time: {model_load_time:.2f} seconds")
    print(f"Average Accuracy: {accuracy:.4f}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Time per Image: {total_time/total_images:.4f} seconds")
    print("\n=== Memory Usage ===")
    print(f"Initial Memory: {initial_mem:.2f} MB")
    print(f"After Model Load: {model_load_mem:.2f} MB")
    print(f"Peak Memory During Processing: {peak_memory:.2f} MB")
    print(f"Final Memory Usage: {final_memory:.2f} MB")
    print(f"Total Memory Increase: {final_memory - initial_mem:.2f} MB")
else:
    print("No images were processed successfully.")
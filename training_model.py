import os
import time
import psutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === Paths ===
DATA_DIR = os.path.abspath('data')
MODEL_PATH = 'breast_cancer_model.h5'
WEIGHTS_PATH = 'breast_cancer_weights.weights.h5'

# === Image Data Generators ===
img_height, img_width = 224, 224
batch_size = 32


test_datagen = ImageDataGenerator(rescale=1./255)


test_gen = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'test'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# === Train Model and Track Time + Memory ===
print("\n🚀 Starting testing...")
start_time = time.time()
start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # in MB

print("Loading model...")
model_load_start = time.time()
model = tf.keras.models.load_model('breast_cancer_model.h5')

# === Evaluate on Test Set ===
loss, accuracy = model.evaluate(test_gen)

end_time = time.time()
end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # in MB

print(f"\n✅ Test Accuracy: {accuracy:.4f}")
print(f"🕒 Training Time: {end_time - start_time:.2f} sec")
print(f"🧠 Memory Used: {end_memory - start_memory:.2f} MB")
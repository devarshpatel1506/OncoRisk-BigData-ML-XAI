import random
import json
from kafka import KafkaProducer
import pandas as pd

producer=KafkaProducer(bootstrap_servers='localhost:9092',
                       value_serializer=lambda v: json.dumps(v).encode('utf-8'))

def generate_data():
    return {
        "menopause": random.choice([0, 1, 9]),
        "age_group": random.choice(range(1, 11)),
        "density": random.choice([1, 2, 3, 4, 9]),
        "race": random.choice([1, 2, 3, 4, 5, 9]),
        "Hispanic": random.choice([0, 1, 9]),
        "BMI": random.choice([1, 2, 3, 4, 9]),
        "Age_First": random.choice([0, 1, 2, 9]),
        "NRELBC": random.choice([0, 1, 2, 9]),
        "BRSTPROC": random.choice([0, 1, 9]),
        "LASTMAMM": random.choice([0, 1, 9]),
        "SURGMENO": random.choice([0, 1, 9]),
        "HRT": random.choice([0, 1, 9]),
    }
for i in range(1000):
    data=generate_data()
    producer.send('ehr_data_topic',data)
    print(f"Sent data: {data}")

producer.flush()

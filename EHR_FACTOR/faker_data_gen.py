import random
import json
from kafka import KafkaProducer
import pandas as pd

producer=KafkaProducer(bootstrap_servers='localhost:9092',value_serializer=lambda v:json.dumps(v).encode('utf-8'))


def generate_data():
    return {
            "age_group_5_years": random.randint(1, 13),
            "race_eth": random.choice([1, 2, 3, 4, 5, 6, 9]),
            "first_degree_hx": random.choice([0, 1, 9]),
            "age_menarche": random.choice([0, 1, 2, 9]),
            "age_first_birth": random.choice([0, 1, 2, 3, 4, 9]),
            "BIRADS_breast_density": random.choice([1, 2, 3, 4, 9]),
            "current_hrt": random.choice([0, 1, 9]),
            "menopaus": random.choice([1, 2, 3, 9]),
            "bmi_group": random.choice([1, 2, 3, 4, 9]),
            "biophx": random.choice([0, 1, 9]),
        }

df=pd.read_csv('final_risk_factor.csv')
def generate_data_2():
    for _,rows in df.iterrows():
        if rows['breast_cancer_history']==1:
            yield {
            "age_group_5_years": int(rows['age_group_5_years']),
            "race_eth": int(rows['race_eth']),
            "first_degree_hx": int(rows['first_degree_hx']),
            "age_menarche":int(rows['age_menarche']),
            "age_first_birth":int(rows['age_first_birth']),
            "BIRADS_breast_density":int(rows['BIRADS_breast_density']),
            "current_hrt": int(rows['current_hrt']),
            "menopaus":int(rows['menopaus']),
            "bmi_group":int(rows['bmi_group']),
            "biophx":int(rows['biophx'])
        }
#for _ in range(10000):
#    data=generate_data()
#    producer.send('ehr_risk_topic',data)
#    print(f"Sent data: {data}")


# Generate and send data to Kafka
data = generate_data_2()
for record in data:
    producer.send('ehr_risk_topic', record)  # Send each data record to Kafka
    print(f"Sent data: {record}")

producer.flush()  # Ensure all data is sent
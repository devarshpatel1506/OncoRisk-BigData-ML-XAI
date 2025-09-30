import pandas as pd
from pybloom_live import BloomFilter
import pickle

#Reading the data from csv file
df=pd.read_csv('final_risk_factor.csv')

#Intialize the Bloom FIlter
bloomfilter=BloomFilter(capacity=13000000,error_rate=0.01)

def generate_key(rows):
    return f"{rows['age_group_5_years']}_{rows['race_eth']}_{rows['first_degree_hx']}_{rows['age_menarche']}_{rows['age_first_birth']}_{rows['BIRADS_breast_density']}_{rows['current_hrt']}_{rows['menopaus']}_{rows['bmi_group']}_{rows['biophx']}"

for _,rows in df.iterrows():
    if rows['breast_cancer_history']==1:
        unique_val=generate_key(rows)
        bloomfilter.add(unique_val)

# Save the trained Bloom filter
with open("bloom_filter.pkl", "wb") as f:
    pickle.dump(bloomfilter, f)

print("✅ Bloom filter trained and saved!")
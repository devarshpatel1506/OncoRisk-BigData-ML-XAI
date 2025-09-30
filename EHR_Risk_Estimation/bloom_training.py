import pandas as pd
from pybloom_live import BloomFilter
import pickle

#Reading the data from csv file
df=pd.read_csv('EHR_RISK_ESTIMATION.csv')

#Intialize the Bloom FIlter
bloomfilter=BloomFilter(capacity=2320000,error_rate=0.01)

def generate_key(rows):
    return f"{rows['menopause']}_{rows['age_group']}_{rows['density']}_{rows['race']}_{rows['Hispanic']}_{rows['BMI']}_{rows['Age_First']}_{rows['NRELBC']}_{rows['BRSTPROC']}_{rows['LASTMAMM']}_{rows['SURGMENO']}_{rows['HRT']}"

for _,rows in df.iterrows():
    if rows['CANCER']==1:
        uniques_value=generate_key(rows)
        bloomfilter.add(uniques_value)

# Save the trained Bloom filter
with open("bloom_filter.pkl", "wb") as f:
    pickle.dump(bloomfilter, f)

print("✅ Bloom filter trained and saved!")

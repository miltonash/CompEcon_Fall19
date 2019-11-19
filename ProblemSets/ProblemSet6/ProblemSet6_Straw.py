#!/usr/bin/env python3
# Creator: Milton Straw
# Date: Fall 2019
# Notes: None.





# Import the necessary packages.
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
#matplotlib inline
import requests
import json
import statsmodels.api as sm
import seaborn as sns; sns.set(color_codes=True)



# Access the data.
url = "https://data.cms.gov/resource/97k6-zzx3.json"
response = requests.get(url)
health_data = response.text
df = pd.read_json(health_data)
df = df.drop(columns='average_medicare_payments_2')

'''
Description of data...
- average_covered_charges: The provider's average charge for services covered
    by Medicare for all discharges in the DRG.
- average_medicare_payments: The average of Medicare payments to the provider
    for the DRG including the DRG amount, teaching, disproportionate share, capital, and outlier payments for all cases. Also included are co-payment and deductible amounts that the patient is responsible for.
- drg_definiion: DRGs are a classification system that groups similar clinical
    conditions (diagnoses) and the procedures furnished by the hospital during the stay.
- total_discharges: The number of discharges billed by the provider for
    inpatient hospital services.
'''

cols = df.columns.tolist()
cols = [cols[5]] + [cols[6]] + [cols[8]] + [cols[4]] + [cols[7]] + [cols[9]] + [cols[3]] + [cols[2]] + [cols[0]] + [cols[1]] + [cols[10]]
cols

df = df[cols]
df = df.sort_values('hospital_referral_region_description')
df = df.reset_index()
df = df.drop(columns='index')
df.head(n=10)

# list of dtypes to include
include =['object', 'float', 'int']
# calling describe method
desc = df.describe(include = include)
# display
desc

print(df.dtypes)

df['average_covered_charges'] = df['average_covered_charges'].astype(int)
df['average_medicare_payments'] = df['average_medicare_payments'].astype(int)
print(df.dtypes)

count = df['hospital_referral_region_description'].value_counts().tolist()
len(count)

index = df['hospital_referral_region_description'].value_counts().index.tolist()
len(index)

density = {'hospital_referral_region_description': index, 'count': count}
densitydf = pd.DataFrame(density, columns=['hospital_referral_region_description', 'count'])
densitydf = densitydf.sort_values('hospital_referral_region_description')
densitydf = densitydf.reset_index()
densitydf = densitydf.drop(columns='index')
densitydf

densitydf['count'] = densitydf['count'].astype(int)
print(densitydf.dtypes)

mergedf = df.merge(densitydf, how='left', on='hospital_referral_region_description')
mergedf.head(n=10)

x0 = mergedf['count']
x = x0
y1 = mergedf['average_covered_charges']
x, y = np.array(x), np.array(y1)
x = sm.add_constant(x)
print(x)

plt.style.use('seaborn')
vis1 = mergedf[mergedf['drg_definition']=="039 - EXTRACRANIAL PROCEDURES W/O CC/MCC"].plot(x=['count'], y=['average_covered_charges'], kind='scatter', title="Charges by Hospital Density - DRG 039")
vis1.set_ylabel("Average Covered Charges")
vis1.set_xlabel("Number of Hospitals in a Region")
plt.xticks(np.arange(min(x0), max(x0)+1, 1.0))
plt.show()
plt.savefig('visual1.png')

model1 = sm.OLS(y1, x)
results1 = model1.fit()
print(results1.summary())

mergedf['percent_repay'] = mergedf.average_medicare_payments / mergedf.average_covered_charges
mergedf.head(n=10)

vis2 = mergedf[mergedf['drg_definition']=="039 - EXTRACRANIAL PROCEDURES W/O CC/MCC"].plot(x=['count'], y=['percent_repay'], kind='scatter', title="Percentage Repayments by Hospital Density - DRG 039")
vis2.set_ylabel("Percentage of Covered Charges Paid by Medicare")
vis2.set_xlabel("Number of Hospitals in a Region")
plt.xticks(np.arange(min(x0), max(x0)+1, 1.0))
plt.show()
plt.savefig('visual2.png')

y2 = mergedf['percent_repay']
model2 = sm.OLS(y2, x)
results2 = model2.fit()
print(results2.summary())

ax1 = sns.regplot(x="count", y="average_covered_charges", data=mergedf,
                 x_estimator=np.mean)

ax2 = sns.regplot(x="count", y="percent_repay", data=mergedf,
                 x_estimator=np.mean)

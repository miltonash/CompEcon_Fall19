'''
Title: ProblemSet5 - Part B
Created by: Milton Straw
Date: Fall 2019
For: ECON 815, DeBacker
'''

# Import packages.
import pandas as pd
# python -m ensurepip
# python -m pip install lifelines
import lifelines
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt

# Read data.
data = pd.read_stata('dropoutdata.dta')
data.dropna(inplace=True)

'''
ECONOMETRICS
Tests of KM survival functions.
'''

data.head(10)
data.describe()

# Variation 1. Part-time v. full-time schooling status.
# sts test prt, wilcoxon
kmf = KaplanMeierFitter()
p1 = data.prt==1
T1_1 = data[p1]['dur']
C1_1 = data[p1]['evt']
p2 = data.prt==0
T2_1 = data[p2]['dur']
C2_1 = data[p2]['evt']
ax = plt.subplot(111)
kmf.fit(T1_1, event_observed=C1_1, label=['Part-time'])
kmf.survival_function_.plot(ax=ax)
kmf.fit(T2_1, event_observed=C2_1, label=['Full-time'])
kmf.survival_function_.plot(ax=ax)
summary1 = logrank_test(T1_1, T2_1, C1_1, C2_1, alpha=99)
print(summary1)

# Variation 2. Ever married.
# sts test evermarried, wilcoxon
m1 = data.evermarried==1
T1_2 = data[m1]['dur']
C1_2 = data[m1]['evt']
m2 = data.evermarried==0
T2_2 = data[m2]['dur']
C2_2 = data[m2]['evt']
kmf.fit(T1_2, event_observed=C1_2, label=['Ever married'])
kmf.survival_function_.plot(ax=ax)
kmf.fit(T2_2, event_observed=C2_2, label=['Never married'])
kmf.survival_function_.plot(ax=ax)
summary2 = logrank_test(T1_2, T2_2, C1_2, C2_2, alpha=99)
print(summary2)

# Variation 3 Gender.
# sts test sex, wilcoxon
s1 = data.sex==1
T1_3 = data[s1]['dur']
C1_3 = data[s1]['evt']
s2 = data.sex==0
T2_3 = data[s2]['dur']
C2_3 = data[s2]['evt']
kmf.fit(T1_3, event_observed=C1_3, label=['Female'])
kmf.survival_function_.plot(ax=ax)
kmf.fit(T2_3, event_observed=C2_3, label=['Male'])
kmf.survival_function_.plot(ax=ax)
summary3 = logrank_test(T1_3, T2_3, C1_3, C2_3, alpha=99)
print(summary3)

# Extras. Proportional hazard and AFT models with Weibull distribution.
# streg sex grd prt evermarried, distribution(weibull)
# streg sex grd prt evermarried, distribution(weibull) time

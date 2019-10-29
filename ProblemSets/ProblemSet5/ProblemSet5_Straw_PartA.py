'''
Title: ProblemSet5 - Part A
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
from lifelines import NelsonAalenFitter
from lifelines import CoxPHFitter
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read data.
data = pd.read_stata('dropoutdata.dta')
data.dropna(inplace=True)

'''
VISUALIZATIONS
'''

# 1. Kaplan Meier Survivor Function
kmf = KaplanMeierFitter()
T = data['dur']
C = data['evt']
kmf.fit(T, event_observed=C)
fig1 = kmf.plot(title='Survivor Function, Drop Out')
fig1.savefig('fig1.png')

# 2. Nelson Aalen Cumulative Hazard Function
naf = NelsonAalenFitter()
naf.fit(T, event_observed=C)
fig2 = naf.plot(title='Cumulative Hazard Function, Drop Out')
fig2.savefig('fig2.png')

# 3. Cox Proportional Hazard Model
cph = CoxPHFitter()
cph.fit(data, 'sex', event_col='evt')
fig3 = cph.predict_survival_function(data).plot()
fig3.savefig('fig3.png')
'''
I couldn't make this one give me the result I wanted.
The functioning Stata code is:
stphplot, by(sex) nolntime
and the resulting visualization is...
'''
img = mpimg.imread('cph.png')
imgplot = plt.imshow(img)
plt.show()

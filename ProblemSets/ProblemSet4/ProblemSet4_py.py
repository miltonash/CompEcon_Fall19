# Creator: Milton Straw
# Date: Fall 2019
# Notes: This assignment is incomplete. I was not able to make my code function, and therefor did not accomplish a PDF document to discuss results.





#Import necessary packages.
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.optimize import differential_evolution as de
# python -m ensurepip
# python -m pip install geopy
import geopy.distance as gp
from geopy.distance import vincenty as vin
from geopy.distance import geodesic as gd



# Importing the data.
# hw_data = pd.read_excel('../../repos/CompEcon_Fall19/Matching/radio_merger_data.xlsx')
hw_data = pd.read_excel('../Matching/radio_merger_data.xlsx')
hw_data.dropna(inplace = True)

# Iterating the columns to see the variables we have for each observation.
print('>' " COLUMN HEADS (Variables):")
for col in hw_data.columns:
    print(col)
hw_data = hw_data.drop_duplicates(['buyer_id','year','buyer_lat','buyer_long'])

# See if the data read properly.
# Percentile list.
perc =[.20, .40, .60, .80]
# List of datatypes to include.
include =['object', 'float', 'int']
desc = hw_data.describe(percentiles = perc, include = include)
pd.set_option('display.expand_frame_repr', False)
print('\n' '>' " DESCRIBE:" '\n', desc)



# Seperate the data by market (year). Because there are only two years, I assign the data without a for loop.
market07 = hw_data[hw_data.year == 2007]
market08 = hw_data[hw_data.year == 2008]

# Scaling data, as recommended by problem set.
market07[['price', 'population_target']] = market07[['price', 'population_target']] / 1000000

market08[['price', 'population_target']] = market08[['price', 'population_target']] / 1000000

# Lists for: characteristics for buyers and targets, and markets.
markets = [market07, market08]
char_b = ['buyer_id', 'buyer_lat', 'buyer_long', 'corp_owner_buyer', 'num_stations_buyer', 'year']
char_t = ['hhi_target', 'population_target', 'price', 'target_id', 'target_lat', 'target_long']

# Empty dataframe to be filled in with counterfactual transactions.
counterfac = pd.DataFrame()

# According to stackoverflow commenters and analyticsvidhya.com, list comprehension is faster than FOR loops and allows 'value for value in variable if conditional' statements.
counterfac = [lc[char_b].iloc[i].values.tolist() + lc[char_t].iloc[j].values.tolist()
    for lc in markets for i in range(len(lc) - 1)
    for j in range(i + 1, len(lc))]

# Finally, fill in the dataframe of counterfactual matches (transactions).
counterfac = pd.DataFrame(counterfac, columns = char_b + char_t)
print('\n' '>' " COUNTERFACTUAL MATCHES:" '\n', counterfac.head(100))



# Calculate the distance between pairs using coordinates.
def distance(df, coords_b, coords_t):
    '''
    Calculate distance between coordinates of pairs in dataframe.
    Input:
        df: dataframe
        coords_b: a tuple of buyer coordinates (longitude and latitude).
        coords_t: a tuple of target coordinates.
    '''
    c = coords_b + coords_t
    lat, long = np.split(df[c].values, 2, axis = 1)
    if len(lat) == len(long):
        distance = [gd(lat[i], long[i]).miles for i in range(len(lat))]
        return distance
    else:
        print('Unequal length of latitude and longitude columns in the dataframe.')

frames = [market07, market08, counterfac]
coords_b = ['buyer_lat', 'buyer_long']
coords_t = ['target_lat', 'target_long']
for lc in frames:
    lc['distance'] = distance(lc, coords_b, coords_t)
distance(hw_data, coords_b, coords_t)



# Calculate the payoffs.
def payoff(df, row, transfer, *params):
    '''
    Calculate value of payoff.
    input:
        df: dataframe
        row: row of dataframe.
        transfer: true or false.
        params: parameters to estimate.
    output:
        payoff: value of payoff in row of dataframe.
    '''
    if transfer == "F":
            alpha, beta = params[0]
            x1, x2, y1, D = ['num_stations_buyer', 'corp_owner_buyer', 'population_target', 'distance']
            i = row
            payoff = (df[x1].iloc[i] * df[y1].iloc[i] + alpha * df[x2].iloc[i] * df[y1].iloc[i] + beta * df[D].iloc[i])
    if transfer == "T":
            delta, alpha, gamma, beta = params[0]
            x1, x2, y1, H, D = ['num_stations_buyer', 'corp_owner_buyer', 'population_target', 'hhi_target', 'distance']
            i = row
            payoff = (delta * df[x1].iloc[i] * df[y1].iloc[i] + alpha * df[x2].iloc[i] * df[y1].iloc[i] + gamma * df[H].iloc[i] + beta * df[D].iloc[i])
    return payoff
payoff(hw_data, row, transfer)



# Generate payoff matrix.
def payoff_matrix(transfer, *params):
    '''
    Generate payoff matrix of values to buyer-target matches.
    input:
        transfer: true or false.
        params: parameters to estimate.
    '''
    n = len(market07)
    cf07 = np.matrix([payoff(counterfac, i, transfer, params) for i in range(n * (n - 1))])
    cf07.resize(n, (n - 1))
    m = len(market08)
    cf08 = np.matrix([payoff(counterfac, i, transfer, params) for i in range(n * (n - 1), len(counterfac))])
    cf08.resize(m, (m - 1))
    real07 = [payoff(market07, i, transfer, params) for i in range(n)]
    real08 = [payoff(market08, i, transfer, params) for i in range(m)]
    return cf07, cf08, real07, real08
payoff_matrix(transfer)



def score(S, transfer):
    '''
    Maximum score.
    input:
        S: Coeffs to estimate.
        transfer: true or false.
    '''
    if transfer == "F":
        alpha, beta = S
        cf07, cf08, real07, real08 = payoff_matrix(transfer, alpha, beta)
        score = [1 for m in [[real07, cf07], [real08, cf08]]
                 for i in range(len(m[0]))
                 for j in range(len(m[0]))
                 if j > i if m[0][i] + m[0][j] >= m[1][j, i] + m[1][i, (j - 1)]]
    if transfer == "T":
        delta, alpha, gamma, beta = S
        cf07, cf08, real07, real08 = payoff_matrix(transfer, delta, alpha, gamma, beta)
        price07 = market07['price'].tolist()
        price08 = market08['price'].tolist()
        score = [1 for m in [[real07, cf07, price_2007], [real08, cf08, price_2008]]
                 for i in range(len(m[0]))
                 for j in range(len(m[0]))
                 if j > i if (m[0][i] - m[1][j, i] >= m[2][i] - m[2][j]) & (m[0][j] - m[1][i, (j - 1)] >= m[2][j] - m[2][i])]
    return sum(score) * -1
score(S, transfer)



# False Transfers.
transfer = "F"
# Guess
S = np.array((1000, -1))
nm_false = opt.minimize(score, S, args = (transfer), method = 'Nelder-Mead')
# Differential Evolution, false transfers.
bound_f = [(0, 500), (-5, 5)]
de_false = de(score, bound_f, params = (transfer))
print('\n' '>' " NELDER-MEAD, FALSE TRANSFERS:" '\n', nm_false)
print('\n' '>' " DIFFERENTIAL EVOLUTION, FALSE TRANSFERS:" '\n', de_false)



# True Transfers
transfer = "T"
# Differential Evolution, true transfers.
bound_t = [(-1000, 1000), (0, 1000), (-1000, 0), (-1000, 0)]
de_true = de(score, bound_t, params = (transfer))
print('\n' '>' " DIFFERENTIAL EVOLUTION, TRUE TRANSFERS:" '\n', de_true)

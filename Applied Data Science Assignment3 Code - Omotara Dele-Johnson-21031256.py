#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:32:14 2022

@author: omotaradele-johnson
"""
import numpy as np
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import wbgapi as wb
import seaborn as sns
import itertools as iter

indicator_Id = ["NY.GDP.PCAP.CD"]

country_groups = ({
    'ZAF' : 'Africa',
    'NGA': 'Africa',
    'GHA': 'Africa',
    'KEN': 'Africa',
    'GBR': 'Europe',
    'BEL': 'Europe',
    'FRA': 'Europe',
    'ITA': 'Europe',
    'IND': 'Asia',
    'PAK': 'Asia',
    'CHN': 'Asia',
    'JPN': 'Asia',
    'CAN': 'North America',
    'MEX': 'North America',
    'USA': 'North America',
    'JAM': 'North America'
})

df  = wb.data.DataFrame(indicator_Id, country_groups, mrv=6)
print(df)

#print(df.describe())
print(df.corr())
print()

# clustering rely on differences apart, so we need to normalize our data to make them have the same range
def norm(column):
    col_min = np.min(column)
    col_max = np.max(column)
    col_norm = (column - col_min) / (col_max - col_min)
# the above is from maths X_norm = X-min(X)/max(x)-min(x) to normalize data to be numerically comparable
    return col_norm
for col in df.columns[1:]:
    df[col] = norm(df[col])
    
print(df.describe())
df_plot = df.describe()

def makeplot(df,col1, col2,color):
    plt.figure()
    plt.scatter(df[col1], df[col2],c=color)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.show()

# Exploratory plot of two consecutive years 
#across the data of consideration

makeplot(df, "YR2015", "YR2016","blue")

#without defining the makeplot the plts wont run
makeplot(df, "YR2017", "YR2018","green")

#without defining the makeplot the plts wont run
makeplot(df, "YR2019", "YR2020","red")

################################################
# Extract X from world bank dataframe to perform  
# K- means cluster using the elbow method
###############################
X = df[1:6].values
def kMeansClusterUsingElbowMethod(df,X):
    
    
    #Performing my clustering using KMean(the elbow method) and AgglomerativeClustering to find the optimal number of clusters of the time and country series
    from sklearn.cluster import KMeans
    wcss = []
    for i in range(1, 6):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 6), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    
kMeansClusterUsingElbowMethod(df,X)

#######################################################
##### set up agglomerative clustering for 5 clusters
#####################################################
def agglomerativeClustering(df):
    ac = cluster.AgglomerativeClustering(n_clusters=5)
    df_fit = df[["YR2015", "YR2016", "YR2017", "YR2018", "YR2019", "YR2020"]].copy()
    ac.fit(df_fit)
    plt.figure()
    labels = ac.labels_
    #compute the cluster centre
    xcen = []
    ycen = []
    for ic in range(6):
        xc = np.average(df_fit["YR2015"][labels==ic])
        yc = np.average(df_fit["YR2016"][labels==ic])
        xcen.append(xc)
        ycen.append(yc)
    # plot using the labels to select colour
    plt.figure(figsize=(5.0,5.0))
    plt.scatter(df_fit["YR2015"], df_fit["YR2016"], df_fit["YR2017"], c='magenta', label = 'Value cluster')
    # show cluster centres
    for ic in range(6):
        plt.plot(xcen[ic], ycen[ic], "dk", markersize=8)    
    plt.xlabel("YR2015")
    plt.ylabel("YR2016")
    plt.show()
    return df_fit

df_fit = agglomerativeClustering(df)
#Determine the curve fit
from scipy.optimize import curve_fit

def func(X , a , b):
  return a+b*X

popt , pcov = curve_fit(func , df_fit["YR2015"] , df_fit["YR2016"] )
print(popt)

#############################################
# To calculate upper and lower limit for
# a curve fit plot
#############################################

def err_ranges(x, func, param, sigma):    
# initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    uplow = []   
# list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
            pmin = p - s
            pmax = p + s
            uplow.append((pmin, pmax))
    pmix = list(iter.product(*uplow))
    for p in pmix:
            y = func(x, *p)
            lower = np.minimum(lower, y)
            upper = np.maximum(upper, y)
    return lower,upper

lower_limit,upper_limit = err_ranges(X,func,popt,np.sum(X,axis=1))
lower_limit,upper_limit

##################################################
# Graph to show the upper and lowe limit
# limit boundaries for the year between 2015 - 2020
###################################################

def showlimitboundaries(lower_limit,upper_limit):
    plt.plot(lower_limit, upper_limit,'r')
    plt.xlabel('lower Limit')
    plt.ylabel('Upper Limit')
    plt.title('Fit Curve')
    plt.show()
    
showlimitboundaries(lower_limit,upper_limit)

##########################################################
##Comparative Analysis was done considering grouping 
#the country of consideration by continent and comparing 
#their GDP per capital for relevant inferences
##########################################################

New_df = wb.data.DataFrame(indicator_Id, country_groups, mrv=6)

#Renaming dataframe headers
New_df = New_df.rename(columns={'YR2015':'2015', 'YR2016':'2016', 'YR2017':'2017', 'YR2018':'2018','YR2019':'2019', 'YR2020':'2020'}, inplace=False)

# Compare all countries GDP per capita using 
#ove the year 2015 - 2020
New_df.transpose().plot(color=['#2a9d8f','#adc178','#a98467','#d4a373','#00eebb','#330077','#B0E0E6',
                               '#7B68EE','#ffc8dd','#a2d2ff','#e76f51','#2a9d8f','#264653','#6d6875',
                               '#b5838d','#e5989b'])

#method to show the analysis between various continents
def comparativeAnalysisAroundContinent(New_df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    
    #plotting a line graph for Africa Continent grouped countries
    Africa = ['NGA', 'GHA', 'KEN', 'ZAF']
    Afr_df = New_df.loc[Africa]
    Afr_dft = Afr_df.transpose()
    
    Afr_dft.plot(ax=axes[0,0])
    axes[0,0].set_title('Line Plot Africa countries')
    axes[0,0].set_xlabel('year')
    axes[0,0].set_ylabel('value')
    
    # Pass the axes into seaborn and Create the bar plot of European grouped countries
    
    Europe = ['BEL','ITA','GBR','FRA']
    Eur_df = New_df.loc[Europe]
    Eurp_dft = Eur_df.transpose()
    
    Eurp_dft.plot(kind='bar',color = ['#00eebb','#330077','#B0E0E6','#7B68EE'], ax=axes[0,1])
    #Adding the aesthetics
    axes[0,1].set_title('Bar Plot of European countries')
    axes[0,1].set_xlabel('Year')
    axes[0,1].set_ylabel('Value');
    
    #Creating a pie chart using the sume the North American grouped countries
    North_America=['USA','MEX','CAN','JAM']
    Na_df = New_df.loc[North_America]
    
    N_Amrc = Na_df.transpose()
    axes[1,0].pie(N_Amrc.sum(), labels=North_America, colors = ['#2a9d8f','#adc178','#a98467','#d4a373'], autopct='%0.5f%%')
    axes[1,0].set_title('Aggregated sum of the North America')
    
    #Creating a bar chart with the Asian grouped countries
    
    Asia = ['IND','PAK','JPN','CHN']
    As_df = New_df.loc[Asia]
    As_dft = As_df.transpose()
    sns.barplot(data=As_dft, palette = "Oranges_r", ax=axes[1,1])
    axes[1,1].set_title('Bar Plot of Asian countries')
    axes[1,1].set_xlabel('X axis title')
    axes[1,1].set_ylabel('Y axis title');
    
    plt.tight_layout(pad=2);

comparativeAnalysisAroundContinent(New_df)



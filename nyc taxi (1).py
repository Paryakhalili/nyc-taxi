#!/usr/bin/env python
# coding: utf-8

# In[64]:


#import libraries
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk


# In[9]:


#reading data
df= pd.read_csv(r"C:\Users\parya\Desktop\train.csv")


# In[10]:


#check types for all the columns
df.dtypes


# In[11]:


#check for null values
df.isna().sum()


# In[12]:


#Convert timestamp to datetime format to fetch the other details as listed below
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])


# In[13]:


#Calculate and assign new columns to the dataframe such as weekday,
#month and pickup_hour which will help us to gain more insights from the data.
df['weekday'] = df.pickup_datetime.dt.day_name
df['month'] = df.pickup_datetime.dt.month
df['weekday_num'] = df.pickup_datetime.dt.weekday
df['pickup_hour'] = df.pickup_datetime.dt.hour


# In[14]:


#Calculate distance between pickup and dropoff coordinates using geodesic
from geopy.distance import geodesic
distance = []
for index in df['pickup_latitude'].index:
    distance.append(geodesic((df['pickup_latitude'].iloc[index],df['pickup_longitude'].
                              iloc[index]),(df['dropoff_latitude'].iloc[index],df['dropoff_longitude'].iloc[index])).miles)
df['distance'] = distance


# In[15]:


#Calculate Speed in miles/hr for further insights
df['speed'] = (df.distance/(df.trip_duration/3600))
df.head(5)


# #### Column Details
# - id - a unique identifier for each trip
# - vendor_id - a code indicating the provider associated with the trip record
# - pickup_datetime - date and time when the meter was engaged
# - dropoff_datetime - date and time when the meter was disengaged
# - passenger_count - the number of passengers in the vehicle (driver entered value)
# - pickup_longitude - the longitude where the meter was engaged
# - pickup_latitude - the latitude where the meter was engaged
# - dropoff_longitude - the longitude where the meter was disengaged
# - dropoff_latitude - the latitude where the meter was disengaged
# - store_and_fwd_flag - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip.
# - trip_duration - duration of the trip in seconds
# - distance - geographic distance between two co-ordinates
# - speed - average spped of the trip (in miles/hr)

# #### Data Analysis

# ### 1. Id
# There are 1048576 Unique id's which represent each row in the data

# ### 2. Vender Id

# In[16]:


fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(12,5))
ax = df['vendor_id'].value_counts().plot(kind='bar',title="Vendors",
                                         ax=axes[0],color = ('blue',(1, 0.5, 0.13)))
df['vendor_id'].value_counts().plot(kind='pie',title="Vendors",ax=axes[1])
ax.set_ylabel("Count")
ax.set_xlabel("Vendor Id")
fig.tight_layout()


#  Here we got to know that there are only 2 venders(1 and 2)
# - Both the venders share almost equal amount of trips, the difference is quite low between two venders
# - But Vendor 2 is evidently more famous among the population as per the above graphs.

# ### 3.Passengers

# In[17]:


pd.options.display.float_format = '{:.2f}'.format #To suppress scientific notation.
df.passenger_count.value_counts()


# In[18]:


fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(16,5))
line = df['passenger_count'].value_counts().plot(kind='bar',title="passenger_count",color = ('blue',(1, 0.5, 0.13)))
line.set_title('Passenger Count',fontsize = 20)
line.set_ylabel(" Count",fontsize = 15)
line.set_xlabel("No. of Passenger ",fontsize = 15)


# - Most of trip consist of passenger either 1 or 2.

# ### 4.Trip duration

# In[19]:


plt.figure(figsize = (20,5))
sns.boxplot(df.trip_duration)
plt.show()


# ### 5.Distance

# In[20]:


plt.figure(figsize = (20,5))
sns.boxplot(df.distance)
plt.show()


# In[21]:


print(f"There are {df.distance[df.distance == 0 ].count()} trip records with 0 miles distance")


# ### 6.Speed

# In[22]:


plt.figure(figsize = (20,5))
sns.boxplot(df.speed)
plt.show()


# ### 7.Total trips Per Hour

# In[23]:


def clock(ax, radii, title, color):
    N = 24
    bottom = 2

    # create theta for 24 hours
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)


    # width of each bin on the plot
    width = (2*np.pi) / N
    
    bars = ax.bar(theta, radii, width=width, bottom=bottom, color=color, edgecolor="#999999")

    # set the lable go clockwise and start from the top
    ax.set_theta_zero_location("N")
    # clockwise
    ax.set_theta_direction(-1)

    # set the label
    ax.set_xticks(theta)
    ticks = ["{}:00".format(x) for x in range(24)]
    ax.set_xticklabels(ticks)
    ax.set_title(title)


# In[24]:


plt.figure(figsize = (15,15))
ax = plt.subplot(2,2,1, polar=True)
    # make the histogram that bined on 24 hour
radii = np.array(df['pickup_hour'].value_counts(sort = False).tolist(), dtype="int64")
# radii = np.array(df['pickup_hour'].value_counts().tolist(), dtype="int64")
title = "Trips per Hour"
clock(ax, radii, title, "blue")


# ### 8.Total trips per weekday

# In[25]:


plt.figure (figsize = (14,5))
line=df['weekday_num'].value_counts().plot(kind='bar',title="weekday_num",color = ('blue', 'green', 'red', 'orange','yellow', 'purple'))
plt.xlabel(' WeekDay ')
plt.ylabel('Trip counts')
plt.title('Trips per Day',fontsize = 20)
plt.show()


# In[26]:


n = sns.FacetGrid(df, col='weekday_num')
n.map(plt.hist, 'pickup_hour')
plt.show()


# - Taxi pickups increased in the late night hours over the weekend possibly due to more outstation rides or for the late night leisures nearby activities.
#  - Early morning pickups i.e before 5 AM have increased over the weekend in comparison to the office hours pickups i.e. after 7 AM which have decreased due to obvious reasons.
#  - Taxi pickups seems to be consistent across the week at 15 Hours i.e. at 3 PM.

# ### 9.Total trips per month

# In[27]:


plt.figure(figsize = (19,5))
line=df['month'].value_counts().plot(kind='bar',title="Months",color = ('blue', 'green', 'red', 'orange','yellow', 'purple'))
plt.ylabel('Trip Counts',fontsize = 15)
plt.xlabel('month',fontsize = 15)
plt.title('Trips per Month',fontsize = 20)
plt.show()


# ### Bivariate Analysis
# ### 1.Trip Duration per hour

# In[28]:


plt.figure(figsize=(14, 5))
group1 = df.groupby('pickup_hour').trip_duration.mean().reset_index()
sns.pointplot(x='pickup_hour', y='trip_duration', data=group1)
plt.ylabel('Trip Duration (seconds)')
plt.xlabel('Pickup Hour')
plt.title('Trip Duration per Hour')
plt.show()


# - Average trip duration is lowest at 6 AM when there is minimal traffic on the roads.
# - Average trip duration is generally highest around 3 PM during the busy streets.
# - Trip duration on an average is similar during early morning hours i.e. before 6 AM & late evening hours i.e. after 6 PM.

# ### 2.Trip duration per WeekDay

# In[29]:


plt.figure(figsize=(14, 5))
group2 = df.groupby('weekday_num').trip_duration.mean()
sns.pointplot(x=group2.index, y=group2.values)
plt.ylabel('Trip Duration (seconds)')
plt.xlabel('Weekday')
plt.title('Trip Duration per WeekDay')
plt.show()


# ### 3.Trip duration per Month

# In[30]:


plt.figure(figsize = (14,5))
group3 = df.groupby('month').trip_duration.mean()
sns.pointplot(x=group3.index, y=group3.values)
plt.ylabel('Trip Duration (seconds)')
plt.xlabel('Month')
plt.title('Trip Duration per Month')
plt.show()


# - We can see an increasing trend in the average trip duration along with each subsequent month. 
# - The duration difference between each month is not much. It has increased gradually over a period of 6 months.
# - It is lowest during february when winters starts declining.
# - There might be some seasonal parameters like wind/rain which can be a factor of this gradual increase in trip duration over a period. Like May is generally the considered as the wettest month in NYC and which is inline with our visualization. As it generally takes longer on the roads due to traffic jams during rainy season. So natually the trip duration would increase towards April May and June.

# ### 4.Trip duration per vendor

# In[31]:


group4 = df.groupby('vendor_id').trip_duration.mean()
sns.barplot(x=group4.index, y=group4.values)
plt.ylabel('Trip Duration (seconds)')
plt.xlabel('Vendor')
plt.title('Trip Duration per Vendor')
plt.show()


# Vendor 2 takes the crown. Average trip duration for vendor 2 is higher than vendor 1. 

# ### 5.Distance per hour

# In[32]:


plt.figure(figsize = (14,5))
group5 = df.groupby('pickup_hour').distance.mean()
sns.pointplot(x=group5.index, y=group5.values)
plt.ylabel('Distance (mile)')
plt.title('Distance per Hour')
plt.show()


# - Trip distance is highest during early morning hours which can account for some things like:
#  1. Outstation trips taken during the weekends.
#  2. Longer trips towards the city airport which is located in the outskirts of the city.
# - Trip distance is fairly equal from morning till the evening varying around 2 - 2.5 mile.
# - It starts increasing gradually towards the late night hours starting from evening till 5 AM and decrease steeply towards morning.

# ### 6.Distance per WeekDay

# In[33]:


plt.figure(figsize = (14,5))
group6 = df.groupby('weekday_num').distance.mean()
sns.pointplot(x=group6.index, y=group6.values)
plt.ylabel('Distance (mile)')
plt.title('Distance per WeekDay')
plt.show()


# So it's a fairly equal distribution with average distance metric verying around 2 mile/h with Sunday being at the top may be due to outstation trips or night trips towards the airport.

# ### 7.Distance per Month

# In[34]:


plt.figure(figsize = (14,5))
group7 = df.groupby('month').distance.mean()
sns.pointplot(x=group7.index, y=group7.values)
plt.ylabel('Distance (mile)')
plt.xlabel('Month')
plt.title('Distance per Month')
plt.show()


# Here also the distibution is almost equivalent, varying mostly around 3.5 km/h with 5th month being the highest in the average distance and 2nd month being the lowest.

# ### 8.Distance per Vendor

# In[35]:


group8 = df.groupby('vendor_id').distance.mean()
sns.barplot(x=group8.index, y=group8.values)
plt.ylabel("Distance mile")
plt.xlabel("Vendor")
plt.title('Distance per Vendor')
plt.show()


# ### 9.Distance v/s Trip duration

# In[36]:


plt.figure(figsize = (10,5))
plt.scatter(df.trip_duration, df.distance , s=5, alpha=1)
plt.ylabel('Distance')
plt.xlabel('Trip Duration')
plt.title('Distance v/s Trip Duration')
plt.show()


# Let's focus on the graph area where distance is < 100 mile and duration is < 1000 seconds.

# In[37]:


plt.figure(figsize = (10,5))
dur_dist = df.loc[(df.distance < 100) & (df.trip_duration < 1000), ['distance','trip_duration']]
plt.scatter(dur_dist.trip_duration, dur_dist.distance , s=1, alpha=0.5)
plt.ylabel('Distance')
plt.xlabel('Trip Duration')
plt.title('Distance v/s Trip Duration')
plt.show()


# - There should have been a linear relationship between the distance covered and trip duration on an average but we can see dense collection of the trips in the lower right corner which showcase many trips with the inconsistent readings.
# 
# We should remove those trips which covered 0 mile distance but clocked more than 1 minute to make our data more consistent for predictive model. Because if the trip was cancelled after booking, than that should not have taken more than a minute time. This is our assumption.

# ### 10.Average speed per hour

# In[38]:


plt.figure(figsize = (14,5))
group9 = df.groupby('pickup_hour').speed.mean()
sns.pointplot(x=group9.index, y=group9.values)
plt.xlabel('Pick Up Hours')
plt.ylabel('Speed mile/h')
plt.title('Average Speed per Hour')
plt.show()


# - The average trend is totally inline with the normal circumstances.
# - Average speed tend to increase after late evening and continues to increase gradually till the late early morning hours.
# - Average taxi speed is highest at 5 AM in the morning, then it declines steeply as the office hours approaches.
# - Average taxi speed is more or less same during the office hours i.e. from 8 AM till 6PM in the evening.

# ### 11.Average speed per weekday

# In[39]:


plt.figure(figsize = (14,5))
group10 = df.groupby('weekday_num').speed.mean()
sns.pointplot(x=group10.index, y=group10.values)
plt.xlabel('Pick Up WeekDay')
plt.ylabel('Speed mile/h')
plt.title('Average Speed per WeekDay')
plt.show()


# - Average taxi speed is higher on weekend as compared to the weekdays which is obvious when there is mostly rush of office goers and business owners.
# - Even on monday the average taxi speed is shown higher which is quite surprising when it is one of the most busiest day after the weekend. There can be several possibility for such behaviour
#  1. Lot of customers who come back from outstation in early hours of Monday before 6 AM to attend office on time.
#  2. Early morning hours customers who come from the airports after vacation to attend office/business on time for the coming week.
# - There could be some more reasons as well which only a local must be aware of. 
# - We also can't deny the anomalies in the dataset. which is quite cumbersome to spot in such a large dataset.

# ### 12.Passenger count per vendor

# In[40]:


group9 = df.groupby('vendor_id').passenger_count.mean()
sns.barplot(x=group9.index, y=group9.values)
plt.ylabel('Passenger count')
plt.xlabel('Vendor Id')
plt.title('Passenger Count per Vendor')
plt.show()


# Clear difference between the two operators for the average passenger count in all trips. It seems that vendor 2 trips generally consist of 2 passengers as compared to the vendor 1 with 1 passenger. Let's bifurcate it further.

# ### 13.Pick Up Points v/s Dropoff Points

# In[41]:


city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True,figsize = (12,5))
ax[0].scatter(df['pickup_longitude'].values, df['pickup_latitude'].values,
color='blue', s=1, label='train', alpha=0.1)
ax[1].scatter(df['dropoff_longitude'].values, df['dropoff_latitude'].values,
color='green', s=1, label='train', alpha=0.1)
ax[1].set_title('Drop-off Co-ordinates')
ax[0].set_title('Pick-up Co-ordinates')
ax[0].set_ylabel('Latitude')
ax[0].set_xlabel('Longitude')
ax[1].set_ylabel('Latitude')
ax[1].set_xlabel('Longitude')
plt.ylim(city_lat_border)
plt.xlim(city_long_border)
plt.show()


# - In the Pickup plot we can see the spread is mostly concentrated on Manhattan area.
#  - We can say Manhattan is quite populated for a pickup spot.
#  - Whereas the drop zone is quite spreaded out compared to pickup.
#  - As analysed in distance analysis the average distance is 2.1 miles which explain the heavy intensity of drop in Manhattan itself.

# #### Clustering

# In[42]:


coords = np.vstack((df[['pickup_latitude', 'pickup_longitude']].values,
                    df[['dropoff_latitude', 'dropoff_longitude']].values))


# In[43]:


from sklearn.cluster import MiniBatchKMeans
sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])
df.loc[:, 'pickup_cluster'] = kmeans.predict(df[['pickup_latitude', 'pickup_longitude']])
df.loc[:, 'dropoff_cluster'] = kmeans.predict(df[['dropoff_latitude', 'dropoff_longitude']])


# In[44]:


fig, ax = plt.subplots(ncols=1, nrows=1)
ax.scatter(df.pickup_longitude.values, df.pickup_latitude.values, s=10, lw=0,
           c=df.pickup_cluster.values, cmap='tab20', alpha=0.2)
ax.set_xlim(city_long_border)
ax.set_ylim(city_lat_border)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Pickup v/s Dropoff Cluster')
plt.show()


# As we can see, the clustering results in a partition which is somewhat similar to the way NY is divided into different neighborhoods. We can see Upper East and West side of Central park in gray and pink respectively. West midtown in blue, Chelsea and West Village in brown, downtown area in blue, East Village and SoHo in purple.
# 
# The airports JFK and La LaGuardia have there own cluster, and so do Queens and Harlem. Brooklyn is divided into 2 clusters, and the Bronx has too few rides to be separated from Harlem.

# ## Feature Engineering
# After looking at the dataset from different perspectives. Let's prepare our dataset before training our model. Since our dataset do not contain very large number of dimensions. We will first try to use feature selection instead of the feature extraction technique.

# ### 1.Feature Selection
# We will use technique to select the best features to train our model.
#     
#     Here the biggest question is which columns are useful for model train

# In[45]:


df.dtypes


# - id column has unique values which means its no use to take id in model training
#  - Columns such as pickup_datetime,dropoff_datetime has object dtype and the values are datetype which may not able to evaluate by model which can affect in accuracy.
#  - Therefore pickup_datetime is been converted into weekday, month, weekday_num, pickup_hour
#  - pickup_longitude,pickup_latitude and dropoff_longitude,dropoff_latitude are the columns which are dependent on each other.
#  - This columns does not mean much if pass as individual columns.
#  - Other than this store_and_fwd_flag is just a service info column.
#  - Excluding this  columns the other columns such as vendor_id, passenger_count, distance, speed correlate with duration which can be used in model train.

# In[46]:


from sklearn.model_selection import train_test_split
x = df.iloc[:, [1, 4, 12, 13, 14, 15, 16, 17, 18]].values
y = df.iloc[:,10].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state = 7294)


# In[47]:


from scipy.stats import pearsonr
df1 = pd.DataFrame(np.concatenate((x_train,y_train.reshape(len(y_train),1)),axis=1))
df1.columns = df1.columns.astype(str)

features = df1.iloc[:,:9].columns.tolist()
target = df1.iloc[:,9].name

correlations = {}
for f in features:
    data_temp = df1[[f,target]]
    x1 = data_temp[f].values
    x2 = data_temp[target].values
    key = f + ' vs ' + target
    correlations[key] = pearsonr(x1,x2)[0]
    
data_correlations = pd.DataFrame(correlations, index=['Value']).T
data_correlations.loc[data_correlations['Value'].abs().sort_values(ascending=False).index]


# Column 7 and 8 shows negative correlation which means its not useful in model training. It can leave reverse impact on model

# ### 2.Feature Extraction

# In[52]:


x = df.iloc[:, [1, 4, 12, 13, 14, 15, 16]].values
y = df.iloc[:,10].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state = 7294)


# In[53]:


from sklearn.decomposition import PCA
pca = PCA().fit(x_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("number of components")
plt.ylabel("Cumulative explained variance")
plt.show()


# In[54]:


arr = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
list(zip(range(1,len(arr)), arr))


# Here we can see that 6 variables are sufficient for capturing atleast 99% of the variance in the training dataset. Hence we will use the same set of variables(i.e this model does not require the use of PCA).

# ### 3.Correlation Analysis
# ***
# 
# Correlation analysis is a method of statistical evaluation used to study the strength of a relationship between two or more, numerically measured, continuous variables. This analysis is useful when we need to check if there are possible connections between variables. We will utilize Heatmap for our analysis.

# ### Heatmap
# A heatmap is a graphical representation of data that uses a system of color-coding to represent statistical relationship between different values.

# In[55]:


plt.figure(figsize=(15,15))
corr = pd.DataFrame(x_train[:,0:]).corr()
corr.index = pd.DataFrame(x_train[:,0:]).columns
sns.heatmap(corr, cmap='RdYlGn', vmin=-1, vmax=1, square=True)
plt.title("Correlation Heatmap", fontsize=16)
plt.show()


# - Some combinations of features shows slight correlation.
#  - But most of the features shows no correlation
#  - There is no negative correlation

# ## Model
# ***
# We need a model to train on our dataset to serve our purpose of prediciting the NYC taxi trip duration given the other features as training and test set. Since our dependent variable contains continous values so we will use regression technique to predict our output.
# We will try almost every Regression model (except SVR)

# ### 1.Data Cleaning
# ***
# 
#  - As we analysed there is no null value but there are many outliers in Speed, Distance, and Trip Duration.

# ### 2.Model Training

# ### 1.Multiple Linear Regression
# It is used to explain the relationship between one continuous dependent variable and two or more independent variables.

# In[51]:


from sklearn.linear_model import LinearRegression
import time
start_time = time.time()
lm_regression = LinearRegression()
lm_regression = lm_regression.fit(x_train, y_train)
end_time = time.time()
lm_time = (end_time - start_time)
print(f"Time taken to train linear regression model : {lm_time} seconds")


# In[56]:


trips = lm_regression.predict(x_test)


# In[57]:


predictions = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': trips.flatten()})


# In[59]:


predictions


# #### Accuracy Metrics

# In[60]:


predictions.sample(20).plot(kind='bar',figsize=(14,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# Lets check the r2_score value

# In[62]:


from sklearn.metrics import r2_score
lm_score = r2_score(y_test, trips)
print(lm_score)


# ### 2.Decision Tree
# 
# Decision tree  is one of the predictive modeling approaches used in statistics, data mining and machine learning. It uses a decision tree (as a predictive model) to go from observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves)

# In[66]:


from sklearn.tree import DecisionTreeRegressor
start_time = time.time()
dt_regression = DecisionTreeRegressor()
dt_regression = dt_regression.fit(x_train, y_train)
end_time = time.time()
dt_time = (end_time - start_time)
print(f"Time taken to train Decision tree model : {dt_time} seconds")


# In[67]:


trips = dt_regression.predict(x_test)


# In[68]:


predictions = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': trips.flatten()})


# In[69]:


predictions


# #### Accuracy Metrics

# In[70]:


predictions.sample(20).plot(kind='bar',figsize=(14,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


#  From the above graph, we can see that difference is quite low between predicted values and Actual values 
#  
#  Lets check the r2_score value

# In[72]:


dt_score = r2_score(y_test, trips)
print(dt_score)


# - The r2 score is best but the time taken for execution is high
#  
# So, Let's check if we can get same accuracy with less execution time in next model.

# ### 3.Random Forest
# 
# Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

# In[74]:


from sklearn.ensemble import RandomForestRegressor
start_time = time.time()
rf_regression = RandomForestRegressor()
rf_regression = rf_regression.fit(x_train, y_train)
end_time = time.time()
rf_time = (end_time - start_time)
print(f"Time taken to train Random Forest model : {rf_time} seconds")


# In[75]:


trips = rf_regression.predict(x_test)


# In[76]:


predictions = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': trips.flatten()})


# In[77]:


predictions


# #### Accuracy Metrics

# In[78]:


predictions.sample(20).plot(kind='bar',figsize=(14,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[79]:


rf_score = r2_score(y_test, trips)
print(rf_score)


# ### Model Comparison

# In[80]:


r2 = [lm_score, dt_score, rf_score]
tm = [lm_time, dt_time, rf_time]
comp = pd.DataFrame({'Time': tm, 'Accu': r2})


# In[81]:


label = ['LM', 'DT', 'RF']
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(12,5))
ax = comp['Time'].plot(kind='bar',title="Time",ax=axes[0],color = (1, 0.5, 0.13))
ax1 = comp['Accu'].plot(kind='bar',title="Accuarcy",ax=axes[1])
ax.set_ylabel("Time (secs)")
ax.set_xlabel('Models')
ax.set_xticklabels(label)
ax1.set_ylabel("Accuracy")
ax1.set_xlabel('Models')
ax1.set_xticklabels(label)
fig.tight_layout()


# From the above fig we can finally decide which Model is best suitable for this dataset.
#  - We should straight away reject Random Forest as it takes the most amount of time.
#  - Linear Regression takes Least amount of time but doesn't give much accuracy.
#  - Decision Tree takes least amount of time.
# 

# ## Conclusion
# ***
# 
# According to the whole data analysis and visualization we have tried three of the best algorithms known and we have came to the conclusion that Decision tree is the best suitable algorithm in this scenario as it gives best accuracy in least amount of time.
# 
# <h4>So, Here we conclude the analysis with the conclusion that <b>Decision Tree</b> algorithm is best suitable algorithm for this dataset.</h4>

# In[ ]:





#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from datetime import datetime
import calendar
from sklearn.inspection import partial_dependence, plot_partial_dependence
from sklearn.ensemble import GradientBoostingRegressor

def main():

    # read in dataset file to a pandas dataframe
    raw_data = pd.read_csv('yellow_tripdata_2017-06.csv')

    # set display to be large enough to include all columns
    pd.set_option('display.max_columns', 500)

    clean_data = clean_preprocess(raw_data)
    clean_data = create_borough(clean_data)
    create_graphs(clean_data)




    learn(raw_data)


def clean_preprocess(raw):
    # convert pick up and dropoff to datetime and total amount to float
    raw['tpep_pickup_datetime'] = pd.to_datetime(raw['tpep_pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
    raw['tpep_dropoff_datetime'] = pd.to_datetime(raw['tpep_dropoff_datetime'], format='%Y-%m-%d %H:%M:%S')
    raw['total_amount'] = raw['total_amount'].astype(float)


    # create fields day, hour, day_of_week, and trip_time
    raw['hour'] = raw.tpep_pickup_datetime.apply(lambda x: x.hour)
    raw['day_of_week'] = raw['tpep_pickup_datetime'].apply(lambda x: calendar.day_name[x.weekday()])
    raw['day'] = raw['tpep_pickup_datetime'].apply(lambda x: x.day)
    raw['trip_time'] = raw['tpep_dropoff_datetime'] - raw['tpep_pickup_datetime']

    #drop trips that don't make sense or aren't useful and drop outliers
    raw = raw[raw['trip_time'] > pd.Timedelta(1, 's')]
    raw = raw[raw['total_amount'] > 0]
    codes = [1,5,6]
    raw = raw[raw.RatecodeID.isin(codes)]
    raw = raw[np.abs(raw.trip_time -raw.trip_time.mean()) <= (3*raw.trip_time.std())]
    raw = raw[np.abs(raw.total_amount - raw.total_amount.mean()) <= (3*raw.total_amount.std())]

    # convert trip time to seconds
    raw['trip_time'] = raw['trip_time'].dt.total_seconds().astype(int)
    return raw




def mps(raw):
    #Function to find the average money per second

    mps_data = raw.groupby(['day', 'hour', 'PULocationID']).agg({
        'trip_time':[np.sum],
        'total_amount': [np.sum]

    })

    mps_data.columns = ['_'.join(col).strip() for col in mps_data.columns.values]
    mps_data = mps_data.reset_index()

    mps_data['mps'] = mps_data['total_amount_sum'] / mps_data['trip_time_sum']
    seaborn.pairplot(mps_data)
    plt.show()

def learn(raw):
    train = raw.sample(frac= .001, random_state = 1)
    feature_cols = ['day', 'hour', 'PULocationID', 'trip_distance']
    X = train.loc[:, feature_cols]
    y = train.pop("total_amount").values
    est = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, loss='huber', random_state=1)
    est.fit(X,y)
    print("done fitting")
    features = [0,1,2,3]
    plot_partial_dependence(est, X, features, feature_names=feature_cols, n_jobs=3, grid_resolution=20)
    fig = plt.gcf()
    fig.suptitle('partial dependence of main variables')

    plt.show()

def create_borough(raw):
    bdf = pd.read_csv('taxi+_zone_lookup.csv')
    df1 = raw.PULocationID
    raw = (raw.merge(bdf, left_on= "PULocationID", right_on="LocationID"))
    return raw

def create_graphs(raw):

    # create money per second pair plot graphs
    mps(raw)
    #create average fare per day bar graph
    avg_fare_per_day(raw)
    #create average far by zone graph
    avg_fare_by_zone(raw)
    #create average fare per hour graph
    avg_fare_per_hour(raw)
    # create bar graph giving number of trips per day
    num_trips_per_day(raw)
    # create bar graph giving number of trips per zone
    num_trips_per_zone(raw)
    # do the same but for boroughs instead of zones
    avg_fare_by_borough(raw)
    num_trips_borough(raw)

def avg_fare_per_day(raw):
    #group data by the day of week and find the mean fare for each day
    afpd = raw.groupby(['day_of_week']).agg({
        'total_amount':[np.mean]
    })

    #flatten data for ease of use
    afpd.columns = ['_'.join(col).strip() for col in afpd.columns.values]
    afpd = afpd.reset_index()

    #creates the graph
    seaborn.barplot(x='day_of_week', y = 'total_amount_mean', data= afpd)
    plt.show()

def avg_fare_by_zone(raw):
    # group data by pick up location, and find the mean fare for each location
    afbz = raw.groupby('PULocationID').agg({
        'total_amount':[np.mean]
    })

    #flatten data for ease of use
    afbz.columns = ['_'.join(col).strip() for col in afbz.columns.values]
    afbz = afbz.reset_index()

    #sort by avg fare
    afbz = afbz.sort_values(by='total_amount_mean', ascending= False)
    #make graphdf variable that contains the top 20 values
    graphDF = afbz.head(20)
    # plot graphdf variable for ease of visualization
    seaborn.barplot(x='PULocationID', y='total_amount_mean', data= graphDF)
    plt.show()

def avg_fare_per_hour(raw):
    # group data by the hour the trip started and find mean fare for each hour
    afph = raw.groupby('hour').agg({
        'total_amount':[np.mean]
    })

    #flatten data for ease of use
    afph.columns = ['_'.join(col).strip() for col in afph.columns.values]
    afph = afph.reset_index()

    # sort by time of day
    afph = afph.sort_values(by= 'hour')
    #graph values
    seaborn.barplot(x= 'hour', y= 'total_amount_mean', data= afph)
    plt.show()

def num_trips_per_day(raw):
    # group the data by the day of the week and find the count for each day
    # put these aggregates into what was the hour field, and rename it Num_trips
    trips_day = raw.groupby(['day_of_week'])['hour'].count().reset_index().rename(columns = {
        'hour':'Num_trips'})

    #graph
    seaborn.barplot(x= 'day_of_week', y = "Num_trips", data=trips_day)
    plt.show()

def num_trips_per_zone(raw):
    # group the data by the pickup zone and find the count for each day
    # put these aggregates into what was the hour field, and rename it Num_trips
    trips_zone = raw.groupby(['PULocationID']).count().reset_index().rename(columns = {
        'hour':'Num_trips'})

    #sort values by the most number of trips
    trips_zone = trips_zone.sort_values(by= 'Num_trips', ascending=False)
    #graph the top 20 of these trips
    seaborn.barplot(x = 'PULocationID', y= 'Num_trips', data = trips_zone.head(20))

    plt.show()


def avg_fare_by_borough(raw):
    #group the data by the borough and find the average fare cost per borough
    afbb = raw.groupby('Borough').agg({'total_amount':[np.mean]})

    afbb.columns = ['_'.join(col).strip() for col in afbb.columns.values]
    afbb = afbb.reset_index()

    seaborn.barplot(x="Borough", y='total_amount_mean', data=afbb)
    plt.show()

def num_trips_borough(raw):
    # group the data by the borough and find the count for each day
    # put these aggregates into what was the hour field, and rename it Num_trips
    trips_borough = raw.groupby(['Borough']).count().reset_index().rename(columns={
        'hour': 'Num_trips'})
    seaborn.barplot(x='Borough', y='Num_trips', data=trips_borough)
    plt.show()

if __name__ == "__main__":
    main()
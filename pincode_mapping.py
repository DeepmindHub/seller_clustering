import mysql.connector
from pandas.io import sql
import pandas as pd
import numpy as np
import datetime
import sys
sys.path.append('/home/ubuntu/ankur/config')
import dbConfig_pr as dbConfig
from DistanceCalculation import computeHaversineDistance as chd


def main():
    cnx = mysql.connector.connect(user=dbConfig.USER, password=dbConfig.PWD,
                                  host=dbConfig.HOST, database=dbConfig.DATABASE)
    cluster = getClusters(cnx)
    pincode = pd.read_csv('../google_places/pincode_data_v2.csv')
    pincode['cluster_id'] = -1
    pincode['cluster_name'] = ''
    pincode['cluster_latitude'] = -1
    pincode['cluster_longitude'] = -1
    pincode['distance'] = -1
    pincode = assignCluster(pincode, cluster)
    pincode.to_csv('pincode_data_v3.csv', index=False)


def getClusters(cnx):
    query = "select c.id cluster_id ,c.cluster_name, c.latitude ,c.longitude \
        from coreengine_cluster as c where cluster_name not like '%Test%' and c.latitude!=0 \
        and c.longitude!=0 and cluster_name not like '%hub%'"
    return sql.read_sql(query, cnx)


def assignCluster(pincode, cluster):
    for i in xrange(len(pincode)):
        if (pincode.loc[i, 'lat_lng_correct'] == False) or \
                (pincode.loc[i, 'city'] in ['Jaipur', 'Pune']):
            continue
        end_lat = pincode.loc[i, 'latitude']
        end_lng = pincode.loc[i, 'longitude']
        dist_list = []
        for j in xrange(cluster.shape[0]):
            st_lat = cluster.loc[j, 'latitude']
            st_lng = cluster.loc[j, 'longitude']
            dist = chd(st_lat, st_lng, end_lat, end_lng)
            dist_list.append(dist)
            # cc_list.append(cluster.iloc[j]['cluster_id'])
        pincode.loc[i, 'distance'] = min(dist_list)
        ind = dist_list.index(min(dist_list))
        pincode.loc[i, 'cluster_id'] = cluster.loc[ind, 'cluster_id']
        pincode.loc[i, 'cluster_name'] = cluster.loc[ind, 'cluster_name']
        pincode.loc[i, 'cluster_latitude'] = cluster.loc[ind, 'latitude']
        pincode.loc[i, 'cluster_longitude'] = cluster.loc[ind, 'longitude']
    return pincode

if __name__ == '__main__':
    main()

#!/usr/bin/python

import pandas as pd
import mysql.connector as sqlcon
from pandas.io import sql
import random as r
from DistanceCalculation import computeHaversineDistance
import math as m
import multiprocessing as mp
import numpy as np
import sys
sys.path.append('/home/ubuntu/yadu/config')
import dbConfig_pr as dbConfig


def main():
    cnx = sqlcon.connect(
        user=dbConfig.USER, password=dbConfig.PWD, host=dbConfig.HOST, database=dbConfig.DATABASE)
    sellers = pd.read_csv('../zomato_scraping/data/bangalore_data_v1.csv')
    sfx_sellers = getSfxSellers('BLR', cnx)
    ind = (sellers['Name'] + sellers['Locality']
           ).isin(sfx_sellers['name'] + sfx_sellers['locality'])
    ind = ~ind
    sellers = sellers.loc[ind]
    sellers['CostFor2'] = sellers['CostFor2'].str.replace(',', '').astype('float')
    sellers = sellers.loc[sellers['CostFor2'] >= 500]
    sellers = sellers.loc[(sellers['Latitude'] > 10) & (sellers['Latitude'] < 20)]
    sellers = sellers.loc[(sellers['Longitude'] > 70) & (sellers['Longitude'] < 80)]
    sellers['wLatitude'] = sellers['Latitude'] * (sellers['Chain'] + 1)
    sellers['wLongitude'] = sellers['Longitude'] * (sellers['Chain'] + 1)
    print 'Seller count: ', len(sellers)
    n = len(sellers)/20

    # clusters = initClusters(sellers, n)
    # print clusters
    # sellers['cluster'] = sellers.apply(lambda x: assignClusters(clusters, x), axis=1)
    # print sellers[['Latitude', 'Longitude', 'cluster']]
    # clusters = updateClusters(clusters, sellers)
    # print clusters

    clusters, sellers = getClusters(sellers, n)

    # sellers = pd.read_csv('BLR_PotSellers.csv')
    # clusters = pd.read_csv('BLR_PotClusters.csv')
    # clusters.index = clusters.cluster_id
    # clusters.drop('cluster_id', axis=1, inplace=True)

    clusters, sellers = rankClusters(clusters, sellers)
    sellers.to_csv('BLR_PotSellers.csv', index=False, encoding='utf-8')
    clusters.to_csv('BLR_PotClusters.csv', index=True, encoding='utf-8')
    sellers.to_csv('BLR_PotSellers_v2.csv', index=False, encoding='utf-8', sep='|')
    clusters.to_csv('BLR_PotClusters_v2.csv', index=True, encoding='utf-8', sep='|')


def getSfxSellers(city, cnx):
    query = "select z.name, \
                z.locality \
            from z_sellerprofile z, \
                coreengine_sfxseller ss, \
                coreengine_cluster c \
            where z.id=ss.id and \
                ss.cluster_id=c.id and \
                c.city='BLR' and \
                z.link is not null;"
    return sql.read_sql(query, cnx)


def getClusters(sellers, n):
    sellers_file = pd.ExcelWriter('sellers.xlsx')
    clusters_file = pd.ExcelWriter('clusters.xlsx')
    sellers_f = None
    clusters_f = None
    rmsd_o = 100
    for r in range(5):
        print 'Run', r+1
        clusters = initClusters(sellers, n)
        sellers['cluster'] = None
        i = 0
        mv = 1
        while mv > 0.05:
            i += 1
            print 'Iteration', i
            sellers['cluster'] = sellers.apply(lambda x: assignClusters(clusters, x), axis=1)
            clusters = updateClusters(clusters, sellers)
            clusters = clusters.loc[(clusters.Latitude.notnull()) & (clusters.Longitude.notnull())]
            mv = clusters['movement'].max()
            print 'Movement: ', mv
        rmsd = getRMSD(clusters, sellers)
        print 'Old RMSD:', rmsd_o, 'New RMSD:', rmsd
        if rmsd < rmsd_o:
            print 'Updating Clusters'
            sellers_f = sellers.copy()
            clusters_f = clusters.copy()
            sellers_f.to_excel(sellers_file, 'Run'+str(r+1), index=False, encoding='utf-8')
            clusters_f.to_excel(clusters_file, 'Run'+str(r+1), index=False, encoding='utf-8')
            rmsd_o = rmsd
    sellers_file.save()
    clusters_file.save()
    return clusters_f, sellers_f


def initClusters(sellers, n):
    ind = r.sample(sellers.index, n)
    clusters = sellers.loc[ind, ['Latitude', 'Longitude']]
    clusters.index = range(len(clusters))
    clusters.index.name = 'cluster_id'
    clusters['movement'] = 1000
    return clusters


def assignClusters(clusters, seller):
    cluster = clusters.apply(lambda x: computeHaversineDistance(x['Latitude'],
                                                                x['Longitude'], seller['Latitude'], seller['Longitude']), axis=1).idxmin()
    return cluster


def updateClusters(clusters, sellers):
    clusters_o = clusters.copy()
    clusters['Latitude'] = pd.pivot_table(
        sellers, values='wLatitude', index='cluster', aggfunc='sum') / pd.pivot_table(
        sellers, values='Chain', index='cluster', aggfunc=lambda x: sum(x) + len(x))
    clusters['Longitude'] = pd.pivot_table(
        sellers, values='wLongitude', index='cluster', aggfunc='sum') / pd.pivot_table(
        sellers, values='Chain', index='cluster', aggfunc=lambda x: sum(x) + len(x))
    clusters['movement'] = pd.Series(clusters.index).apply(lambda x: computeHaversineDistance(
        clusters_o.loc[x, 'Latitude'], clusters_o.loc[x, 'Longitude'], clusters.loc[x, 'Latitude'],
        clusters.loc[x, 'Longitude']))
    return clusters


def getRMSD(clusters, sellers):
    x = sellers.iloc[0]
    dist = sellers.apply(lambda x: computeHaversineDistance(x['Latitude'], x['Longitude'],
                                                            clusters.loc[x['cluster'], 'Latitude'], clusters.loc[x['cluster'], 'Longitude']),
                         axis=1)
    return m.sqrt((dist*dist).mean())


def rankClusters(clusters, sellers):
    clusters['seller_count'] = pd.Series(clusters.index).apply(
        lambda x: (sellers['cluster'] == x).sum())
    clusters['chain_count'] = pd.Series(clusters.index).apply(
        lambda x: ((sellers['cluster'] == x) & (sellers['Chain'] == 1)).sum())
    sellers['dist'] = sellers.apply(lambda x: computeHaversineDistance(x['Latitude'], x['Longitude'], clusters.loc[
        int(x['cluster']), 'Latitude'], clusters.loc[int(x['cluster']), 'Longitude']), axis=1)
    sellers['wDistSq'] = sellers['dist'] * sellers['dist'] * (sellers['Chain'] + 1)
    clusters['rmsd'] = (pd.pivot_table(
        sellers, values='wDistSq', index='cluster', aggfunc='sum') / pd.pivot_table(
        sellers, values='Chain', index='cluster', aggfunc=lambda x: sum(x) + len(x))).apply(np.sqrt)
    clusters['max_dist'] = pd.pivot_table(sellers, values='dist', index='cluster', aggfunc='max')
    return clusters, sellers

if __name__ == "__main__":
    r.seed(111)
    main()
    # sellers = pd.DataFrame([[1, 1], [1, 2], [1, 3], [1, 5], [2, 3], [2, 1], [2, 6]])
    # sellers.columns = ['cluster', 'Latitude']
    # clusters = pd.DataFrame([0, 0], index=[1, 2], columns=['Latitude'])
    # clusters = updateClusters(clusters, sellers)
    # print clusters

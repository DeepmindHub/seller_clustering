import pandas as pd
import csv
import math
import random
import DistanceCalculation as dc
import xlwt
import os
import xlrd
import io
import GoogleDistanceCalculation as gd
random.seed(432)

DistMethod = 1


def main():
    print 'RawData file should be pipe separated'
    fileName = raw_input('Enter CSV file Name : ')
    indL = raw_input(
        'Enter index (starting from 0) for (sellerID,Latitude,longitude,avgOrders) in same order')
    exec "indL = " + indL
    print indL
    # print indL[0]
    NoOfCluster = None
    while NoOfCluster == None:
        try:
            nExp = raw_input('Enter trial value(s) for no. of clusters:')
            exec "NoOfCluster = " + nExp
            # n_start = input('Enter lower limit : ')
            # n_end = input('Enter upper limit : ')+1
        except:
            print 'Invalid Expression!!'
    if isinstance(NoOfCluster, int):
        NoOfCluster = [NoOfCluster]
    print 'Initializing Process...'
    print 'Reading Data...'
    df = pd.read_csv(fileName, encoding='utf-8', sep=',')
    a = list(df.columns)
    col_list = [a[indL[0]], a[indL[1]], a[indL[2]], a[indL[3]]]
    execute(df, NoOfCluster, indL, col_list)


def random_cluster(df, n):
    cc_df = df.ix[random.sample(df.index, n)]
    cc_df.drop('id', axis=1, inplace=True)
    cc_df.rename(columns={'latitude': 'clusterLat', 'longitude': 'clusterLong'}, inplace=True)
    cc_df['ClusterID'] = range(1, n+1)
    return cc_df


def Assign_Cluster(df, cc_df):
    df['MinDistance'] = 1
    df['ClusterID'] = 1
    # print cc_df
    for i in xrange(df.shape[0]):
        end_lat = df.iloc[i]['latitude']
        end_lng = df.iloc[i]['longitude']
        dist_list = []
        cc_list = []
        for j in xrange(cc_df.shape[0]):
            st_lat = cc_df.iloc[j]['clusterLat']
            st_lng = cc_df.iloc[j]['clusterLong']
            dist = Compute_Dist(st_lat, st_lng, end_lat, end_lng)
            # print dist, cc_df.iloc[j]['id']
            dist_list.append(dist)
            cc_list.append(cc_df.iloc[j]['ClusterID'])
        # AssClu_list=[dist_list,cc_list]
        # print dist_list, min(dist_list)
        # print cc_list
        df.ix[i, 'MinDistance'] = min(dist_list)
        # print df.iloc[i]['MinDistance']
        df.ix[i, 'ClusterID'] = cc_list[dist_list.index(min(dist_list))]
        # print df.ix[i,'ClusterID']
    return df


def NewClusterLocation(df, PreviousCC_df):
    df1 = df.groupby(['ClusterID']).sum()
    # print df1.shape
    # print df1
    # df1.avgOrders=df1.avgOrders.replace(['Nan'],1)
    NewCluster_df = pd.DataFrame(
        [[0]*3]*df1.shape[0], columns=['ClusterID', 'clusterLat', 'clusterLong'])
    # print NewCluster_df.shape
    NewCluster_df['clusterLat'] = (df1['WeightedLat']/df1['avgOrders']).values
    NewCluster_df['clusterLong'] = (df1['WeightedLong']/df1['avgOrders']).values
    NewCluster_df.clusterLat = NewCluster_df.clusterLat.replace(['NaN'], 1)
    NewCluster_df.clusterLong = NewCluster_df.clusterLong.replace(['NaN'], 1)
    PreviousCC_df['Movement'] = 0
    for i in xrange(NewCluster_df.shape[0]):
        NewCluster_df.ix[i, 'ClusterID'] = 101+i
        st_lat = PreviousCC_df.iloc[i]['clusterLat']
        st_lng = PreviousCC_df.iloc[i]['clusterLong']
        end_lat = NewCluster_df.iloc[i]['clusterLat']
        end_lng = NewCluster_df.iloc[i]['clusterLong']
        # print 100+i,st_lat, st_lng, end_lat, end_lng
        dist = Compute_Dist(st_lat, st_lng, end_lat, end_lng)
        PreviousCC_df.ix[i, 'Movement'] = dist
    return NewCluster_df


def InterClusterDist(df, n):
    InterClusterDist = []
    for i in xrange(df.shape[0]):
        st_lat = df.iloc[i]['clusterLat']
        st_lng = df.iloc[i]['clusterLong']
        for j in xrange(i+1, df.shape[0]):
            end_lat = df.iloc[j]['clusterLat']
            end_lng = df.iloc[j]['clusterLong']
            dist = Compute_Dist(st_lat, st_lng, end_lat, end_lng)
            InterClusterDist.append(dist)
    # print InterClusterDist
    MeanInterClusterDist = sum(InterClusterDist)/len(InterClusterDist)
    return MeanInterClusterDist


def get_rmsd(sellerdf, clusterdf):
    temp = sellerdf
    temp['SqDistance'] = temp['MinDistance']**2
    AllRMSD = []
    for i in xrange(clusterdf.shape[0]):
        sqDistSum = temp.loc[
            temp['ClusterID'] == clusterdf.iloc[i]['ClusterID'], 'SqDistance'].sum()
        sellers = temp.loc[temp['ClusterID'] == clusterdf.iloc[i]['ClusterID'], 'id'].count()
        rmsd = (sqDistSum/sellers)**0.5
        AllRMSD.append(rmsd)
    MeanRMSD = sum(AllRMSD)/len(AllRMSD)
    print 'RMSD for Clusters', AllRMSD
    return MeanRMSD


def get_BestCaseIndex(file_URL, ICDistList, RMSDList, k, col_list):
    DiffList = [ICDistList_i-RMSDList_i for ICDistList_i, RMSDList_i in zip(ICDistList, RMSDList)]
    index = DiffList.index(max(DiffList))
    sheetNameAssign = 'ClusterAssign %d' % index
    sheetNameClu = 'Clusters %d' % index
    # wb=xlrd.open_workbook(file_URL)
    wb = pd.ExcelFile(file_URL, encoding='utf-8')
    BestCaseClusters = wb.parse(sheetNameClu)
    BestCaseAsssign = wb.parse(sheetNameAssign)
    BestCaseAsssign, BestCaseClusters = get_stats(BestCaseAsssign, BestCaseClusters)
    BestCaseAsssign.rename(
        columns={'id': col_list[0], 'latitude': 'Seller Lat', 'longitude': 'Seller Long'}, inplace=True)
    BestCaseAsssign.to_csv('BestCaseAssign'+str(DistMethod)+'_n%d' % k+'.csv', encoding='utf-8')
    BestCaseClusters.to_csv('BestCaseClusters'+str(DistMethod)+'_n%d' % k+'.csv', encoding='utf-8')


def execute(df, NoOfCluster, indL, col_list):
    df.rename(columns={df.columns[indL[0]]: 'id', df.columns[indL[1]]: 'latitude', df.columns[
              indL[2]]: 'longitude', df.columns[indL[3]]: 'avgOrders'}, inplace=True)
    # df.columns[indL[]] = ['id', 'latitude', 'longitude', 'avgOrders']
    print df.shape
    print df.shape[0], 'rows read.'
    df.ix[:, 'avgOrders'] = df.ix[:, 'avgOrders'].replace(['NaN'], 10)
    print 'Null Values for avg Orders were replaced with 10.'
    # print df.dtypes
    df['WeightedLat'] = df['avgOrders'].astype('float')*df['latitude']
    df['WeightedLong'] = df['avgOrders'].astype('float')*df['longitude']
    for k in NoOfCluster:
        print '.................for %d clusters....................' % k
        report = 'ClusteringNew_n %d' % k + '.xls'
        writer = pd.ExcelWriter(report)
        ICD_BestCase = []
        RMSD_BestCase = []
        for i in range(5):
            print 'random clustering', i
            RandCluster = random_cluster(df, k)
            # print RandCluster
            print 'Clusters Chosen Randomly...', i
            print RandCluster
            df1 = Assign_Cluster(df, RandCluster)
            # print df1
            print 'Clusters Assigned...', i
            NewCluster_df = NewClusterLocation(df1, RandCluster)
            print 'New Clusters found...'
            print NewCluster_df
            print 'Iterating...'
            df2 = df1
            PreviousCC_df = RandCluster
            count = 0
        # print count
            while PreviousCC_df['Movement'].max() > 0.5:
                print 'Iteration : ', count
                # PreviousCC_df=NewCluster_df
                df2 = Assign_Cluster(df, NewCluster_df)
                # df2 = df2.apply(lambda x: x.str.encode('utf-8'), axis=0)
                NewCluster1_df = NewClusterLocation(df2, NewCluster_df)
                PreviousCC_df = NewCluster_df
                NewCluster_df = NewCluster1_df
                print PreviousCC_df
                count += 1
            MeanIC_Dist = InterClusterDist(PreviousCC_df, k)
            print 'Inter Cluster Distance', i, 'is', MeanIC_Dist
            ICD_BestCase.append(MeanIC_Dist)
            Mean_RMSD = get_rmsd(df2, PreviousCC_df)
            print 'Mean_RMSD', i, 'is', Mean_RMSD
            RMSD_BestCase.append(Mean_RMSD)
            # PreviousCC_df['StatsName']=''
            # PreviousCC_df['StatsValue']=0`
            # PreviousCC_df.ix[0:1,'StatsName']=['RMSD','IC_Dist']
            # PreviousCC_df.ix[0:1,'StatsValue']=[Mean_RMSD,MeanIC_Dist]
            tab_name = 'ClusterAssign %d' % i
            tab_name2 = 'Clusters %d' % i
            df2.to_excel(writer, tab_name, index=True)
            PreviousCC_df.to_excel(writer, tab_name2, index=True)
        writer.save()
        BestCaseIndex = get_BestCaseIndex(report, ICD_BestCase, RMSD_BestCase, k, col_list)


def get_stats(SellerDF, ClusterDF):
    temp = SellerDF
    SellerDF['sqDist'] = SellerDF['MinDistance']**2
    ClusterDF['RMSD'], ClusterDF['TotalSellers'], ClusterDF['TotalavgOrders'], ClusterDF[
        'sellers_1KM'], ClusterDF['Order_1KM'], ClusterDF['RemoteSellerDist'] = [0, 0, 0, 0, 0, 0]
    for i in xrange(ClusterDF.shape[0]):
        sqDistSum = temp.loc[temp['ClusterID'] == ClusterDF.iloc[i]['ClusterID'], 'sqDist'].sum()
        TotalSellers = temp.loc[temp['ClusterID'] == ClusterDF.iloc[i]['ClusterID'], 'id'].count()
        rmsd = (sqDistSum/TotalSellers)**0.5
        TotalOrders = temp.loc[
            temp['ClusterID'] == ClusterDF.iloc[i]['ClusterID'], 'avgOrders'].sum()
        sellers_1KM = temp.loc[(temp['ClusterID'] == ClusterDF.iloc[i]['ClusterID']) & (
            temp['MinDistance'] <= 1.0), 'id'].count()
        Orders_1KM = temp.loc[(temp['ClusterID'] == ClusterDF.iloc[i]['ClusterID']) & (
            temp['MinDistance'] <= 1.0), 'avgOrders'].sum()
        RemoteSellerDist = temp.loc[
            temp['ClusterID'] == ClusterDF.iloc[i]['ClusterID'], 'MinDistance'].max()
        ClusterDF.ix[i, ('RMSD', 'TotalSellers', 'TotalavgOrders', 'sellers_1KM', 'Order_1KM', 'RemoteSellerDist')] = [
            rmsd, TotalSellers, TotalOrders, sellers_1KM, Orders_1KM, RemoteSellerDist]
    return SellerDF, ClusterDF


def execute_withoutRandom(df, NoOfCluster, indL, clusterdf):
    df.rename(columns={df.columns[indL[0]]: 'id', df.columns[indL[1]]: 'latitude', df.columns[
              indL[2]]: 'longitude', df.columns[indL[3]]: 'avgOrders'}, inplace=True)
    # df.columns[indL[]] = ['id', 'latitude', 'longitude', 'avgOrders']
    print df.shape
    print df.shape[0], 'rows read.'
    df.ix[:, 'avgOrders'] = df.ix[:, 'avgOrders'].replace(['NaN'], 10)
    print 'Null Values for avg Orders were replaced with 10.'
    # print df.dtypes
    df['WeightedLat'] = df['avgOrders'].astype('float')*df['latitude']
    df['WeightedLong'] = df['avgOrders'].astype('float')*df['longitude']
    for k in NoOfCluster:
        print '.................for %d clusters....................' % k
        report = 'ClusteringNewWR_n %d' % k + '.xls'
        writer = pd.ExcelWriter(report)
        ICD_BestCase = []
        RMSD_BestCase = []
        i = 0
        print 'random clustering'
        RandCluster = clusterdf
        # print RandCluster
        # print 'Clusters Chosen Randomly...'
        df1 = Assign_Cluster(df, RandCluster)
        # print df1
        print 'Clusters Assigned...'
        NewCluster_df = NewClusterLocation(df1, RandCluster)
        print 'New Clusters found...'
        print NewCluster_df
        print 'Iterating...'
        df2 = df1
        PreviousCC_df = RandCluster
        count = 0
        # print count
        while PreviousCC_df['Movement'].max() > 0.5:
            print 'Iteration : ', count
            # PreviousCC_df=NewCluster_df
            df2 = Assign_Cluster(df, NewCluster_df)
            # df2 = df2.apply(lambda x: x.str.encode('utf-8'), axis=0)
            NewCluster1_df = NewClusterLocation(df2, NewCluster_df)
            PreviousCC_df = NewCluster_df
            NewCluster_df = NewCluster1_df
            print PreviousCC_df
            count += 1
        MeanIC_Dist = InterClusterDist(PreviousCC_df, k)
        print 'Inter Cluster Distance', 'is', MeanIC_Dist
        ICD_BestCase.append(MeanIC_Dist)
        Mean_RMSD = get_rmsd(df2, PreviousCC_df)
        print 'Mean_RMSD', 'is', Mean_RMSD
        RMSD_BestCase.append(Mean_RMSD)
        # PreviousCC_df['StatsName']=''
        # PreviousCC_df['StatsValue']=0`
        # PreviousCC_df.ix[0:1,'StatsName']=['RMSD','IC_Dist']
        # PreviousCC_df.ix[0:1,'StatsValue']=[Mean_RMSD,MeanIC_Dist]
        tab_name = 'ClusterAssign %d' % i
        tab_name2 = 'Clusters %d' % i
        df2.to_excel(writer, tab_name, index=True)
        PreviousCC_df.to_excel(writer, tab_name2, index=True)
    writer.save()
    BestCaseIndex = get_BestCaseIndex(report, ICD_BestCase, RMSD_BestCase, k, indL)


def Compute_Dist(st_lat, st_lng, end_lat, end_lng):
    if DistMethod == 1:
        return dc.computeHaversineDistance(st_lat, st_lng, end_lat, end_lng)
    if DistMethod == 0:
        origins = str(st_lat)+','+str(st_lng)
        destinations = str(end_lat)+','+str(end_lng)
        return gd.compute_google_dist(origins=origins, destinations=destinations, count=5500)[0]

if __name__ == "__main__":
    main()

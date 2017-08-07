# -- coding: utf-8 --
import pandas as pd
import numpy as np
import datetime
import gc
from sklearn.utils import shuffle 

#�Ѷ������մ����һ����������һ�����������ţ���Ŵ�0��ʼ
def orders_prepare(orders):
    max_order=orders.groupby('user_id', as_index=False)['order_number'].max()
    max_order.columns=['user_id','max_order_number']
    xorders=pd.merge(orders, max_order, how='left',on='user_id')
    #�������һ��������Ŷ�Ϊ0������ÿ�����������һ������ĵ���
    xorders['order_number_reverse']=xorders.max_order_number-xorders.order_number
    return xorders

#��������ĵ��ţ��Ѷ�����ΪҪԤ��Ķ������͸ö���֮ǰ��������ʷ����
def split_orders(xorders, split_point, delta_order=-1):
    if delta_order<=0:
        prior_orders=xorders[xorders.order_number_reverse>split_point].copy()
        predict_order=xorders[xorders.order_number_reverse==split_point].copy()
    else:
        prior_orders=xorders[(xorders.order_number_reverse>split_point) & (xorders.order_number_reverse<=(split_point+delta_order))].copy()
        predict_order=xorders[xorders.order_number_reverse==split_point].copy()        
    return (prior_orders, predict_order)

#����ʷ������ҪԤ�ⶩ����ʱ����Ϣ�������
def get_xprior(prior_orders, predict_order, detail_prior, products):
    prior_orders['days_since_prior_order']=prior_orders.days_since_prior_order.fillna(0)
    #�û����׵���ʼ���������ۼӣ��û�ÿһ�����׵����������
    prior_orders['days_since_first_order']=prior_orders.groupby('user_id', as_index=False)['days_since_prior_order'].cumsum()
    #�û����һ�����׵����������
    latest_order=prior_orders.groupby('user_id', as_index=False)[['order_number','days_since_first_order']].max()
    latest_order.columns=['user_id','latest_order_number','latest_order_days_since_first_order']
    xprior_orders=pd.merge(prior_orders, latest_order, how='left',on='user_id')
    xprior_orders=pd.merge(xprior_orders,predict_order[['user_id','order_number','order_dow','order_hour_of_day','days_since_prior_order']], how='left',on='user_id', suffixes=('','_predict'))
    #�û�ÿһ��������ҪԤ��Ķ������������
    xprior_orders['delta_days_to_predict']=xprior_orders.latest_order_days_since_first_order-xprior_orders.days_since_first_order+xprior_orders.days_since_prior_order_predict
    #�û�ÿһ��������ҪԤ��Ķ�������ĵ���
    xprior_orders['delta_order_number_to_predict']=xprior_orders.order_number_predict-xprior_orders.order_number
    xprior_orders['delta_order_dow_to_predict']=np.abs(xprior_orders.order_dow-xprior_orders.order_dow_predict)
    xprior_orders['delta_order_hour_to_predict']=np.abs(xprior_orders.order_hour_of_day-xprior_orders.order_hour_of_day_predict)
    
    xprior=pd.merge(xprior_orders, detail_prior, how='left', on='order_id')
    xprior=pd.merge(xprior, products,how='left', on='product_id')
    return xprior

#ת��Сʱ,ʹʱ�����ȱ��
def convert_hour(hour):
    if 0<=hour<=5:
        return 1
    elif 6<=hour<=8:
        return 2
    elif 9<=hour<=11:
        return 3
    elif 12<=hour<=14:
        return 4
    elif 15<=hour<=18:
        return 5
    elif 19<=hour<=21:
        return 6
    else:
        return 7
 
#����ʷ������ȡ����    
def extract_features(xprior):
    print 'extract feature begin: %s'%datetime.datetime.now()  
    print 'delta days feature: %s'%datetime.datetime.now()
    #user-item features
    #�û����һ�ι���ĳ��Ʒ��ʱ��͵���, Ϊ�˺�����ȡ�������㣬��ʱ��͵����ǰ������һ���������ġ����� ���һ����1��������һ������2
    user_item_delta_last=xprior[['user_id','product_id','delta_days_to_predict', 'delta_order_number_to_predict', 'order_number_reverse']].groupby(['user_id','product_id'],as_index=False).min()
    user_item_delta_last.columns=['user_id','product_id','item_delta_days_last', 'item_delta_order_number_last', 'order_number_reverse']
    #�û����һ�ι���ĳС���ʱ��͵���
    user_aisle_delta_last=xprior[['user_id','aisle_id','delta_days_to_predict', 'delta_order_number_to_predict','order_number_reverse']].groupby(['user_id','aisle_id'], as_index=False).min()
    user_aisle_delta_last.columns=['user_id','aisle_id','aisle_delta_days_last', 'aisle_delta_order_number_last','order_number_reverse']
    #�û����һ�ι���ĳ�����ʱ��͵���
    user_department_delta_last=xprior[['user_id','department_id','delta_days_to_predict', 'delta_order_number_to_predict','order_number_reverse']].groupby(['user_id','department_id'], as_index=False).min()
    user_department_delta_last.columns=['user_id','department_id','depart_delta_days_last', 'depart_delta_order_number_last','order_number_reverse']
    
    print 'merge delta days feature: %s'%datetime.datetime.now()
    user_item_feat=xprior[['user_id','product_id','aisle_id','department_id']].drop_duplicates()
    user_item_feat=pd.merge(user_item_feat, user_item_delta_last, how='left',on=['user_id','product_id'])
    user_item_feat=pd.merge(user_item_feat, user_aisle_delta_last, how='left',on=['user_id','aisle_id'])
    user_item_feat=pd.merge(user_item_feat, user_department_delta_last, how='left',on=['user_id','department_id'])
    del user_item_delta_last
    del user_aisle_delta_last
    del user_department_delta_last
    gc.collect()

    print 'delta orders feature: %s'%datetime.datetime.now()
    #�û���һ�ι�����Ʒ��ʱ��͵���
    user_item_delta_first=xprior[['user_id','product_id','delta_days_to_predict', 'delta_order_number_to_predict']].groupby(['user_id','product_id'], as_index=False).max()
    user_item_delta_first.columns=['user_id','product_id','item_delta_days_first', 'item_delta_order_number_first']
    #�û���һ�ι���ĳС���ʱ��͵���
    user_aisle_delta_first=xprior[['user_id','aisle_id','delta_days_to_predict', 'delta_order_number_to_predict']].groupby(['user_id','aisle_id'], as_index=False).max()
    user_aisle_delta_first.columns=['user_id','aisle_id','aisle_delta_days_first', 'aisle_delta_order_number_first']
    #�û���һ�ι���ĳ�����ʱ��͵���
    user_department_delta_first=xprior[['user_id','department_id','delta_days_to_predict', 'delta_order_number_to_predict']].groupby(['user_id','department_id'], as_index=False).max()
    user_department_delta_first.columns=['user_id','department_id','depart_delta_days_first', 'depart_delta_order_number_first']
    
    print 'merge delta orders feature: %s'%datetime.datetime.now()
    user_item_feat=pd.merge(user_item_feat, user_item_delta_first, how='left',on=['user_id','product_id'])
    user_item_feat=pd.merge(user_item_feat, user_aisle_delta_first, how='left',on=['user_id','aisle_id'])
    user_item_feat=pd.merge(user_item_feat, user_department_delta_first, how='left',on=['user_id','department_id'])
    del user_item_delta_first
    del user_aisle_delta_first
    del user_department_delta_first
    gc.collect()
    
    #�û���ĳ��Ʒ�״ι�������һ�ι�������������͵���
    user_item_feat['item_delta_days']=user_item_feat['item_delta_days_first']-user_item_feat['item_delta_days_last']
    user_item_feat['aisle_delta_days']=user_item_feat['aisle_delta_days_first']-user_item_feat['aisle_delta_days_last']
    user_item_feat['depart_delta_days']=user_item_feat['depart_delta_days_first']-user_item_feat['depart_delta_days_last']
    user_item_feat['item_delta_order_number']=user_item_feat['item_delta_order_number_first']-user_item_feat['item_delta_order_number_last']
    user_item_feat['aisle_delta_order_number']=user_item_feat['aisle_delta_order_number_first']-user_item_feat['aisle_delta_order_number_last']
    user_item_feat['depart_delta_order_number']=user_item_feat['depart_delta_order_number_first']-user_item_feat['depart_delta_order_number_last']

    user_item_feat['item_aisle_delta_days']=user_item_feat['aisle_delta_days_last']-user_item_feat['item_delta_days_last']
    user_item_feat['aisle_depart_delta_days']=user_item_feat['depart_delta_days_last']-user_item_feat['aisle_delta_days_last']
    user_item_feat['item_aisle_delta_order_number']=user_item_feat['item_delta_order_number_last']-user_item_feat['aisle_delta_order_number_last']
    user_item_feat['aisle_depart_delta_order_number']=user_item_feat['aisle_delta_order_number_last']-user_item_feat['depart_delta_order_number_last']
    
    print 'user item reorded feature: %s'%datetime.datetime.now()
   #�û���ĳ��Ʒ�ظ�����Ĵ������ܹ������
    item_times_in_user=xprior[['user_id','product_id','reordered']].groupby(['user_id','product_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'user_item_reordered_times','count':'user_item_buy_times'}).reset_index()
    #�û��ظ�����Ĵ���ռ��
    item_times_in_user['user_item_reordered_times_frac']=item_times_in_user.user_item_reordered_times/(1.0*item_times_in_user.user_item_buy_times)
    #�û��ظ������С��������͹����С�����ܴ���
    aisle_times_in_user=xprior[['user_id','aisle_id','reordered']].groupby(['user_id','aisle_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'user_aisle_reordered_times','count':'user_aisle_buy_times'}).reset_index()
    #�û��ظ������С������ռ��
    aisle_times_in_user['user_aisle_reordered_times_frac']=aisle_times_in_user.user_aisle_reordered_times/(1.0*aisle_times_in_user.user_aisle_buy_times)
    #�û������Ĵ�������Ͱ������ظ������С�������
    depart_times_in_user=xprior[['user_id','department_id','reordered']].drop_duplicates().groupby(['user_id','department_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'user_depart_reordered_times','count':'user_depart_buy_times'}).reset_index()
    #�û��ظ�����Ĵ������ռ��
    depart_times_in_user['user_depart_reordered_times_frac']=depart_times_in_user.user_depart_reordered_times/(1.0*depart_times_in_user.user_depart_buy_times)
    
    print 'merge user item reorded feature: %s'%datetime.datetime.now()
    user_item_feat=pd.merge(user_item_feat, item_times_in_user, how='left',on=['user_id','product_id'])
    user_item_feat=pd.merge(user_item_feat, aisle_times_in_user, how='left',on=['user_id','aisle_id'])
    user_item_feat=pd.merge(user_item_feat, depart_times_in_user, how='left',on=['user_id','department_id'])
    del item_times_in_user
    del aisle_times_in_user
    del depart_times_in_user
    gc.collect()
    
    #�û�������Ʒ�ڹ����ĳС����Ʒռ�ı���
    user_item_feat['user_item_aisle_sum_frac']=user_item_feat.user_item_reordered_times/((user_item_feat.user_aisle_reordered_times+1)*1.0)
    user_item_feat['user_item_aisle_count_frac']=user_item_feat.user_item_buy_times/(user_item_feat.user_aisle_buy_times*1.0)
    user_item_feat['user_item_aisle_sum_count_frac']=user_item_feat.user_item_reordered_times/(user_item_feat.user_aisle_buy_times*1.0)

    #�û�������Ʒ�ڹ����ĳ������Ʒռ�ı���
    user_item_feat['user_item_depart_count_frac']=user_item_feat.user_item_buy_times/(user_item_feat.user_depart_buy_times*1.0)
    user_item_feat['user_item_depart_sum_frac']=user_item_feat.user_item_reordered_times/((user_item_feat.user_depart_reordered_times+1)*1.0)
    #�û�����С���ڹ����ĳ������Ʒռ�ı���
    user_item_feat['user_aisle_depart_count_frac']=user_item_feat.user_aisle_buy_times/(user_item_feat.user_depart_buy_times*1.0)
    user_item_feat['user_aisle_depart_sum_frac']=user_item_feat.user_aisle_reordered_times/((user_item_feat.user_depart_reordered_times+1)*1.0)
    
    #�û�����ĳ��Ʒ���������Ʒ�״ι�������һ�ι��򶩵���ȵı���
    user_item_feat['user_item_buy_order_frac']=user_item_feat.user_item_buy_times/((user_item_feat.item_delta_order_number+1)*1.0)
    user_item_feat['user_aisle_buy_order_frac']=user_item_feat.user_aisle_buy_times/((user_item_feat.aisle_delta_order_number+1)*1.0)
    user_item_feat['user_depart_buy_order_frac']=user_item_feat.user_depart_buy_times/((user_item_feat.depart_delta_order_number+1)*1.0)
    #�û�ÿ�������������Ʒ
    user_item_feat['user_item_orders_buy_avg']=(user_item_feat.item_delta_order_number+1)/(user_item_feat.user_item_buy_times*1.0)
    user_item_feat['user_aisle_orders_buy_avg']=(user_item_feat.aisle_delta_order_number+1)/(user_item_feat.user_aisle_buy_times*1.0)
    user_item_feat['user_depart_orders_buy_avg']=(user_item_feat.depart_delta_order_number+1)/(user_item_feat.user_depart_buy_times*1.0)
                                                                                           
    #�û�ÿ�����칺�����Ʒ                                                                                           
    user_item_feat['user_item_days_buy_avg']=user_item_feat.item_delta_days/(user_item_feat.user_item_buy_times*1.0)
    user_item_feat['user_aisle_days_buy_avg']=user_item_feat.item_delta_days/(user_item_feat.user_aisle_buy_times*1.0)
    user_item_feat['user_depart_days_buy_avg']=user_item_feat.item_delta_days/(user_item_feat.user_depart_buy_times *1.0)

    #�û����һ�ι���Ԥ���������������ȥƽ����������
    user_item_feat['sub_user_item_days_buy_avg']=np.abs(user_item_feat.item_delta_days_last-user_item_feat.user_item_days_buy_avg)
    user_item_feat['sub_user_aisle_days_buy_avg']=np.abs(user_item_feat.aisle_delta_days_last-user_item_feat.user_aisle_days_buy_avg)
    user_item_feat['sub_user_depart_days_buy_avg']=np.abs(user_item_feat.depart_delta_days_last-user_item_feat.user_depart_days_buy_avg)

    #�û����һ�ι���Ԥ�ⶩ�����������ȥƽ��������
    user_item_feat['sub_user_item_order_buy_avg']=np.abs(user_item_feat.item_delta_order_number_last-user_item_feat.user_item_orders_buy_avg)
    user_item_feat['sub_user_aisle_order_buy_avg']=np.abs(user_item_feat.aisle_delta_order_number_last-user_item_feat.user_aisle_orders_buy_avg)
    user_item_feat['sub_user_depart_order_buy_avg']=np.abs(user_item_feat.depart_delta_order_number_last-user_item_feat.user_depart_orders_buy_avg)

    print 'user item_add_to_cart feature: %s'%datetime.datetime.now()  

    #�û�����Ʒ���붩�������
    item_add_to_cart=xprior[['user_id','product_id','add_to_cart_order']].groupby(['user_id','product_id'])['add_to_cart_order'].agg(['mean','std']).rename(columns={'mean':'item_add_mean','std':'item_add_std'}).reset_index()
    item_add_to_cart.item_add_std=item_add_to_cart.item_add_std.fillna(0)
    #�û�����ƷС������붩�������
    aisle_add_to_cart=xprior[['user_id','aisle_id','add_to_cart_order']].groupby(['user_id','aisle_id'])['add_to_cart_order'].agg(['mean','std']).rename(columns={'mean':'aisle_add_mean','std':'aisle_add_std'}).reset_index()
    aisle_add_to_cart.aisle_add_std=aisle_add_to_cart.aisle_add_std.fillna(0)
    #�û�����Ʒ�������붩�������
    depart_add_to_cart=xprior[['user_id','department_id','add_to_cart_order']].groupby(['user_id','department_id'])['add_to_cart_order'].agg(['mean','std']).rename(columns={'mean':'depart_add_mean','std':'depart_add_std'}).reset_index()
    depart_add_to_cart.depart_add_std=depart_add_to_cart.depart_add_std.fillna(0)
    
    print 'merge item_add_to_cart feature: %s'%datetime.datetime.now()  
    user_item_feat=pd.merge(user_item_feat, item_add_to_cart, how='left',on=['user_id','product_id'])
    user_item_feat=pd.merge(user_item_feat, aisle_add_to_cart, how='left',on=['user_id','aisle_id'])
    user_item_feat=pd.merge(user_item_feat, depart_add_to_cart, how='left',on=['user_id','department_id'])
    
    del item_add_to_cart
    del aisle_add_to_cart
    del depart_add_to_cart
    gc.collect()
    
    print 'user item_sum_dow_to_cart feature: %s'%datetime.datetime.now()  
    #�û�һ���ڸ��칺��ĳ��Ʒ���ܴ������ظ��������
    item_sum_dow_to_cart=xprior[['user_id','product_id','order_dow','reordered']].groupby(['user_id','product_id','order_dow'])['reordered'].agg(['sum','count']).rename(columns={'sum':'item_dow_reordered','count':'item_dow_buy'}).reset_index()
    item_sum_dow_to_cart['item_dow_reordered_frac']=item_sum_dow_to_cart.item_dow_reordered/(item_sum_dow_to_cart.item_dow_buy*1.0)
    #�û�һ���ڸ��칺��ĳС����ܴ������ظ��������
    aisle_sum_dow_to_cart=xprior[['user_id','aisle_id','order_dow','reordered']].groupby(['user_id','aisle_id','order_dow'])['reordered'].agg(['sum','count']).rename(columns={'sum':'aisle_dow_reordered','count':'aisle_dow_buy'}).reset_index()
    aisle_sum_dow_to_cart['item_dow_reordered_frac']=aisle_sum_dow_to_cart.aisle_dow_reordered/(aisle_sum_dow_to_cart.aisle_dow_buy*1.0)

    #�û�һ���ڸ��칺��ĳ������ܴ������ظ��������
    depart_sum_dow_to_cart=xprior[['user_id','department_id','order_dow','reordered']].groupby(['user_id','department_id','order_dow'])['reordered'].agg(['sum','count']).rename(columns={'sum':'depart_dow_reordered','count':'depart_dow_buy'}).reset_index()
    depart_sum_dow_to_cart['item_dow_reordered_frac']=depart_sum_dow_to_cart.depart_dow_reordered/(depart_sum_dow_to_cart.depart_dow_buy*1.0)

    item_dow_feat=pd.merge(xprior[['user_id','product_id','order_dow_predict']].drop_duplicates(), item_sum_dow_to_cart, how='left', left_on=['user_id','product_id','order_dow_predict'],right_on=['user_id','product_id','order_dow'])
    item_dow_feat.fillna(0, inplace=True)
    aisle_dow_feat=pd.merge(xprior[['user_id','aisle_id','order_dow_predict']].drop_duplicates(), aisle_sum_dow_to_cart, how='left', left_on=['user_id','aisle_id','order_dow_predict'],right_on=['user_id','aisle_id','order_dow'])
    aisle_dow_feat.fillna(0, inplace=True)
    depart_dow_feat=pd.merge(xprior[['user_id','department_id','order_dow_predict']].drop_duplicates(), depart_sum_dow_to_cart, how='left', left_on=['user_id','department_id','order_dow_predict'],right_on=['user_id','department_id','order_dow'])
    depart_dow_feat.fillna(0, inplace=True)
    
    item_dow_feat.drop(['order_dow','order_dow_predict'], axis=1, inplace=True)
    aisle_dow_feat.drop(['order_dow','order_dow_predict'], axis=1, inplace=True)
    depart_dow_feat.drop(['order_dow','order_dow_predict'], axis=1, inplace=True)

    print 'merge user item_sum_dow_to_cart feature: %s'%datetime.datetime.now()  

    user_item_feat=pd.merge(user_item_feat, item_dow_feat, how='left',on=['user_id','product_id'])
    user_item_feat=pd.merge(user_item_feat, aisle_dow_feat, how='left',on=['user_id','aisle_id'])
    user_item_feat=pd.merge(user_item_feat, depart_dow_feat, how='left',on=['user_id','department_id'])
     
    del item_dow_feat
    del aisle_dow_feat
    del depart_dow_feat
    gc.collect()
    
    xprior['hour_seg']=xprior.order_hour_of_day.apply(lambda hour:convert_hour(hour))
    xprior['hour_seg_predict']=xprior.order_hour_of_day_predict.apply(lambda hour:convert_hour(hour))
    xprior['delta_hour_seg']=xprior.hour_seg-xprior.hour_seg_predict

    print 'item_sum_hour_to_cart feature: %s'%datetime.datetime.now()  

    #�û�һ�����ʱ��ι���ĳ��Ʒ���ܴ���
    item_sum_hour_to_cart=xprior[['user_id','product_id','hour_seg','reordered']].groupby(['user_id','product_id','hour_seg'])['reordered'].agg(['sum']).rename(columns={'sum':'item_order_hour_seg_sum'}).reset_index()
    #�û�һ�����ʱ��ι���ĳС����ܴ���
    aisle_sum_hour_to_cart=xprior[['user_id','aisle_id','hour_seg','reordered']].groupby(['user_id','aisle_id','hour_seg'])['reordered'].agg(['sum']).rename(columns={'sum':'aisle_hour_seg_sum'}).reset_index()
    #�û�һ�����ʱ��ι���ĳ������ܴ���
    depart_sum_hour_to_cart=xprior[['user_id','department_id','hour_seg','reordered']].groupby(['user_id','department_id','hour_seg'])['reordered'].agg(['sum']).rename(columns={'sum':'depart_hour_seg_sum'}).reset_index()

    item_hour_seg_feat=pd.merge(xprior[['user_id','product_id','hour_seg_predict']].drop_duplicates(), item_sum_hour_to_cart, how='left', left_on=['user_id','product_id','hour_seg_predict'],right_on=['user_id','product_id','hour_seg'])
    item_hour_seg_feat.fillna(0, inplace=True)
    aisle_hour_seg_feat=pd.merge(xprior[['user_id','aisle_id','hour_seg_predict']].drop_duplicates(), aisle_sum_hour_to_cart, how='left', left_on=['user_id','aisle_id','hour_seg_predict'],right_on=['user_id','aisle_id','hour_seg'])
    aisle_hour_seg_feat.fillna(0, inplace=True)
    depart_hour_seg_feat=pd.merge(xprior[['user_id','department_id','hour_seg_predict']].drop_duplicates(), depart_sum_hour_to_cart, how='left', left_on=['user_id','department_id','hour_seg_predict'],right_on=['user_id','department_id','hour_seg'])
    depart_hour_seg_feat.fillna(0, inplace=True)

    item_hour_seg_feat.drop(['hour_seg','hour_seg_predict'], axis=1, inplace=True)
    aisle_hour_seg_feat.drop(['hour_seg','hour_seg_predict'], axis=1, inplace=True)
    depart_hour_seg_feat.drop(['hour_seg','hour_seg_predict'], axis=1, inplace=True)
    
    print 'merge item_hour_seg_feat feature: %s'%datetime.datetime.now()  
    user_item_feat=pd.merge(user_item_feat, item_hour_seg_feat, how='left',on=['user_id','product_id'])
    user_item_feat=pd.merge(user_item_feat, aisle_hour_seg_feat, how='left',on=['user_id','aisle_id'])
    user_item_feat=pd.merge(user_item_feat, depart_hour_seg_feat, how='left',on=['user_id','department_id'])
    
    del item_sum_hour_to_cart
    del aisle_sum_hour_to_cart
    del depart_sum_hour_to_cart
    del item_hour_seg_feat
    del aisle_hour_seg_feat
    del depart_hour_seg_feat
    gc.collect()
    
    #user features
    print 'total_order_number feature: %s'%datetime.datetime.now()  

    #ÿ���û����ܶ�����
    user_total_order_number=xprior[['user_id','order_id','reordered']].drop_duplicates()[['user_id','reordered']].groupby(['user_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'user_total_reordered_orders','count':'user_total_orders'}).reset_index()
    user_item_feat=pd.merge(user_item_feat, user_total_order_number, how='left',on=['user_id'])
    user_item_feat['user_reordered_orders_frac']=user_item_feat.user_total_reordered_orders/(user_item_feat.user_total_orders*1.0)
    del user_total_order_number
    gc.collect()
    
    #�û�ƽ��ÿ���¶������������
    item_add_to_cart=xprior[['user_id','order_number','days_since_prior_order']].drop_duplicates()[['user_id','days_since_prior_order']].groupby(['user_id'])['days_since_prior_order'].agg(['mean','std']).rename(columns={'mean':'user_order_days_mean','std':'user_order_days_std'}).reset_index()
    user_item_feat=pd.merge(user_item_feat, item_add_to_cart, how='left',on=['user_id'])
    del item_add_to_cart
    gc.collect()
    
    #�û��ظ�������Ʒ�Ĵ����͹������
    total_item_count_in_user=xprior[['user_id','product_id','reordered']].groupby(['user_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'total_item_sum','count':'total_item_count'}).reset_index()
    #�û��ظ�����Ĵ���ռ��
    total_item_count_in_user['total_user_item_reordered_frac']=total_item_count_in_user.total_item_sum/(1.0*total_item_count_in_user.total_item_count)
    #�û������С�������Ͱ������ظ������С�������
    total_aisle_count_in_user=xprior[['user_id','aisle_id','reordered']].groupby(['user_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'total_aisle_sum','count':'total_aisle_count'}).reset_index()
    #�û��ظ������С������ռ��
    total_aisle_count_in_user['total_user_aisle_reordered_frac']=total_aisle_count_in_user.total_aisle_sum/(1.0*total_aisle_count_in_user.total_aisle_count)
    #�û������Ĵ�������Ͱ������ظ������С�������
    total_depart_count_in_user=xprior[['user_id','department_id','reordered']].groupby(['user_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'total_depart_sum','count':'total_depart_count'}).reset_index()
    #�û��ظ�����Ĵ������ռ��
    total_depart_count_in_user['total_user_depart_reordered_frac']=total_depart_count_in_user.total_depart_sum/(1.0*total_depart_count_in_user.total_depart_count)
    
    print 'merge total_order_number feature: %s'%datetime.datetime.now()  
    user_item_feat=pd.merge(user_item_feat, total_item_count_in_user, how='left',on=['user_id'])
    user_item_feat=pd.merge(user_item_feat, total_aisle_count_in_user, how='left',on=['user_id'])
    user_item_feat=pd.merge(user_item_feat, total_depart_count_in_user, how='left',on=['user_id'])
    del total_item_count_in_user
    del total_aisle_count_in_user
    del total_depart_count_in_user
    gc.collect()
    
    #�û�������Ʒ�ڹ����ĳС����Ʒռ�ı���
    user_item_feat['total_item_aisle_sum_frac']=user_item_feat.total_item_sum/(user_item_feat.total_aisle_sum*1.0)
    user_item_feat['total_item_aisle_count_frac']=user_item_feat.total_item_count/(user_item_feat.total_aisle_count*1.0)
    #�û�������Ʒ�ڹ����ĳ������Ʒռ�ı���
    user_item_feat['total_item_depart_count_frac']=user_item_feat.total_item_count/(user_item_feat.total_depart_count*1.0)
    user_item_feat['total_item_depart_sum_frac']=user_item_feat.total_item_sum/(user_item_feat.total_depart_sum*1.0)
    #�û�����С���ڹ����ĳ������Ʒռ�ı���
    user_item_feat['total_aisle_depart_count_frac']=user_item_feat.total_aisle_count/(user_item_feat.total_depart_count*1.0)
    user_item_feat['total_aisle_depart_sum_frac']=user_item_feat.total_aisle_sum/(user_item_feat.total_depart_sum*1.0)
    
    user_item_feat['total_user_item_numbers_per_order']=user_item_feat.total_item_count/(user_item_feat.user_total_orders*1.0)
    user_item_feat['total_user_item_reordered_numbers_per_order']=user_item_feat.total_item_sum/(user_item_feat.user_total_orders*1.0)
    user_item_feat['total_user_aisle_numbers_per_order']=user_item_feat.total_aisle_count/(user_item_feat.user_total_orders*1.0)
    user_item_feat['total_user_aisle_reordered_numbers_per_order']=user_item_feat.total_aisle_sum/(user_item_feat.user_total_orders*1.0)
    user_item_feat['total_user_depart_numbers_per_order']=user_item_feat.total_depart_sum/(user_item_feat.user_total_orders*1.0)
    user_item_feat['total_user_depart_reordered_numbers_per_order']=user_item_feat.total_depart_count/(user_item_feat.user_total_orders*1.0)


    print 'item_count_in_user feature: %s'%datetime.datetime.now()  
    #�û��ظ��������Ʒ���͹��������Ʒ��
    item_count_in_user=xprior[['user_id','product_id','reordered']].drop_duplicates().groupby(['user_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'item_sum','count':'item_count'}).reset_index()
    #�û��ظ�����Ĵ���ռ��
    item_count_in_user['item_reordered_frac']=item_count_in_user.item_sum/(1.0*item_count_in_user.item_count)
    #�û������С�������Ͱ������ظ������С�������
    aisle_count_in_user=xprior[['user_id','aisle_id','reordered']].drop_duplicates().groupby(['user_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'aisle_sum','count':'aisle_count'}).reset_index()
    #�û��ظ������С������ռ��
    aisle_count_in_user['aisle_reordered_frac']=aisle_count_in_user.aisle_sum/(1.0*aisle_count_in_user.aisle_count)
    #�û������Ĵ�������Ͱ������ظ������С�������
    depart_count_in_user=xprior[['user_id','department_id','reordered']].drop_duplicates().groupby(['user_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'depart_sum','count':'depart_count'}).reset_index()
    #�û��ظ�����Ĵ������ռ��
    depart_count_in_user['depart_reordered_frac']=depart_count_in_user.depart_sum/(1.0*depart_count_in_user.depart_count)
    
    print 'merge item_count_in_user feature: %s'%datetime.datetime.now()  
    user_item_feat=pd.merge(user_item_feat, item_count_in_user, how='left',on=['user_id'])
    user_item_feat=pd.merge(user_item_feat, aisle_count_in_user, how='left',on=['user_id'])
    user_item_feat=pd.merge(user_item_feat, depart_count_in_user, how='left',on=['user_id'])
    
    del item_count_in_user
    del aisle_count_in_user
    del depart_count_in_user
    gc.collect()
    
    #�û�������Ʒ�ڹ����ĳС����Ʒռ�ı���
    user_item_feat['item_aisle_sum_frac']=user_item_feat.item_sum/(user_item_feat.aisle_sum*1.0)
    user_item_feat['item_aisle_count_frac']=user_item_feat.item_count/(user_item_feat.aisle_count*1.0)
    #�û�������Ʒ�ڹ����ĳ������Ʒռ�ı���
    user_item_feat['item_depart_count_frac']=user_item_feat.item_count/(user_item_feat.depart_count*1.0)
    user_item_feat['item_depart_sum_frac']=user_item_feat.item_sum/(user_item_feat.depart_sum*1.0)
    #�û�����С���ڹ����ĳ������Ʒռ�ı���
    user_item_feat['aisle_depart_count_frac']=user_item_feat.aisle_count/(user_item_feat.depart_count*1.0)
    user_item_feat['aisle_depart_sum_frac']=user_item_feat.aisle_sum/(user_item_feat.depart_sum*1.0)

    user_item_feat['user_item_numbers_per_order']=user_item_feat.item_count/(user_item_feat.user_total_orders*1.0)
    user_item_feat['user_item_reordered_numbers_per_order']=user_item_feat.aisle_sum/(user_item_feat.user_total_orders*1.0)
    user_item_feat['user_aisle_numbers_per_order']=user_item_feat.aisle_count/(user_item_feat.user_total_orders*1.0)
    user_item_feat['user_aisle_reordered_numbers_per_order']=user_item_feat.aisle_sum/(user_item_feat.user_total_orders*1.0)
    user_item_feat['user_depart_numbers_per_order']=user_item_feat.depart_count/(user_item_feat.user_total_orders*1.0)
    user_item_feat['user_depart_reordered_numbers_per_order']=user_item_feat.depart_sum/(user_item_feat.user_total_orders*1.0)
    
    #item features
    #��Ʒ�ظ�����Ĵ������ܹ������
    print 'total_item_buy_times feature: %s'%datetime.datetime.now()  
    total_item_buy_times=xprior[['product_id','reordered']].groupby(['product_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'total_item_reorder_times','count':'total_item_buy_times'}).reset_index()
    #��Ʒ�ظ�����Ĵ���ռ��
    total_item_buy_times['total_item_reordered_frac']=total_item_buy_times.total_item_reorder_times/(1.0*total_item_buy_times.total_item_buy_times)
    #С�����ظ�����������ܹ������
    total_aisle_buy_times=xprior[['aisle_id','reordered']].groupby(['aisle_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'total_aisle_reorder_times','count':'total_aisle_buy_times'}).reset_index()
    #С�����ظ�����Ĵ���ռ��
    total_aisle_buy_times['total_aisle_reordered_frac']=total_aisle_buy_times.total_aisle_reorder_times/(1.0*total_aisle_buy_times.total_aisle_buy_times)
    #��������ظ�����������ܹ������
    total_depart_buy_times=xprior[['department_id','reordered']].groupby(['department_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'total_depart_reorder_times','count':'total_depart_buy_times'}).reset_index()
    #�ظ������������ռ��
    total_depart_buy_times['total_depart_reordered_frac']=total_depart_buy_times.total_depart_reorder_times/(1.0*total_depart_buy_times.total_depart_buy_times)
    
    print 'merge total_item_buy_times feature: %s'%datetime.datetime.now()  
    user_item_feat=pd.merge(user_item_feat, total_item_buy_times, how='left',on=['product_id'])
    user_item_feat=pd.merge(user_item_feat, total_aisle_buy_times, how='left',on=['aisle_id'])
    user_item_feat=pd.merge(user_item_feat, total_depart_buy_times, how='left',on=['department_id'])
    user_item_feat['ia_frac']=user_item_feat.total_item_reorder_times/(user_item_feat.total_aisle_reorder_times*1.0)
    user_item_feat['ad_frac']=user_item_feat.total_aisle_reorder_times/(user_item_feat.total_depart_reorder_times*1.0)
    user_item_feat['id_frac']=user_item_feat.total_item_reorder_times/(user_item_feat.total_depart_reorder_times*1.0)
    
    del total_item_buy_times
    del total_aisle_buy_times
    del total_depart_buy_times
    gc.collect()
    
    print 'item_buy_users feature: %s'%datetime.datetime.now()  
    #�������Ʒ���û������ظ��������Ʒ���û���
    item_buy_users=xprior[['product_id','user_id','reordered']].drop_duplicates().groupby(['product_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'item_reordered_users','count':'item_buy_users'}).reset_index()    
    item_buy_users['item_buy_users_reordered_frac']=item_buy_users.item_reordered_users/(item_buy_users.item_buy_users*1.0)
    aisle_buy_users=xprior[['aisle_id','user_id','reordered']].drop_duplicates().groupby(['aisle_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'aisle_reordered_users','count':'aisle_buy_users'}).reset_index()    
    aisle_buy_users['aisle_buy_users_reordered_frac']=aisle_buy_users.aisle_reordered_users/(aisle_buy_users.aisle_buy_users*1.0)
    depart_buy_users=xprior[['department_id','user_id','reordered']].drop_duplicates().groupby(['department_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'depart_reordered_users','count':'depart_buy_users'}).reset_index()    
    depart_buy_users['depart_buy_users_reordered_frac']=depart_buy_users.depart_reordered_users/(depart_buy_users.depart_buy_users*1.0)

    print 'merge item_buy_users feature: %s'%datetime.datetime.now()  
    user_item_feat=pd.merge(user_item_feat, item_buy_users, how='left',on=['product_id'])
    user_item_feat=pd.merge(user_item_feat, aisle_buy_users, how='left',on=['aisle_id'])
    user_item_feat=pd.merge(user_item_feat, depart_buy_users, how='left',on=['department_id'])

    del item_buy_users
    del aisle_buy_users
    del depart_buy_users
    gc.collect()
    
    user_item_feat['ia_users_reordered_frac']=user_item_feat.item_reordered_users/(user_item_feat.aisle_reordered_users*1.0)
    user_item_feat['ad_users_reordered_frac']=user_item_feat.aisle_reordered_users/(user_item_feat.depart_reordered_users*1.0)
    user_item_feat['id_users_reordered_frac']=user_item_feat.item_reordered_users/(user_item_feat.depart_reordered_users*1.0)

    user_item_feat.drop(['aisle_id','department_id'], axis=1, inplace=True)
    user_item_feat.fillna(0, inplace=True)

    print 'extract feature ok: %s'%datetime.datetime.now() 
    return user_item_feat
	
#��ҪԤ��Ķ�������������������ʷ�������츺���������Ը������ʵ�����
def get_label(xprior, predict_order, train_detail, sample_rate=-1):
    xpredict_order=pd.merge(predict_order, train_detail, how='left', on='order_id')
    pos_label=xpredict_order[xpredict_order.reordered==1][['user_id', 'product_id', 'reordered']]

    user_item_pairs=xprior[['user_id', 'product_id']].drop_duplicates()
    all_label=pd.merge(user_item_pairs, pos_label, how='left', on=['user_id', 'product_id'])
    all_label.fillna(0, inplace=True)
    
    if sample_rate==-1:#������
        label=all_label
    else:#�Ӹ������в���
        sample_n=pos_label.shape[0]*sample_rate
        if (sample_n+pos_label.shape[0])<all_label.shape[0]:
            all_negative_label=all_label[all_label.reordered==0]
            negative_label=all_negative_label.sample(sample_n)
            label=pd.concat([pos_label, negative_label])
        else:
            label=all_label
    label = shuffle(label)

    return label

#�Ѷ���ת��Ϊ�û�-�ظ������Ʒ�б��DataFrame��ʽ����������
def convert_order_to_reordered_result(users, orders):
    uid=pd.DataFrame(users)
    uid.columns=['user_id']
    user_reordered_products=orders[orders.reordered==1][['user_id','product_id']].groupby('user_id').agg(lambda x: ' '.join(x.astype(str))).reset_index()
    reordered_result=pd.merge(uid, user_reordered_products, how='left', on='user_id')
    reordered_result.fillna('None', inplace=True)
    return reordered_result

#�û�-�ظ������Ʒ�б��DataFrame��ʽת��Ϊpython�ֵ䣬����������ƽ��F1
def reorder_df_to_dict(result):
    result=result.set_index('user_id')
    result_dict=dict()
    for uid in result.index.values:
        result_dict[uid]=set(result.loc[uid,'product_id'].split(' '))
    return result_dict
	
#ȡ���������Ϻ�label
def get_x_y(xorders, prior_detail, products, train, sample_rate=-1):
    prior_orders,predict_order = split_orders(xorders, 0)
    xprior=get_xprior(prior_orders, predict_order, prior_detail, products)	
    print 'get full features'
    full_features=extract_features(xprior)     
    '''
    print 'get delta_3 features'
    prior_orders_delta_3,predict_order = split_orders(xorders, 0, 3)
    xprior_delta_3=get_xprior(prior_orders_delta_3, predict_order, prior_detail, products)
    features_delta_3=extract_features(xprior_delta_3)
    print 'merge features'
    all_features=pd.merge(full_features,features_delta_3,how='left',on=['user_id', 'product_id'], suffixes=('_all', '_3'))                     
    '''
    all_features=full_features   
    
    train_users=predict_order[predict_order.eval_set=='train'].user_id.unique().tolist()
    test_users=predict_order[predict_order.eval_set=='test'].user_id.unique().tolist()
    
    train_features=all_features[all_features.user_id.isin(train_users)]
    test_features=all_features[all_features.user_id.isin(test_users)]
    
    train_predict_order=predict_order[predict_order.eval_set=='train']
    train_xprior=xprior[xprior.user_id.isin(train_users)]
    
    train_label=get_label(train_xprior, train_predict_order, train, sample_rate)
    
    train_data=pd.merge(train_label, train_features, how='left',on=['user_id', 'product_id'])
    train_y=train_data[['user_id','product_id','reordered']]
    train_x=train_data.drop(['user_id','product_id','reordered'], axis=1)
    
    test_ui=test_features[['user_id','product_id']]
    test_x=test_features.drop(['user_id','product_id'], axis=1)
    
    return (train_x, train_y, test_x, test_ui)

#��ȡѵ����ʵ���ظ�������
def get_train_reordered_result(orders, train):
    train_reordered=pd.merge(orders[orders.eval_set=='train'][['order_id','user_id']],train[['order_id','product_id','reordered']],how='left', on='order_id')
    train_reordered_result=convert_order_to_reordered_result(train_reordered.user_id.unique().tolist(), train_reordered)
    return train_reordered_result

    
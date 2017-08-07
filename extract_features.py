# -- coding: utf-8 --
import pandas as pd
import numpy as np
import datetime
import gc
from sklearn.utils import shuffle 

#把订单按照从最后一个订单到第一个订单逆序编号，序号从0开始
def orders_prepare(orders):
    max_order=orders.groupby('user_id', as_index=False)['order_number'].max()
    max_order.columns=['user_id','max_order_number']
    xorders=pd.merge(orders, max_order, how='left',on='user_id')
    #现在最后一个订单序号都为0，计算每个订单和最后一单相隔的单号
    xorders['order_number_reverse']=xorders.max_order_number-xorders.order_number
    return xorders

#按照最近的单号，把订单分为要预测的订单，和该订单之前的所有历史订单
def split_orders(xorders, split_point, delta_order=-1):
    if delta_order<=0:
        prior_orders=xorders[xorders.order_number_reverse>split_point].copy()
        predict_order=xorders[xorders.order_number_reverse==split_point].copy()
    else:
        prior_orders=xorders[(xorders.order_number_reverse>split_point) & (xorders.order_number_reverse<=(split_point+delta_order))].copy()
        predict_order=xorders[xorders.order_number_reverse==split_point].copy()        
    return (prior_orders, predict_order)

#把历史订单和要预测订单的时间信息结合起来
def get_xprior(prior_orders, predict_order, detail_prior, products):
    prior_orders['days_since_prior_order']=prior_orders.days_since_prior_order.fillna(0)
    #用户从首单开始，按订单累加，用户每一单与首单相隔的天数
    prior_orders['days_since_first_order']=prior_orders.groupby('user_id', as_index=False)['days_since_prior_order'].cumsum()
    #用户最后一单与首单相隔的天数
    latest_order=prior_orders.groupby('user_id', as_index=False)[['order_number','days_since_first_order']].max()
    latest_order.columns=['user_id','latest_order_number','latest_order_days_since_first_order']
    xprior_orders=pd.merge(prior_orders, latest_order, how='left',on='user_id')
    xprior_orders=pd.merge(xprior_orders,predict_order[['user_id','order_number','order_dow','order_hour_of_day','days_since_prior_order']], how='left',on='user_id', suffixes=('','_predict'))
    #用户每一个订单与要预测的订单相隔的天数
    xprior_orders['delta_days_to_predict']=xprior_orders.latest_order_days_since_first_order-xprior_orders.days_since_first_order+xprior_orders.days_since_prior_order_predict
    #用户每一个订单与要预测的订单相隔的单数
    xprior_orders['delta_order_number_to_predict']=xprior_orders.order_number_predict-xprior_orders.order_number
    xprior_orders['delta_order_dow_to_predict']=np.abs(xprior_orders.order_dow-xprior_orders.order_dow_predict)
    xprior_orders['delta_order_hour_to_predict']=np.abs(xprior_orders.order_hour_of_day-xprior_orders.order_hour_of_day_predict)
    
    xprior=pd.merge(xprior_orders, detail_prior, how='left', on='order_id')
    xprior=pd.merge(xprior, products,how='left', on='product_id')
    return xprior

#转换小时,使时间粒度变粗
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
 
#从历史订单提取特征    
def extract_features(xprior):
    print 'extract feature begin: %s'%datetime.datetime.now()  
    print 'delta days feature: %s'%datetime.datetime.now()
    #user-item features
    #用户最后一次购买某物品的时间和单号, 为了后门提取特征方便，该时间和单号是按照最后一单倒序来的。比如 最后一单是1，倒数第一单就是2
    user_item_delta_last=xprior[['user_id','product_id','delta_days_to_predict', 'delta_order_number_to_predict', 'order_number_reverse']].groupby(['user_id','product_id'],as_index=False).min()
    user_item_delta_last.columns=['user_id','product_id','item_delta_days_last', 'item_delta_order_number_last', 'order_number_reverse']
    #用户最后一次购买某小类的时间和单号
    user_aisle_delta_last=xprior[['user_id','aisle_id','delta_days_to_predict', 'delta_order_number_to_predict','order_number_reverse']].groupby(['user_id','aisle_id'], as_index=False).min()
    user_aisle_delta_last.columns=['user_id','aisle_id','aisle_delta_days_last', 'aisle_delta_order_number_last','order_number_reverse']
    #用户最后一次购买某大类的时间和单号
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
    #用户第一次购买物品的时间和单号
    user_item_delta_first=xprior[['user_id','product_id','delta_days_to_predict', 'delta_order_number_to_predict']].groupby(['user_id','product_id'], as_index=False).max()
    user_item_delta_first.columns=['user_id','product_id','item_delta_days_first', 'item_delta_order_number_first']
    #用户第一次购买某小类的时间和单号
    user_aisle_delta_first=xprior[['user_id','aisle_id','delta_days_to_predict', 'delta_order_number_to_predict']].groupby(['user_id','aisle_id'], as_index=False).max()
    user_aisle_delta_first.columns=['user_id','aisle_id','aisle_delta_days_first', 'aisle_delta_order_number_first']
    #用户第一次购买某大类的时间和单号
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
    
    #用户对某物品首次购买和最后一次购买相隔的天数和单数
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
   #用户对某物品重复购买的次数和总购买次数
    item_times_in_user=xprior[['user_id','product_id','reordered']].groupby(['user_id','product_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'user_item_reordered_times','count':'user_item_buy_times'}).reset_index()
    #用户重复购买的次数占比
    item_times_in_user['user_item_reordered_times_frac']=item_times_in_user.user_item_reordered_times/(1.0*item_times_in_user.user_item_buy_times)
    #用户重复购买的小分类次数和购买的小分类总次数
    aisle_times_in_user=xprior[['user_id','aisle_id','reordered']].groupby(['user_id','aisle_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'user_aisle_reordered_times','count':'user_aisle_buy_times'}).reset_index()
    #用户重复购买的小分类数占比
    aisle_times_in_user['user_aisle_reordered_times_frac']=aisle_times_in_user.user_aisle_reordered_times/(1.0*aisle_times_in_user.user_aisle_buy_times)
    #用户订购的大分类数和包含的重复购买的小大分类数
    depart_times_in_user=xprior[['user_id','department_id','reordered']].drop_duplicates().groupby(['user_id','department_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'user_depart_reordered_times','count':'user_depart_buy_times'}).reset_index()
    #用户重复购买的大分类数占比
    depart_times_in_user['user_depart_reordered_times_frac']=depart_times_in_user.user_depart_reordered_times/(1.0*depart_times_in_user.user_depart_buy_times)
    
    print 'merge user item reorded feature: %s'%datetime.datetime.now()
    user_item_feat=pd.merge(user_item_feat, item_times_in_user, how='left',on=['user_id','product_id'])
    user_item_feat=pd.merge(user_item_feat, aisle_times_in_user, how='left',on=['user_id','aisle_id'])
    user_item_feat=pd.merge(user_item_feat, depart_times_in_user, how='left',on=['user_id','department_id'])
    del item_times_in_user
    del aisle_times_in_user
    del depart_times_in_user
    gc.collect()
    
    #用户购买商品在购买的某小类商品占的比例
    user_item_feat['user_item_aisle_sum_frac']=user_item_feat.user_item_reordered_times/((user_item_feat.user_aisle_reordered_times+1)*1.0)
    user_item_feat['user_item_aisle_count_frac']=user_item_feat.user_item_buy_times/(user_item_feat.user_aisle_buy_times*1.0)
    user_item_feat['user_item_aisle_sum_count_frac']=user_item_feat.user_item_reordered_times/(user_item_feat.user_aisle_buy_times*1.0)

    #用户购买商品在购买的某大类商品占的比例
    user_item_feat['user_item_depart_count_frac']=user_item_feat.user_item_buy_times/(user_item_feat.user_depart_buy_times*1.0)
    user_item_feat['user_item_depart_sum_frac']=user_item_feat.user_item_reordered_times/((user_item_feat.user_depart_reordered_times+1)*1.0)
    #用户购买小类在购买的某大类商品占的比例
    user_item_feat['user_aisle_depart_count_frac']=user_item_feat.user_aisle_buy_times/(user_item_feat.user_depart_buy_times*1.0)
    user_item_feat['user_aisle_depart_sum_frac']=user_item_feat.user_aisle_reordered_times/((user_item_feat.user_depart_reordered_times+1)*1.0)
    
    #用户购买某物品次数与该物品首次购买和最后一次购买订单跨度的比例
    user_item_feat['user_item_buy_order_frac']=user_item_feat.user_item_buy_times/((user_item_feat.item_delta_order_number+1)*1.0)
    user_item_feat['user_aisle_buy_order_frac']=user_item_feat.user_aisle_buy_times/((user_item_feat.aisle_delta_order_number+1)*1.0)
    user_item_feat['user_depart_buy_order_frac']=user_item_feat.user_depart_buy_times/((user_item_feat.depart_delta_order_number+1)*1.0)
    #用户每隔几单购买该物品
    user_item_feat['user_item_orders_buy_avg']=(user_item_feat.item_delta_order_number+1)/(user_item_feat.user_item_buy_times*1.0)
    user_item_feat['user_aisle_orders_buy_avg']=(user_item_feat.aisle_delta_order_number+1)/(user_item_feat.user_aisle_buy_times*1.0)
    user_item_feat['user_depart_orders_buy_avg']=(user_item_feat.depart_delta_order_number+1)/(user_item_feat.user_depart_buy_times*1.0)
                                                                                           
    #用户每隔几天购买该物品                                                                                           
    user_item_feat['user_item_days_buy_avg']=user_item_feat.item_delta_days/(user_item_feat.user_item_buy_times*1.0)
    user_item_feat['user_aisle_days_buy_avg']=user_item_feat.item_delta_days/(user_item_feat.user_aisle_buy_times*1.0)
    user_item_feat['user_depart_days_buy_avg']=user_item_feat.item_delta_days/(user_item_feat.user_depart_buy_times *1.0)

    #用户最后一次购买到预测日期相隔天数减去平均购买天数
    user_item_feat['sub_user_item_days_buy_avg']=np.abs(user_item_feat.item_delta_days_last-user_item_feat.user_item_days_buy_avg)
    user_item_feat['sub_user_aisle_days_buy_avg']=np.abs(user_item_feat.aisle_delta_days_last-user_item_feat.user_aisle_days_buy_avg)
    user_item_feat['sub_user_depart_days_buy_avg']=np.abs(user_item_feat.depart_delta_days_last-user_item_feat.user_depart_days_buy_avg)

    #用户最后一次购买到预测订单相隔单数减去平均购买单数
    user_item_feat['sub_user_item_order_buy_avg']=np.abs(user_item_feat.item_delta_order_number_last-user_item_feat.user_item_orders_buy_avg)
    user_item_feat['sub_user_aisle_order_buy_avg']=np.abs(user_item_feat.aisle_delta_order_number_last-user_item_feat.user_aisle_orders_buy_avg)
    user_item_feat['sub_user_depart_order_buy_avg']=np.abs(user_item_feat.depart_delta_order_number_last-user_item_feat.user_depart_orders_buy_avg)

    print 'user item_add_to_cart feature: %s'%datetime.datetime.now()  

    #用户把商品加入订单的序号
    item_add_to_cart=xprior[['user_id','product_id','add_to_cart_order']].groupby(['user_id','product_id'])['add_to_cart_order'].agg(['mean','std']).rename(columns={'mean':'item_add_mean','std':'item_add_std'}).reset_index()
    item_add_to_cart.item_add_std=item_add_to_cart.item_add_std.fillna(0)
    #用户把商品小分类加入订单的序号
    aisle_add_to_cart=xprior[['user_id','aisle_id','add_to_cart_order']].groupby(['user_id','aisle_id'])['add_to_cart_order'].agg(['mean','std']).rename(columns={'mean':'aisle_add_mean','std':'aisle_add_std'}).reset_index()
    aisle_add_to_cart.aisle_add_std=aisle_add_to_cart.aisle_add_std.fillna(0)
    #用户把商品大分类加入订单的序号
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
    #用户一星期各天购买某商品的总次数和重复购买次数
    item_sum_dow_to_cart=xprior[['user_id','product_id','order_dow','reordered']].groupby(['user_id','product_id','order_dow'])['reordered'].agg(['sum','count']).rename(columns={'sum':'item_dow_reordered','count':'item_dow_buy'}).reset_index()
    item_sum_dow_to_cart['item_dow_reordered_frac']=item_sum_dow_to_cart.item_dow_reordered/(item_sum_dow_to_cart.item_dow_buy*1.0)
    #用户一星期各天购买某小类的总次数和重复购买次数
    aisle_sum_dow_to_cart=xprior[['user_id','aisle_id','order_dow','reordered']].groupby(['user_id','aisle_id','order_dow'])['reordered'].agg(['sum','count']).rename(columns={'sum':'aisle_dow_reordered','count':'aisle_dow_buy'}).reset_index()
    aisle_sum_dow_to_cart['item_dow_reordered_frac']=aisle_sum_dow_to_cart.aisle_dow_reordered/(aisle_sum_dow_to_cart.aisle_dow_buy*1.0)

    #用户一星期各天购买某大类的总次数和重复购买次数
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

    #用户一天各个时间段购买某商品的总次数
    item_sum_hour_to_cart=xprior[['user_id','product_id','hour_seg','reordered']].groupby(['user_id','product_id','hour_seg'])['reordered'].agg(['sum']).rename(columns={'sum':'item_order_hour_seg_sum'}).reset_index()
    #用户一天各个时间段购买某小类的总次数
    aisle_sum_hour_to_cart=xprior[['user_id','aisle_id','hour_seg','reordered']].groupby(['user_id','aisle_id','hour_seg'])['reordered'].agg(['sum']).rename(columns={'sum':'aisle_hour_seg_sum'}).reset_index()
    #用户一天各个时间段购买某大类的总次数
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

    #每个用户的总订单数
    user_total_order_number=xprior[['user_id','order_id','reordered']].drop_duplicates()[['user_id','reordered']].groupby(['user_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'user_total_reordered_orders','count':'user_total_orders'}).reset_index()
    user_item_feat=pd.merge(user_item_feat, user_total_order_number, how='left',on=['user_id'])
    user_item_feat['user_reordered_orders_frac']=user_item_feat.user_total_reordered_orders/(user_item_feat.user_total_orders*1.0)
    del user_total_order_number
    gc.collect()
    
    #用户平均每次下订单相隔的天数
    item_add_to_cart=xprior[['user_id','order_number','days_since_prior_order']].drop_duplicates()[['user_id','days_since_prior_order']].groupby(['user_id'])['days_since_prior_order'].agg(['mean','std']).rename(columns={'mean':'user_order_days_mean','std':'user_order_days_std'}).reset_index()
    user_item_feat=pd.merge(user_item_feat, item_add_to_cart, how='left',on=['user_id'])
    del item_add_to_cart
    gc.collect()
    
    #用户重复购买商品的次数和购买次数
    total_item_count_in_user=xprior[['user_id','product_id','reordered']].groupby(['user_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'total_item_sum','count':'total_item_count'}).reset_index()
    #用户重复购买的次数占比
    total_item_count_in_user['total_user_item_reordered_frac']=total_item_count_in_user.total_item_sum/(1.0*total_item_count_in_user.total_item_count)
    #用户购买的小分类数和包含的重复购买的小分类次数
    total_aisle_count_in_user=xprior[['user_id','aisle_id','reordered']].groupby(['user_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'total_aisle_sum','count':'total_aisle_count'}).reset_index()
    #用户重复购买的小分类数占比
    total_aisle_count_in_user['total_user_aisle_reordered_frac']=total_aisle_count_in_user.total_aisle_sum/(1.0*total_aisle_count_in_user.total_aisle_count)
    #用户订购的大分类数和包含的重复购买的小大分类数
    total_depart_count_in_user=xprior[['user_id','department_id','reordered']].groupby(['user_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'total_depart_sum','count':'total_depart_count'}).reset_index()
    #用户重复购买的大分类数占比
    total_depart_count_in_user['total_user_depart_reordered_frac']=total_depart_count_in_user.total_depart_sum/(1.0*total_depart_count_in_user.total_depart_count)
    
    print 'merge total_order_number feature: %s'%datetime.datetime.now()  
    user_item_feat=pd.merge(user_item_feat, total_item_count_in_user, how='left',on=['user_id'])
    user_item_feat=pd.merge(user_item_feat, total_aisle_count_in_user, how='left',on=['user_id'])
    user_item_feat=pd.merge(user_item_feat, total_depart_count_in_user, how='left',on=['user_id'])
    del total_item_count_in_user
    del total_aisle_count_in_user
    del total_depart_count_in_user
    gc.collect()
    
    #用户购买商品在购买的某小类商品占的比例
    user_item_feat['total_item_aisle_sum_frac']=user_item_feat.total_item_sum/(user_item_feat.total_aisle_sum*1.0)
    user_item_feat['total_item_aisle_count_frac']=user_item_feat.total_item_count/(user_item_feat.total_aisle_count*1.0)
    #用户购买商品在购买的某大类商品占的比例
    user_item_feat['total_item_depart_count_frac']=user_item_feat.total_item_count/(user_item_feat.total_depart_count*1.0)
    user_item_feat['total_item_depart_sum_frac']=user_item_feat.total_item_sum/(user_item_feat.total_depart_sum*1.0)
    #用户购买小类在购买的某大类商品占的比例
    user_item_feat['total_aisle_depart_count_frac']=user_item_feat.total_aisle_count/(user_item_feat.total_depart_count*1.0)
    user_item_feat['total_aisle_depart_sum_frac']=user_item_feat.total_aisle_sum/(user_item_feat.total_depart_sum*1.0)
    
    user_item_feat['total_user_item_numbers_per_order']=user_item_feat.total_item_count/(user_item_feat.user_total_orders*1.0)
    user_item_feat['total_user_item_reordered_numbers_per_order']=user_item_feat.total_item_sum/(user_item_feat.user_total_orders*1.0)
    user_item_feat['total_user_aisle_numbers_per_order']=user_item_feat.total_aisle_count/(user_item_feat.user_total_orders*1.0)
    user_item_feat['total_user_aisle_reordered_numbers_per_order']=user_item_feat.total_aisle_sum/(user_item_feat.user_total_orders*1.0)
    user_item_feat['total_user_depart_numbers_per_order']=user_item_feat.total_depart_sum/(user_item_feat.user_total_orders*1.0)
    user_item_feat['total_user_depart_reordered_numbers_per_order']=user_item_feat.total_depart_count/(user_item_feat.user_total_orders*1.0)


    print 'item_count_in_user feature: %s'%datetime.datetime.now()  
    #用户重复购买的商品数和购买的总商品数
    item_count_in_user=xprior[['user_id','product_id','reordered']].drop_duplicates().groupby(['user_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'item_sum','count':'item_count'}).reset_index()
    #用户重复购买的次数占比
    item_count_in_user['item_reordered_frac']=item_count_in_user.item_sum/(1.0*item_count_in_user.item_count)
    #用户购买的小分类数和包含的重复购买的小分类次数
    aisle_count_in_user=xprior[['user_id','aisle_id','reordered']].drop_duplicates().groupby(['user_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'aisle_sum','count':'aisle_count'}).reset_index()
    #用户重复购买的小分类数占比
    aisle_count_in_user['aisle_reordered_frac']=aisle_count_in_user.aisle_sum/(1.0*aisle_count_in_user.aisle_count)
    #用户订购的大分类数和包含的重复购买的小大分类数
    depart_count_in_user=xprior[['user_id','department_id','reordered']].drop_duplicates().groupby(['user_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'depart_sum','count':'depart_count'}).reset_index()
    #用户重复购买的大分类数占比
    depart_count_in_user['depart_reordered_frac']=depart_count_in_user.depart_sum/(1.0*depart_count_in_user.depart_count)
    
    print 'merge item_count_in_user feature: %s'%datetime.datetime.now()  
    user_item_feat=pd.merge(user_item_feat, item_count_in_user, how='left',on=['user_id'])
    user_item_feat=pd.merge(user_item_feat, aisle_count_in_user, how='left',on=['user_id'])
    user_item_feat=pd.merge(user_item_feat, depart_count_in_user, how='left',on=['user_id'])
    
    del item_count_in_user
    del aisle_count_in_user
    del depart_count_in_user
    gc.collect()
    
    #用户购买商品在购买的某小类商品占的比例
    user_item_feat['item_aisle_sum_frac']=user_item_feat.item_sum/(user_item_feat.aisle_sum*1.0)
    user_item_feat['item_aisle_count_frac']=user_item_feat.item_count/(user_item_feat.aisle_count*1.0)
    #用户购买商品在购买的某大类商品占的比例
    user_item_feat['item_depart_count_frac']=user_item_feat.item_count/(user_item_feat.depart_count*1.0)
    user_item_feat['item_depart_sum_frac']=user_item_feat.item_sum/(user_item_feat.depart_sum*1.0)
    #用户购买小类在购买的某大类商品占的比例
    user_item_feat['aisle_depart_count_frac']=user_item_feat.aisle_count/(user_item_feat.depart_count*1.0)
    user_item_feat['aisle_depart_sum_frac']=user_item_feat.aisle_sum/(user_item_feat.depart_sum*1.0)

    user_item_feat['user_item_numbers_per_order']=user_item_feat.item_count/(user_item_feat.user_total_orders*1.0)
    user_item_feat['user_item_reordered_numbers_per_order']=user_item_feat.aisle_sum/(user_item_feat.user_total_orders*1.0)
    user_item_feat['user_aisle_numbers_per_order']=user_item_feat.aisle_count/(user_item_feat.user_total_orders*1.0)
    user_item_feat['user_aisle_reordered_numbers_per_order']=user_item_feat.aisle_sum/(user_item_feat.user_total_orders*1.0)
    user_item_feat['user_depart_numbers_per_order']=user_item_feat.depart_count/(user_item_feat.user_total_orders*1.0)
    user_item_feat['user_depart_reordered_numbers_per_order']=user_item_feat.depart_sum/(user_item_feat.user_total_orders*1.0)
    
    #item features
    #商品重复购买的次数和总购买次数
    print 'total_item_buy_times feature: %s'%datetime.datetime.now()  
    total_item_buy_times=xprior[['product_id','reordered']].groupby(['product_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'total_item_reorder_times','count':'total_item_buy_times'}).reset_index()
    #商品重复购买的次数占比
    total_item_buy_times['total_item_reordered_frac']=total_item_buy_times.total_item_reorder_times/(1.0*total_item_buy_times.total_item_buy_times)
    #小分类重复购买次数和总购买次数
    total_aisle_buy_times=xprior[['aisle_id','reordered']].groupby(['aisle_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'total_aisle_reorder_times','count':'total_aisle_buy_times'}).reset_index()
    #小分类重复购买的次数占比
    total_aisle_buy_times['total_aisle_reordered_frac']=total_aisle_buy_times.total_aisle_reorder_times/(1.0*total_aisle_buy_times.total_aisle_buy_times)
    #大分类数重复购买次数和总购买次数
    total_depart_buy_times=xprior[['department_id','reordered']].groupby(['department_id'])['reordered'].agg(['sum','count']).rename(columns={'sum':'total_depart_reorder_times','count':'total_depart_buy_times'}).reset_index()
    #重复购买大分类次数占比
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
    #购买该商品的用户数和重复购买该商品的用户数
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
	
#用要预测的订单构造正样本，用历史订单构造负样本，并对负样本适当采样
def get_label(xprior, predict_order, train_detail, sample_rate=-1):
    xpredict_order=pd.merge(predict_order, train_detail, how='left', on='order_id')
    pos_label=xpredict_order[xpredict_order.reordered==1][['user_id', 'product_id', 'reordered']]

    user_item_pairs=xprior[['user_id', 'product_id']].drop_duplicates()
    all_label=pd.merge(user_item_pairs, pos_label, how='left', on=['user_id', 'product_id'])
    all_label.fillna(0, inplace=True)
    
    if sample_rate==-1:#不采样
        label=all_label
    else:#从负样本中采样
        sample_n=pos_label.shape[0]*sample_rate
        if (sample_n+pos_label.shape[0])<all_label.shape[0]:
            all_negative_label=all_label[all_label.reordered==0]
            negative_label=all_negative_label.sample(sample_n)
            label=pd.concat([pos_label, negative_label])
        else:
            label=all_label
    label = shuffle(label)

    return label

#把订单转化为用户-重复购买产品列表的DataFrame格式，方便评测
def convert_order_to_reordered_result(users, orders):
    uid=pd.DataFrame(users)
    uid.columns=['user_id']
    user_reordered_products=orders[orders.reordered==1][['user_id','product_id']].groupby('user_id').agg(lambda x: ' '.join(x.astype(str))).reset_index()
    reordered_result=pd.merge(uid, user_reordered_products, how='left', on='user_id')
    reordered_result.fillna('None', inplace=True)
    return reordered_result

#用户-重复购买产品列表的DataFrame格式转化为python字典，方便后面计算平均F1
def reorder_df_to_dict(result):
    result=result.set_index('user_id')
    result_dict=dict()
    for uid in result.index.values:
        result_dict[uid]=set(result.loc[uid,'product_id'].split(' '))
    return result_dict
	
#取出特征集合和label
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

#获取训练集实际重复购买结果
def get_train_reordered_result(orders, train):
    train_reordered=pd.merge(orders[orders.eval_set=='train'][['order_id','user_id']],train[['order_id','product_id','reordered']],how='left', on='order_id')
    train_reordered_result=convert_order_to_reordered_result(train_reordered.user_id.unique().tolist(), train_reordered)
    return train_reordered_result

    
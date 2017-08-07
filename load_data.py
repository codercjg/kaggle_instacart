# -- coding: utf-8 --
import pandas as pd
import numpy as np
import datetime
import gc

#加载数据，线下训练先用小数据集，线上训练用完整数据集
def load_data(data):
    print 'load data: %s'%datetime.datetime.now()  
    if data=='all':
        orders=pd.read_csv('orders.csv', dtype={
                                'order_id': np.int32,
                                'user_id': np.int64,
                                'eval_set': 'category',
                                'order_number': np.int16,
                                'order_dow': np.int8,
                                'order_hour_of_day': np.int8,
                                'days_since_prior_order': np.float32})

        train=pd.read_csv('order_products__train.csv', dtype={
                            'order_id': np.int32,
                            'product_id': np.uint16,
                            'add_to_cart_order': np.int16,
                            'reordered': np.int8})

        prior=pd.read_csv('order_products__prior.csv', dtype={
                            'order_id': np.int32,
                            'product_id': np.uint16,
                            'add_to_cart_order': np.int16,
                            'reordered': np.int8})
    elif data=='train':
        orders=pd.read_csv('sample_orders.csv', dtype={
                                'order_id': np.int32,
                                'user_id': np.int64,
                                'eval_set': 'category',
                                'order_number': np.int16,
                                'order_dow': np.int8,
                                'order_hour_of_day': np.int8,
                                'days_since_prior_order': np.float32})

        train=pd.read_csv('sample_train.csv', dtype={
                            'order_id': np.int32,
                            'product_id': np.uint16,
                            'add_to_cart_order': np.int16,
                            'reordered': np.int8})

        prior=pd.read_csv('sample_prior.csv', dtype={
                            'order_id': np.int32,
                            'product_id': np.uint16,
                            'add_to_cart_order': np.int16,
                            'reordered': np.int8})
    elif data=='valid':
	orders=pd.read_csv('valid_orders.csv', dtype={
                            'order_id': np.int32,
                            'user_id': np.int64,
                            'eval_set': 'category',
                            'order_number': np.int16,
                            'order_dow': np.int8,
                            'order_hour_of_day': np.int8,
                            'days_since_prior_order': np.float32})

        train=pd.read_csv('valid_train.csv', dtype={
                            'order_id': np.int32,
                            'product_id': np.uint16,
                            'add_to_cart_order': np.int16,
                            'reordered': np.int8})

        prior=pd.read_csv('valid_prior.csv', dtype={
                            'order_id': np.int32,
                            'product_id': np.uint16,
                            'add_to_cart_order': np.int16,
                            'reordered': np.int8})
    else:
	print 'load error'
		
    products = pd.read_csv('products.csv')
    aisles = pd.read_csv("aisles.csv")
    departments = pd.read_csv("departments.csv")
    sample_submission = pd.read_csv("sample_submission.csv")
    print 'load data ok: %s'%datetime.datetime.now() 
    return (orders, prior, train, products, aisles, departments, sample_submission)

#选择部分数据
def select_data_from_all(user_number):
    orders, prior, train, products, aisles, departments, sample_submission=load_data('all')
    all_users=orders.user_id.unique().tolist()
    select_users=all_users[-user_number:]
    select_orders=orders[orders.user_id.isin(select_users)]
    select_order_ids=select_orders.order_id.unique().tolist()
    select_train=train[train.order_id.isin(select_order_ids)]
    select_prior=prior[prior.order_id.isin(select_order_ids)]
    del orders
    del prior
    del train
    gc.collect()
    return (select_orders, select_prior, select_train, products, aisles, departments, sample_submission)

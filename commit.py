# -- coding: utf-8 --
import pandas as pd
import numpy as np
from eval import *
from extract_features import *

def list_to_str(ll):
    ret=''
    for e in ll:
        ret=ret+str(e)+" "
        
    return ret  
	
def eval_f1(user_item_pairs, user_order, train, pred_label):
    user_item_pairs['pred']=pred_label
    predict=pd.merge(user_item_pairs, user_order, how='left', on='user_id')
    train_predict=predict[predict.eval_set=='train']
    train_result=pd.merge(train_predict, train, how='left', on=['order_id','product_id'])
    train_result.reordered.fillna(0, inplace=True)
    f1=cal_f1(train_result[['user_id','product_id','reordered']], train_result.pred.tolist())
    return f1

#生成线上提交文件	
def generate_online_evalueate_file(user_item_pairs, user_order, valid_pred_label, sample_submission, filepath):
    print 'generate sbumit file'
    user_item_pairs['pred']=valid_pred_label
    predict=pd.merge(user_item_pairs,user_order, how='left', on='user_id')
    test_predict=predict[predict.pred==1][['order_id', 'product_id']]
    test_predict_result=test_predict.groupby('order_id').agg(lambda x: ' '.join(x.astype(str))).reset_index()
    test_predict_result.columns=sample_submission.columns.tolist()
    submit=pd.merge(sample_submission[['order_id']],test_predict_result, how='left', on='order_id').fillna('None')
    submit.to_csv(filepath, index=False)
    print 'generate online file %s ok'%filepath	

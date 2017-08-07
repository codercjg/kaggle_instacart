# -- coding: utf-8 --
import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
import xgboost as xgb
import operator
from load_model import *
from extract_features import *
from eval import *
from load_model import *
from commit import *
from load_data import *
from best_f1 import *

#python 2.7 

#画出xgboost特征重要性
def get_xgb_feature_importance(model, x, draw=True):
    features = [f for f in x.columns]  
    #ceate_feature_map(features)  
    outfile = open('xgb.fmap', 'w')  
    i = 0  
    for feat in features:  
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))  
        i = i + 1  
    outfile.close()
    
    importance = model.get_fscore(fmap='xgb.fmap')  
    importance = sorted(importance.items(), key=operator.itemgetter(1))  
  
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])  
    df['fscore'] = df['fscore'] / df['fscore'].sum()
  
    if draw:
        plt.figure()  
        df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))  
        plt.title('XGBoost Feature Importance')  
        plt.xlabel('relative importance')  
        plt.show()
    
    df.sort_values('fscore', ascending =False, inplace=True)
    df.to_csv("feat_importance.csv", index=False)
    
    return df
	
#xgboost train
def xgb_model_train(x, y, test_x=None, test_y=None):
    xgb_params = {
        'max_depth':5, 
        'learning_rate':0.1,
        'n_estimators':400, 
        'silent':True, 
        'objective':'binary:logistic', 
        'booster':'gbtree', 
        'n_jobs':1, 
        'nthread':0, 
        'gamma':0.5, 
        'min_child_weight':3, 
        'max_delta_step':0, 
        'subsample':0.8, 
        'colsample_bytree':0.9, 
        #'scale_pos_weight':1,
        'scale_pos_weight':(len(y)-np.sum(y))/(np.sum(y)*1.0), 
        'eval_metric': 'auc',
        'random_state':1,
        'reg_alpha': 2
    }
    
    dtrain = xgb.DMatrix(x, y)
    watchlist = [(dtrain, 'train')]

    #dtest = xgb.DMatrix(test_x, test_y)
    #watchlist = [(dtrain, 'train'),(dtest, 'test')]

    #model = xgb.train(dict(xgb_params, silent=0), dtrain, early_stopping_rounds=50, num_boost_round=400, evals=watchlist,feval=xgb_f1)
    model = xgb.train(dict(xgb_params, silent=0), dtrain, early_stopping_rounds=20, num_boost_round=180,evals=watchlist)

    scores = model.predict(dtrain)

    model_save(model, 'xgb.txt')
    return (model,scores)
	
#xgboost predict
def xgb_model_predict(model, x):
    dtrain = xgb.DMatrix(x)
    scores = model.predict(dtrain)
    return scores
	
	
if __name__ == '__main__':
    
    orders, prior, train, products, aisles, departments, sample_submission=load_data('all')
    xorders=orders_prepare(orders)
    
    reordered_result=get_train_reordered_result(orders, train)
    train_x, train_y, test_x, test_ui=get_x_y(xorders, prior, products, train)
    print train_x.shape, test_x.shape
 
    feat_data=pd.concat([train_y,train_x], axis=1)
    users=feat_data.user_id.unique().tolist()
    user_number=len(users)
    train_users=users[:user_number-10000]
    test_users=users[user_number-10000:]
    print 'split train test feat'
    
    train_feat_data=feat_data[feat_data.user_id.isin(train_users)]
    test_feat_data=feat_data[feat_data.user_id.isin(test_users)]

    train_Y=train_feat_data[['user_id','product_id', 'reordered']]
    train_X=train_feat_data.drop(['user_id','product_id', 'reordered'], axis=1)

    test_Y=test_feat_data[['user_id','product_id', 'reordered']]
    test_X=test_feat_data.drop(['user_id','product_id', 'reordered'], axis=1)

    reordered_result=get_train_reordered_result(orders, train)
    train_reordered_result=reordered_result[reordered_result.user_id.isin(train_users)]
    test_reordered_result=reordered_result[reordered_result.user_id.isin(test_users)]
    print train_X.shape, test_X.shape
    print 'train xgboost'
    xgb_model,train_scores=xgb_model_train(train_X, train_Y.reordered.tolist(),test_X, test_Y.reordered.tolist())	
    display_score(train_Y.reordered.tolist(), train_scores)

    print 'find best train threshold'    
    best_f1, best_t=find_best_avg_f1_pred(train_reordered_result, train_Y[['user_id', 'product_id']], train_scores, thresholds=np.linspace(0.65, 0.75, 10))

    print 'predict online result'
    online_scores=xgb_model_predict(xgb_model, test_x)
    print 'using train best_t to generate online result'
    test_pred_label=get_label_from_scores(online_scores, best_t)
    generate_online_evalueate_file(test_ui, orders, test_pred_label, sample_submission, 'xgb_commit_train_t.csv')
    
    print 'predict offline result'    
    test_pred_result=test_Y.copy()
    test_scores=xgb_model_predict(xgb_model, test_X)
	
    test_pred_label=get_label_from_scores(test_scores, best_t)
    test_pred_result['reordered']=test_pred_label
    pred_reordered_result_df=convert_order_to_reordered_result(test_reordered_result.user_id.unique().tolist(), test_pred_result)
    avg_f1=cal_avg_f1(reorder_df_to_dict(test_reordered_result), reorder_df_to_dict(pred_reordered_result_df))
    print 'test avg_f1: %0.05f'%avg_f1	
	
    print 'find best test threshold'    
    best_f1, best_t=find_best_avg_f1_pred(test_reordered_result, test_Y[['user_id', 'product_id']], test_scores, thresholds=np.linspace(0.65, 0.75, 10))	
	    
    print 'using test best_t to generate online result'
    #test_scores=xgb_model_predict(xgb_model, test_x)
    test_pred_label=get_label_from_scores(online_scores, best_t)
    generate_online_evalueate_file(test_ui, orders, test_pred_label, sample_submission, 'xgb_commit_test_t.csv')

    print 'save test UI score'
    test_ui.to_csv('test_score.csv', index=False)
    print 'finish'

# -- coding: utf-8 --
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from extract_features import *
from best_f1 import *
		
#根据阈值转化为0,1	
def get_label_from_scores(scores, threshold):
	return [np.int32(score>threshold) for score in scores]

def cal_avg_f1(label_dict, predict_dict):
    total_f1 = 0
    for user, products in label_dict.items():
        predict_products=predict_dict[user]
        tp=len(predict_products & products)
        precision=tp/(len(predict_products)*1.0)
        recall=tp/(len(products)*1.0)
        if tp==0:
            f1=0
        else:
            f1=2*precision*recall/(precision+recall)
        total_f1+=f1
    avg_f1=total_f1/(len(label_dict)*1.0)
    return avg_f1
'''
usage:

t1={1:set(['None']),2:set(['1'])}
t2={1:set(['None']), 2:set(['None'])
cal_avg_f1(t1, t2)
0.5
'''

#选取使F1最大的阈值，预测是否购买
def find_best_avg_f1_pred(reordered_result, user_item_pairs, pred_socres, thresholds=np.linspace(0.1, 0.95, 10)):
    best_avg_f1=0
    best_t=0.5
    for t in thresholds:
        print 'thresholds %0.05f'%t
        pred=get_label_from_scores(pred_socres, t)
        user_item_pairs['reordered']=pred
        pred_reordered_result=convert_order_to_reordered_result(reordered_result.user_id.unique().tolist(), user_item_pairs)
        avg_f1=cal_avg_f1(reorder_df_to_dict(reordered_result), reorder_df_to_dict(pred_reordered_result))
   
        if avg_f1>best_avg_f1:
            best_t=t
            best_avg_f1=avg_f1
        print 'avg f1 %0.05f'%avg_f1
    print 'best_f1:%.05f, threshold: %.05f'%(best_avg_f1,best_t)
    return (best_avg_f1, best_t)
#find_best_f1_pred(reordered_result, train_y[['user_id', 'product_id']], train_scores, thresholds=np.linspace(0.1, 0.95, 10))
	
#画出prc、roc曲线
def display_score(y, scores, t=0.5, draw=False):

	if draw:
		precision, recall, thresholds = precision_recall_curve( y, scores)
		average_precision = average_precision_score(y, scores, average="micro")
    
		lw = 2
        
		# Plot Precision-Recall curve
		plt.clf()
		plt.plot(recall, precision, lw=lw, color='navy',
			label='Precision-Recall curve')
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.ylim([0.0, 1.05])
		plt.xlim([0.0, 1.0])
		plt.title('Precision-Recall: ROC={0:0.5f}'.format(average_precision))
		plt.legend(loc="lower left")
		plt.show()

	fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
	auc=metrics.auc(fpr, tpr)
	print 'auc:%0.5f'%auc

	if draw:
		plt.figure()
		plt.plot(fpr, tpr, color='darkorange',lw=lw, label='auc curve (area = %0.5f)' % auc)
		#plt.plot(fpr, tpr, color='darkorange',lw=lw)
		plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic (area = %0.5f)'% auc)
		plt.legend(loc="lower right")
		plt.show()

def max_f1_result(ui, predict_score):
	ui['score']=predict_score
	groups=ui[['user_id','product_id','score']].sort_values('score', ascending=False).groupby('user_id')
	predict_result=pd.DataFrame()
	for user, product_score in groups:
		P=product_score['score']
		best_k, predNone, max_f1 = F1Optimizer.maximize_expectation(P)
		predict_result=pd.concat([predict_result, product_score[['user_id', 'product_id']].head(best_k)])
	return predict_result
	

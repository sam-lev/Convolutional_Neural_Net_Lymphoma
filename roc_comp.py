import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
import scikitplot as skplt
from os import listdir
from os.path import isfile, join

bl_path = "./predictions/BL"#./exp_2_predictions/BL" # prediction_results/BL"#fourth_results/BL"#  full_training_predictions/BL" #
dl_path = "./predictions/DLBCL"#./exp_2_predictions/DLBCL" # prediction_results/DLBCL"#fourth_results/DLBCL"#  full_training_predictions/DLBCL" #
BL_files = [join(bl_path,f) for f in listdir(bl_path) if isfile(join(bl_path, f)) and ("txt" in f or "rtf" in f or "rtfd" in f)]
DL_files = [join(dl_path,f) for f in listdir(dl_path) if isfile(join(dl_path, f)) and ("txt" in f or "rtfd" in f  or "rtf" in f)]

def precision(tp, fp):
    return tp/(tp+fp)
def recall(tp, fn):
    return float(tp)/(tp+fn)
def F1(pr, re):
    return 2.0*((pr*re)/(pr+re))
def FP_rate(fp, tn):
    return float(fp)/(fp+tn)

# False Positive: Truly BL but predicts DLBCL
# True Positive: Truly DLBCL predicted DLBCL

print("Total Files DL: ", len(DL_files))
print("Total Files BL: ", len(BL_files))

FP = []
TP = []
TN = []
FN = []
pred_dl_neg = []
pred_dl_pos = []
pred_bl_neg = []
pred_bl_pos = []
thresh = np.linspace(0,1,10)
F1_thresh =  np.zeros(len(thresh))
num_neg = 0
for case in BL_files:

    f = open(case, "r")
    lines = f.readlines()[:-2]
    for line in lines:
        num_neg+=1
        comma = ','
        bracket = '['
        under_score = '_'
        if comma in line:
            pbl, pdl = line.split(',')
            if bracket in pbl:
                pbl = pbl.split('[')[1]#.split(',')[0]       
        else:
            pbl, pdl = line.split()
            if bracket in pbl:
                pbl = line.split('[')[1].split()[0]
        pbl = float(pbl)
        pdl = float(pdl)
        pred_dl_neg.append(pdl)
        pred_bl_pos.append(pbl)
        if pbl > 0.5:
            TN.append(pbl)
        else:
            FP.append(pdl)
        #print(pbl, " ", pdl)
print(len(TN))
print(">>>")
print(len(FP))
num_pos=0        
for case in DL_files:

    f = open(case, "r")
    lines = f.readlines()[:-2]
    for line in lines:
        num_pos +=1
        comma = ','
        bracket = '['
        if comma in line:
            pbl, pdl = line.split(',')
            if bracket in pbl:
                pbl = pbl.split('[')[1]#.split(',')[0]              
        else:
            pbl, pdl = line.split()
            if bracket in pbl:
                pbl = line.split('[')[1].split()[0]
        pbl = float(pbl)
        pdl = float(pdl)
        pred_dl_pos.append(pdl)
        pred_bl_neg.append(pbl)
        if pdl > 0.5:
            TP.append(pdl)
        else:
            FN.append(pbl)
        #print(pbl, " ", pdl)

fp_count = np.zeros(len(thresh))

for i,t in zip(range(len(thresh)), thresh):
    for fp in FP:
        if fp >= t:
            fp_count[i] += 1
    
tp_count = np.zeros(len(thresh))

for i,t in zip(range(len(thresh)), thresh):
    for tp in TP:
        if tp >= t:
            tp_count[i] += 1

tn_count = np.zeros(len(thresh))

for i,t in zip(range(len(thresh)), thresh):
    for tn in TN:
        if tn >= t:
            tn_count[i] += 1

fn_count = np.zeros(len(thresh))

for i,t in zip(range(len(thresh)), thresh):
    for fn in FN:
        if fn >= t:
            fn_count[i] += 1

# Compute F1 at each threshold
for i in range(len(F1_thresh)):
    pr = precision(tp_count[i],fp_count[i])
    re = recall(tp_count[i], fn_count[i])
    F1_thresh[i] = F1(pr,re)

fp_rate = np.zeros(len(thresh))
tp_rate = np.zeros(len(thresh))
for i in range(len(fp_rate)):
    tp_r = recall(tp_count[i], fn_count[i])
    fp_r = FP_rate(fp_count[i], tn_count[i])
    fp_rate[i] = fp_r
    tp_rate[i] = tp_r

def ci_95(FP, FN, TP, TN, z=1.96):
    z_a = z
    error = (FP + FN) / float(TP + TN + FP + FN)
    print("Error: ", error)
    std_err = np.sqrt((float(error)*(1.0-error))/float(TP+FN))
    return z_a*std_err
print("")
print("------------------------------------------------------------")
print("| True Negative (TN): ", len(TN), "  |  ","False Positive (FP): ", len(FP),"|")
print("-----------------------------------------------------------")
print("| False Negative (FN): ", len(FN),"  |  ","True Positive (TP): ", len(TP)," |")
print("------------------------------------------------------------")
print("")
print("----------")
print("Total DLBCL: ", len(TP)+len(FN))
print("Total BL: ", len(FP) + len(TN))
print("Total Images: ",len(FN)+len(FP)+len(TP)+len(TN))
print("----------")
print("")
print("----------")
print("Max F1: ",np.max(F1_thresh))
print("F1 score (+/- CI_95): ", F1(precision(len(TP),len(FP)), recall(len(TP),len(FN)))," +/- ",  ci_95(len(FP),len(FN),len(TP),len(TN)))
print("----------")
#print(DL_files,BL_files)

#print(F1(precision(592.,147.),recall(592., 139.)))
print('min(tp, tn, fp,fn) ',np.min(tp_count),' ',np.min(tn_count),' ',np.min(fp_count),' ',np.min(fn_count))
print('max(tp,tn,fp,fn) ',np.max(tp_count),' ',np.max(tn_count),' ',np.max(fp_count),' ',np.max(fn_count))

y_dl_true = list(np.ones(num_pos))+list(np.zeros(num_neg))
y_dl_probas = pred_dl_pos+pred_dl_neg

y_bl_true = list(np.ones(num_neg))+list(np.zeros(num_pos))
y_bl_probas = pred_bl_pos+pred_bl_neg

fpr_dl, tpr_dl, threshold_dl = metrics.roc_curve(y_dl_true, y_dl_probas,pos_label=1)
fpr_bl, tpr_bl, threshold_bl = metrics.roc_curve(y_bl_true, y_bl_probas,pos_label=1)

roc_auc_dl = metrics.auc(fpr_dl, tpr_dl)
roc_auc_bl = metrics.auc(fpr_bl, tpr_bl)

print("AUC DLBCL: ", roc_auc_dl)

#roc_auc = metrics.auc(fp_rate, tp_rate)
font = {'family' : 'Calibri',
        'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)

#plt.title('ROC (fourth of training set)')
plt.plot(fpr_dl, tpr_dl, color='mediumspringgreen', linewidth=1.3,linestyle=':',label = 'Accuracy (AUC) DLBCL = %0.2f' % roc_auc_dl)
#plt.plot(fpr_bl, tpr_bl, color='mediumblue', linestyle='-.',linewidth=1.3,label = 'Accuracy (AUC) BL = %0.2f' % roc_auc_bl)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],color='darkgray',linestyle='dashed')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate',fontname="Calibri", fontsize=20, fontweight='bold')
plt.xlabel('False Positive Rate',fontname="Calibri", fontsize=20, fontweight='bold')
plt.show()


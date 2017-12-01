import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from scipy import interp

def evaluate_model(x, y, classifier, preprocess_train=None):
    
    if preprocess_train is None:
        def ll(x,y):
            return x,y
        preprocess_train = ll

    cv = StratifiedKFold(n_splits=10)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    importances = []

    i = 0
    plt.figure(figsize=(20,10))
    for train, test in cv.split(x, y):
        
        x_train, y_train = preprocess_train(x.iloc[train], y.iloc[train])

        probas_ = classifier.fit(x_train, y_train).predict_proba(x.iloc[test])
        importances.append(classifier.steps[1][1].feature_importances_)
        
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))   
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(bbox_to_anchor=(1.04,1), loc="lower right")
    plt.show()

    return importances
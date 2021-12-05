# -*- coding: utf-8 -*-
"""
demo ROC
@author: yw
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
print(confusion_matrix(y_true, y_pred))


from sklearn.metrics import  roc_curve

pos_class = 2
class_true = np.array([1, 1, 2, 2])
prob_pos_class = np.array([0.1, 0.4, 0.35, 0.8])

fpr, tpr, thresholds = roc_curve(y_true = class_true,\
                                         y_score = prob_pos_class,\
                                         pos_label = pos_class)

print('FPR = ', fpr)
print('TPR = ', tpr)
print('Thresholds = ', thresholds)

plt.scatter(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC')
plt.show()

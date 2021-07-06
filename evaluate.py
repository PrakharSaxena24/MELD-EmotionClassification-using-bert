from sklearn.metrics import f1_score
import numpy as np
import torch


label_dict= {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
def f1_score_func(preds,labels):
    preds_flat=np.argmax(preds,axis=1).flatten()
    labels_flat=labels.flatten()
    return f1_score(labels_flat,preds_flat,average="weighted")

def accuracy_per_class(preds,labels):
    label_dict_inverse={v:k for k, v in label_dict.items()}
    preds_flat=np.argmax(preds,axis=1).flatten()
    labels_flat=labels.flatten()
    totalacc=0
    tot=0
    for label in np.unique(labels_flat):
        y_preds=preds_flat[labels_flat==label]
        y_true=labels_flat[labels_flat==label]
        totalacc+=len(y_preds[y_preds == label])
        tot+=len(y_true)
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds == label])}/{len(y_true)}\n')
    print("Acc=",totalacc/tot)



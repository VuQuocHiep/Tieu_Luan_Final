"""Unified evaluation and deterministic training for all 3 Kaggle apps."""
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix,classification_report
from .config import ART, METHODS, CONFIG
from .datasets import load_data
from .cnn_numpy import train_scratch
from .backends import train_torch, train_keras, predict_torch, predict_keras


def predict_proba(model,method,x):
    if method=='scratch':return model.predict_proba(x)
    if method=='pytorch':return predict_torch(model,x)
    return predict_keras(model,x)


def fit_one(key,method,opts=None,prepared=None):
    if method not in METHODS:raise ValueError(method)
    opts=opts or {}
    data=prepared if prepared is not None else load_data(key,max_train=opts.get('max_train',1600),max_val=opts.get('max_val',450),
                   max_test=opts.get('max_test',900),seed=opts.get('seed',42))
    epochs=int(opts.get('epochs',3)); batch=int(opts.get('batch_size',64))
    lr=float(opts.get('lr',0.05)); seed=int(opts.get('seed',42))
    fit={'scratch':train_scratch,'pytorch':train_torch,'keras':train_keras}[method]
    model,history=fit(data.x_train,data.y_train,data.x_val,data.y_val,len(data.classes),
                      epochs=epochs,batch_size=batch,lr=lr,seed=seed)
    probs=predict_proba(model,method,data.x_test)
    preds=probs.argmax(1)
    pre,rec,f1,_=precision_recall_fscore_support(data.y_test,preds,average='macro',zero_division=0)
    mat=confusion_matrix(data.y_test,preds,labels=range(len(data.classes))).tolist()
    result={'dataset':key,'method':method,'method_name':METHODS[method],
            'accuracy':float(accuracy_score(data.y_test,preds)),
            'precision_macro':float(pre),'recall_macro':float(rec),'f1_macro':float(f1),
            'n_train':int(len(data.y_train)),'n_val':int(len(data.y_val)),'n_test':int(len(data.y_test)),
            'n_train_original':data.n_train_original,'n_test_original':data.n_test_original,
            'train_class_distribution':data.train_distribution,
            'classes':data.classes,'label_codes':data.label_codes,
            'confusion_matrix':mat,'history':history,
            'classification_report':classification_report(data.y_test,preds,target_names=data.classes,labels=range(len(data.classes)),zero_division=0,output_dict=True),
            'run_settings':{'epochs':epochs,'batch_size':batch,'lr':lr,'seed':seed,
                            'max_train':opts.get('max_train',1600),'max_val':opts.get('max_val',450),
                            'max_test':opts.get('max_test',900)},
            'source':'Original CSV Kaggle', 'is_synthetic':False,
            'created_at':datetime.now(timezone.utc).isoformat()}
    ART.mkdir(parents=True,exist_ok=True)
    prefix=ART / f'{key}_{method}'
    if method=='scratch': model.save(str(prefix)+'.npz')
    elif method=='pytorch':
        import torch
        torch.save(model.state_dict(),str(prefix)+'.pt')
    else: model.save(str(prefix)+'.keras')
    (Path(str(prefix)+'.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    return result


def run_all(key=None,method=None,opts=None):
    keys=[key] if key else list(CONFIG)
    methods=[method] if method else list(METHODS)
    out=[]
    opts=opts or {}
    for k in keys:
        # Read large Kaggle CSV once per application, shared by all three implementations.
        prepared=load_data(k,max_train=opts.get('max_train',1600),max_val=opts.get('max_val',450),
                           max_test=opts.get('max_test',900),seed=opts.get('seed',42))
        for m in methods:
            print(f'Đang huấn luyện {CONFIG[k]["short"]} – {METHODS[m]} ...',flush=True)
            r=fit_one(k,m,opts,prepared=prepared)
            out.append(r)
            print(f'PASS {k:14} {m:8} Accuracy={r["accuracy"]:.4f} Macro-F1={r["f1_macro"]:.4f} ({r["n_test"]} test)',flush=True)
    (ART/'comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8') if key is None and method is None else None
    return out

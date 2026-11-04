"""Load saved CNN artifacts, predict images, and avoid unsafe object deserialization."""
import json
import os
os.environ['KERAS_BACKEND']='torch'
from pathlib import Path
import numpy as np
from .config import ART, METHODS
from .cnn_numpy import CNNFromScratch
from .backends import TorchCNN,predict_keras,predict_torch


def get_result(key,method):
    f=ART/f'{key}_{method}.json'
    if not f.exists(): raise FileNotFoundError('Chưa có mô hình đã train. Chạy python train_all.py trước.')
    return json.loads(f.read_text(encoding='utf-8'))


def load_model(key,method):
    meta=get_result(key,method)
    nc=len(meta['classes'])
    path=ART/f'{key}_{method}'
    if method=='scratch':return CNNFromScratch.load(str(path)+'.npz')
    if method=='pytorch':
        import torch
        m=TorchCNN(nc)
        m.load_state_dict(torch.load(str(path)+'.pt',map_location='cpu',weights_only=True))
        m.eval()
        return m
    if method=='keras':
        import keras
        return keras.models.load_model(str(path)+'.keras',compile=False)
    raise ValueError(method)


def predict_image(model,method,img):
    x=np.asarray(img,dtype=np.float32)
    if x.ndim==2: x=x[None,None,:,:]
    elif x.ndim==3: x=x[:,None,:,:]
    if x.ndim!=4 or x.shape[1:]!=(1,28,28):
        raise ValueError('Đầu vào yêu cầu ảnh xám 28×28.')
    if np.max(x)>1.0:x=x/255.0
    if not np.isfinite(x).all() or np.min(x)<0 or np.max(x)>1:
        raise ValueError('Dữ liệu pixel phải nằm trong [0,1] hoặc [0,255].')
    if method=='scratch':return model.predict_proba(x)
    if method=='pytorch':return predict_torch(model,x)
    return predict_keras(model,x)

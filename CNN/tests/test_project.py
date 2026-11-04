"""Offline integration: real CNN math and 9 backend pipelines on synthetic fixtures.
No accuracy values from these fixtures should be cited as Kaggle results.
"""
import os
os.environ['KERAS_BACKEND']='torch'
import tempfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
import shared.datasets as ds
import shared.config as config
import shared.train_runner as runner
import shared.inference as infer
from shared.cnn_numpy import CNNFromScratch, softmax
from shared.backends import build_torch,build_keras,predict_torch,predict_keras


def create_fixture(key,folder):
    specs=config.CONFIG[key]
    label_codes=([i for i in range(26) if i not in (9,25)] if key=='sign_language' else list(range(10)))
    folder.mkdir(parents=True,exist_ok=True)
    for typ,n_per in [('train',8),('test',4)]:
        pixels=[];labels=[]
        for cl in label_codes:
            for i in range(n_per):
                rng=np.random.default_rng(500*cl+i+int(typ=='test'))
                im=rng.integers(0,12,(28,28),dtype=np.uint8)
                # class-specific visible 3x3 bright square
                x=2+(cl%8)*3
                y=2+(cl//8)*7
                im[y:y+3,x:x+3]=220
                pixels.append(im.reshape(-1));labels.append(cl)
        data=np.array(pixels)
        df=pd.DataFrame(data,columns=[f'pixel{i}' for i in range(784)])
        df.insert(0,'label',labels)
        df.to_csv(folder/specs[typ],index=False)


def check_initial_equivalence():
    x=np.random.default_rng(3).random((4,1,28,28),dtype=np.float32)
    a=CNNFromScratch(10,seed=42).predict_proba(x)
    b=predict_torch(build_torch(10,seed=42),x)
    c=predict_keras(build_keras(10,seed=42),x)
    assert np.max(abs(a-b))<1e-5,('NumPy/PyTorch mismatch',np.max(abs(a-b)))
    assert np.max(abs(a-c))<1e-5,('NumPy/Keras mismatch',np.max(abs(a-c)))
    print('PASS initial weights/forward equivalence NumPy - PyTorch - Keras')


def check_gradient():
    # Finite-difference check on one learnable conv element.
    rng=np.random.default_rng(5)
    x=rng.random((3,1,28,28),dtype=np.float32)
    y=np.array([0,1,2])
    m=CNNFromScratch(3,seed=11)
    def loss():
        p=m.predict_proba(x)
        return -np.log(p[np.arange(len(y)),y]).mean()
    original=m.params['conv_w'][0,0,0,0].copy()
    eps=1e-3
    m.params['conv_w'][0,0,0,0]=original+eps;plus=loss()
    m.params['conv_w'][0,0,0,0]=original-eps;minus=loss()
    m.params['conv_w'][0,0,0,0]=original
    grad_fd=(plus-minus)/(2*eps)
    before=original.copy()
    lr=1e-2
    m.train_batch(x,y,lr)
    grad_bp=(before-m.params['conv_w'][0,0,0,0])/lr
    assert abs(grad_fd-grad_bp)<4e-3,(grad_fd,grad_bp)  # max-pool/ReLU finite differences can cross kinks
    print('PASS manual Conv2D backprop finite-difference gradient')


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp)
        ds.RAW=tmp/'data'
        runner.ART=tmp/'models'
        infer.ART=tmp/'models'
        check_initial_equivalence()
        check_gradient()
        for key in config.CONFIG:
            create_fixture(key,ds.RAW/key)
            bundle=ds.load_data(key,max_train=96 if key=='sign_language' else 48,max_val=48,
                                max_test=72 if key=='sign_language' else 30)
            assert bundle.x_train.shape[1:]==(1,28,28)
            assert len(bundle.classes)==len(config.CONFIG[key]['classes'])
            # All tests on synthetic CSV: NOT empirical Kaggle benchmark results.
            settings=dict(max_train=96 if key=='sign_language' else 48,max_val=48,
                          max_test=72 if key=='sign_language' else 30,epochs=1,batch_size=16,lr=.05,seed=42)
            for method in config.METHODS:
                result=runner.fit_one(key,method,settings)
                model=infer.load_model(key,method)
                scores=infer.predict_image(model,method,bundle.x_test[:2])
                assert scores.shape==(2,len(bundle.classes))
                assert np.isfinite(scores).all()
                assert np.allclose(scores.sum(axis=1),1.0,atol=1e-5)
                assert 0<=result['accuracy']<=1
                assert result['is_synthetic'] is False  # runner assumes caller provided authentic Kaggle files
                print(f'PASS {key:<14} {method:<7} train/save/reload/infer [synthetic fixture]')
        # A missing Kaggle dataset must stop with a helpful error.
    print('ALL OFFLINE TESTS PASSED — Kaggle benchmark remains unverified until original CSV training.')

if __name__=='__main__':main()

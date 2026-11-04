import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile
from pathlib import Path
import numpy as np

from shared.rnn_scratch import ScratchRNN
from shared.text_utils import build_vocab, encode_texts


def synthetic():
    texts=[]; labels=[]
    groups=[('red apple sweet fruit',0),('goal team match sport',1),('stock market money business',2)]
    for phrase,label in groups:
        for i in range(36):
            texts.append(f'{phrase} example {i%5}')
            labels.append(label)
    return texts, np.array(labels,dtype=np.int64)


def test_tokenizer_and_scratch():
    texts,y=synthetic(); vocab=build_vocab(texts,100); x=encode_texts(texts,vocab,12)
    m=ScratchRNN(len(vocab),8,12,3,seed=7)
    before=m.predict_proba(x[:4])
    assert before.shape==(4,3) and np.allclose(before.sum(1),1,atol=1e-5)
    m.fit(x,y,epochs=10,batch_size=24,lr=.01,seed=7,verbose=False)
    acc=(m.predict(x)==y).mean(); assert acc>.95,acc
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'m.npz'; m.save(p); m2=ScratchRNN.load(p)
        assert np.allclose(m.predict_proba(x[:8]),m2.predict_proba(x[:8]),atol=1e-6)
    print(f'PASS Scratch RNN synthetic accuracy={acc:.3f}, save/reload/infer')


def test_gradient():
    texts=['red apple','blue sky','red fruit','blue ocean']; y=np.array([0,1,0,1])
    vocab=build_vocab(texts,30); x=encode_texts(texts,vocab,5)
    m=ScratchRNN(len(vocab),4,5,2,seed=3)
    loss,gr=m.loss_and_grads(x,y)
    name='Wy'; idx=(0,0); eps=1e-3
    old=m.p[name][idx]
    m.p[name][idx]=old+eps; lp=m.loss_and_grads(x,y)[0]
    m.p[name][idx]=old-eps; lm=m.loss_and_grads(x,y)[0]
    m.p[name][idx]=old
    num=(lp-lm)/(2*eps); ana=gr[name][idx]
    rel=abs(num-ana)/max(1e-5,abs(num)+abs(ana))
    assert rel<2e-2,(num,ana,rel)
    print(f'PASS Scratch BPTT finite-difference gradient rel_error={rel:.3e}')


def test_keras_pytorch():
    # Test each framework independently on a tiny obvious dataset.
    from types import SimpleNamespace
    from shared.models import train_keras, keras_predict_proba, train_pytorch, pytorch_predict_proba, load_keras, load_pytorch
    texts,y=synthetic(); vocab=build_vocab(texts,100); x=encode_texts(texts,vocab,12)
    bundle=SimpleNamespace(x_train=x,y_train=y,x_val=x[:18],y_val=y[:18],classes=['a','b','c'],vocab=vocab)
    opts={'embed_dim':8,'hidden_dim':12,'seed':7,'epochs':6,'batch_size':24,'lr':.01}
    with tempfile.TemporaryDirectory() as td:
        kp=Path(td)/'k.keras'; km,_=train_keras(bundle,opts,kp); pk=keras_predict_proba(km,x); assert (pk.argmax(1)==y).mean()>.9
        km2=load_keras(kp); assert np.allclose(pk[:4],keras_predict_proba(km2,x[:4]),atol=1e-5)
        pp=Path(td)/'p.pt'; pm,_=train_pytorch(bundle,opts,pp); ppy=pytorch_predict_proba(pm,x); assert (ppy.argmax(1)==y).mean()>.9
        pm2=load_pytorch(pp); assert np.allclose(ppy[:4],pytorch_predict_proba(pm2,x[:4]),atol=1e-5)
    print('PASS Keras/PyTorch RNN train/save/reload/infer [synthetic fixture]')

if __name__=='__main__':
    test_tokenizer_and_scratch(); test_gradient(); test_keras_pytorch()
    print('ALL OFFLINE TESTS PASSED — Kaggle benchmark cần chạy sau khi tải dữ liệu thật.')

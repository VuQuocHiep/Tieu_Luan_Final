"""A real, manually differentiated Conv2D -> ReLU -> MaxPool2D -> Dense CNN.

Uses only NumPy for model operations and gradient computations. No autograd,
no torch/keras/model.fit. Layout: NCHW. Valid 3x3 convolution, stride 1.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

FILTERS=4
KH=KW=3
POOL=2
FEATURE_SIZE=FILTERS*13*13


def initial_weights(n_classes,seed=42):
    rg=np.random.default_rng(seed)
    return {
        'conv_w':rg.normal(0,0.10,(FILTERS,1,3,3)).astype(np.float32),
        'conv_b':np.zeros(FILTERS,dtype=np.float32),
        'dense_w':rg.normal(0,0.04,(FEATURE_SIZE,n_classes)).astype(np.float32),
        'dense_b':np.zeros(n_classes,dtype=np.float32),
    }


def softmax(z):
    e=np.exp(z-np.max(z,axis=1,keepdims=True))
    return e/e.sum(axis=1,keepdims=True)


class CNNFromScratch:
    def __init__(self,n_classes,seed=42,params=None):
        self.params = initial_weights(n_classes,seed) if params is None else {k:v.copy() for k,v in params.items()}
        self.n_classes=n_classes

    def forward(self,x,cache=False):
        p=self.params
        patches=sliding_window_view(x,(3,3),axis=(2,3))
        z=np.einsum('nchwkl,fckl->nfhw',patches,p['conv_w'],optimize=True)+p['conv_b'][None,:,None,None]
        a=np.maximum(z,0)
        n=a.shape[0]
        poolwins=a.reshape(n,FILTERS,13,2,13,2).transpose(0,1,2,4,3,5).reshape(n,FILTERS,13,13,4)
        maxidx=poolwins.argmax(axis=-1)
        pooled=poolwins.max(axis=-1)
        features=pooled.reshape(n,-1)
        logits=features@p['dense_w']+p['dense_b']
        if cache:
            return logits,(patches,z,maxidx,features)
        return logits

    def train_batch(self,x,y,lr):
        logits,(patches,z,maxidx,feat)=self.forward(x,cache=True)
        probs=softmax(logits)
        n=len(y)
        loss=-np.log(np.maximum(probs[np.arange(n),y],1e-9)).mean()
        delta=probs.copy()
        delta[np.arange(n),y]-=1.0
        delta/=n
        dw_dense=feat.T@delta
        db_dense=delta.sum(0)
        da=(delta@self.params['dense_w'].T).reshape(n,FILTERS,13,13)
        # Manual backward for 2x2 max pooling; exactly one winning pixel per pool block.
        pool_grad=np.zeros((n,FILTERS,13,13,4),dtype=np.float32)
        np.put_along_axis(pool_grad,maxidx[...,None],da[...,None],axis=-1)
        dz=pool_grad.reshape(n,FILTERS,13,13,2,2).transpose(0,1,2,4,3,5).reshape(n,FILTERS,26,26)
        dz*=z>0
        dw_conv=np.einsum('nchwkl,nfhw->fckl',patches,dz,optimize=True)
        db_conv=dz.sum(axis=(0,2,3))
        for name,gr in [('conv_w',dw_conv),('conv_b',db_conv),('dense_w',dw_dense),('dense_b',db_dense)]:
            self.params[name]-=lr*gr
        return float(loss)

    def predict_proba(self,x,batch_size=128):
        return np.concatenate([softmax(self.forward(np.asarray(x[i:i+batch_size],dtype=np.float32)))
                               for i in range(0,len(x),batch_size)],axis=0)

    def save(self,path):
        np.savez_compressed(path,**self.params)

    @classmethod
    def load(cls,path):
        with np.load(path) as saved:
            p={k:saved[k] for k in saved.files}
        return cls(len(p['dense_b']),params=p)


def epoch_indices(n,epoch,seed):
    return np.random.default_rng(seed+epoch).permutation(n)


def train_scratch(x,y,x_val,y_val,n_classes,epochs=3,batch_size=64,lr=0.05,seed=42):
    model=CNNFromScratch(n_classes,seed)
    history=[]
    for epoch in range(epochs):
        order=epoch_indices(len(y),epoch,seed)
        losses=[]
        for start in range(0,len(order),batch_size):
            ix=order[start:start+batch_size]
            losses.append(model.train_batch(x[ix],y[ix],lr))
        val_pred=model.predict_proba(x_val).argmax(1)
        history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),
                        'val_accuracy':float((val_pred==y_val).mean())})
    return model,history

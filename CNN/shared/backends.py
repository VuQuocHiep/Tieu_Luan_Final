"""Keras 3 (torch backend) and PyTorch implementations of the SAME CNN."""
import os
os.environ['KERAS_BACKEND']='torch'
import numpy as np
import torch
from .cnn_numpy import FILTERS, FEATURE_SIZE, initial_weights, epoch_indices

# Tiny 28x28 Conv -> ReLU -> MaxPool -> Flatten -> Dense; no hidden differences.
class TorchCNN(torch.nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        self.conv=torch.nn.Conv2d(1,FILTERS,kernel_size=3,stride=1,padding=0)
        self.pool=torch.nn.MaxPool2d(2)
        self.fc=torch.nn.Linear(FEATURE_SIZE,n_classes)

    def forward(self,x):
        x=self.pool(torch.relu(self.conv(x)))
        return self.fc(x.flatten(1))


def build_torch(n_classes,seed=42):
    m=TorchCNN(n_classes)
    w=initial_weights(n_classes,seed)
    with torch.no_grad():
        m.conv.weight.copy_(torch.from_numpy(w['conv_w']))
        m.conv.bias.copy_(torch.from_numpy(w['conv_b']))
        m.fc.weight.copy_(torch.from_numpy(w['dense_w'].T.copy()))
        m.fc.bias.copy_(torch.from_numpy(w['dense_b']))
    return m


def train_torch(x,y,x_val,y_val,n_classes,epochs=3,batch_size=64,lr=0.05,seed=42):
    torch.set_num_threads(min(torch.get_num_threads(),4))
    m=build_torch(n_classes,seed)
    optim=torch.optim.SGD(m.parameters(),lr=lr)
    loss_fn=torch.nn.CrossEntropyLoss()
    hist=[]
    m.train()
    for ep in range(epochs):
        order=epoch_indices(len(y),ep,seed)
        losses=[]
        for st in range(0,len(order),batch_size):
            idx=order[st:st+batch_size]
            xb=torch.from_numpy(np.ascontiguousarray(x[idx]))
            yb=torch.from_numpy(np.ascontiguousarray(y[idx])).long()
            optim.zero_grad()
            loss=loss_fn(m(xb),yb)
            loss.backward()
            optim.step()
            losses.append(float(loss.detach()))
        vp=predict_torch(m,x_val).argmax(1)
        hist.append({'epoch':ep+1,'loss':float(np.mean(losses)),'val_accuracy':float((vp==y_val).mean())})
    return m,hist


def predict_torch(m,x,batch_size=128):
    m.eval()
    parts=[]
    with torch.no_grad():
        for i in range(0,len(x),batch_size):
            xb=torch.from_numpy(np.ascontiguousarray(x[i:i+batch_size],dtype=np.float32))
            parts.append(torch.softmax(m(xb),dim=1).cpu().numpy())
    return np.concatenate(parts)


def build_keras(n_classes,seed=42):
    import keras
    m=keras.Sequential([
        keras.layers.Input(shape=(28,28,1)),
        keras.layers.Conv2D(FILTERS,(3,3),padding='valid',activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(n_classes)
    ],name='TinyCNN')
    w=initial_weights(n_classes,seed)
    m.layers[0].set_weights([w['conv_w'].transpose(2,3,1,0),w['conv_b']])
    # Keras flattens NHWC while NumPy/PyTorch flatten NCHW: align weight ordering.
    nchw_to_nhwc=np.arange(FEATURE_SIZE).reshape(FILTERS,13,13).transpose(1,2,0).reshape(-1)
    m.layers[3].set_weights([w['dense_w'][nchw_to_nhwc],w['dense_b']])
    m.compile(optimizer=keras.optimizers.SGD(learning_rate=0.05),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    return m


def train_keras(x,y,x_val,y_val,n_classes,epochs=3,batch_size=64,lr=0.05,seed=42):
    import keras
    m=build_keras(n_classes,seed)
    m.compile(optimizer=keras.optimizers.SGD(learning_rate=lr),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    x_k=x.transpose(0,2,3,1).copy()
    hist=[]
    for ep in range(epochs):
        order=epoch_indices(len(y),ep,seed)
        losses=[]
        for st in range(0,len(order),batch_size):
            ix=order[st:st+batch_size]
            val=m.train_on_batch(x_k[ix],y[ix])
            losses.append(float(val[0] if isinstance(val,(list,tuple)) else val))
        vp=predict_keras(m,x_val).argmax(1)
        hist.append({'epoch':ep+1,'loss':float(np.mean(losses)),
                     'val_accuracy':float((vp==y_val).mean())})
    return m,hist


def predict_keras(m,x,batch_size=128):
    import keras
    predictions=[]
    for i in range(0,len(x),batch_size):
        raw=np.array(m.predict(x[i:i+batch_size].transpose(0,2,3,1).copy(),verbose=0),dtype=np.float32)
        exp=np.exp(raw-raw.max(axis=1,keepdims=True))
        predictions.append(exp/exp.sum(axis=1,keepdims=True))
    return np.concatenate(predictions)

"""Run AFTER python download_kaggle.py and python train_all.py."""
import os,sys,json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('KERAS_BACKEND','torch')
import numpy as np
from shared.datasets import SOURCES,load_split
from shared.inference import ARTIFACTS,load_metadata,load_model,predict_proba

expected={'iris':(150,4,3), 'wine':(1599,11,2), 'breast_cancer':(569,30,2)}
for name,(n,d,c) in expected.items():
    raw,scaler,(train,val,test),splits=load_split(name)
    assert raw.data.shape==(n,d),f'{name}: sai số mẫu / feature'
    assert len(raw.target_names)==c
    sets=[set(z.tolist()) for z in splits]
    assert not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    assert len(sets[0]|sets[1]|sets[2])==n
    meta=load_metadata(name)
    assert len(meta['features'])==d and meta['kaggle_url']==SOURCES[name]['source']
    probs={}
    for method in ['scratch','keras','pytorch']:
        model=load_model(name,method,d,c)
        p=predict_proba(model,method,test[0])
        assert p.shape==(len(test[1]),c)
        np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-6)
        probs[method]=p
    for method in ['keras','pytorch']:
        assert np.array_equal(probs['scratch'].argmax(1),probs[method].argmax(1))
        assert np.max(np.abs(probs['scratch']-probs[method]))<1e-4
    records=json.loads((ARTIFACTS/f'{name}_metrics.json').read_text())
    for record in records:
        acc=np.mean(probs[record['backend']].argmax(1)==test[1])
        assert abs(acc-record['accuracy'])<1e-5
    print(f'PASS {name}: {n} CSV rows, d={d}, 3 backends agree')
print('ALL TESTS PASSED (9 trained models on the CSV supplied)')

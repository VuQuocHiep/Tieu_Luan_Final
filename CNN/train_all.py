"""Train real CNN classifiers with original Kaggle CSV files, never built-in datasets."""
import argparse
from shared.config import CONFIG,METHODS
from shared.train_runner import run_all

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Train Kaggle CNN models')
    p.add_argument('--dataset',choices=list(CONFIG),default=None)
    p.add_argument('--method',choices=list(METHODS),default=None)
    p.add_argument('--max-train',type=int,default=1600)
    p.add_argument('--max-val',type=int,default=450)
    p.add_argument('--max-test',type=int,default=900)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--lr',type=float,default=0.05)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    run_all(a.dataset,a.method,{k.replace('-','_'):v for k,v in vars(a).items() if k not in ('dataset','method')})

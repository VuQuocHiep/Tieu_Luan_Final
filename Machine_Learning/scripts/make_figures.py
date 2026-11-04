"""Biểu đồ phân bố theo tài liệu dataset. Sau khi tải Kaggle, có thể chạy lại từ dữ liệu thật."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parents[1]/'figures'
OUT.mkdir(exist_ok=True)
DATASETS={'iris':(['Setosa','Versicolor','Virginica'],[50,50,50]),
          'wine':(['Dưới 6 điểm','Từ 6 điểm'],[744,855]),
          'breast_cancer':(['Malignant','Benign'],[212,357])}
for key,(labels,counts) in DATASETS.items():
    fig,ax=plt.subplots(figsize=(6.6,2.55))
    b=ax.bar(labels,counts)
    ax.bar_label(b,padding=3,fontsize=10)
    ax.set_ylim(0,max(counts)*1.2)
    ax.set_ylabel('Số mẫu')
    ax.set_title('Phân bố nhãn – '+key.replace('_',' ').title(),fontsize=12)
    ax.spines[['right','top']].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT/f'{key}_class_balance.png',dpi=140)
    plt.close(fig)

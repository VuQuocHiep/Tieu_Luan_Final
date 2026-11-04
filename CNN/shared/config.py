"""Single source of truth: Kaggle dataset and model configuration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
ART = ROOT / 'artifacts'
SEED = 42
METHODS = {'scratch':'NumPy From Scratch','keras':'Keras 3 (Torch backend)','pytorch':'PyTorch'}
CONFIG = {
    'mnist': {
        'title':'Nhận dạng chữ số viết tay', 'short':'MNIST', 'emoji':'🔢',
        'kaggle':'oddrationale/mnist-in-csv',
        'url':'https://www.kaggle.com/datasets/oddrationale/mnist-in-csv',
        'train':'mnist_train.csv','test':'mnist_test.csv','n_train':60000,'n_test':10000,
        'description':'Ảnh xám của chữ số viết tay từ 0 đến 9; mỗi hàng CSV gồm nhãn và 784 mức xám.',
        'classes':['0','1','2','3','4','5','6','7','8','9'],
        'note':'Một số chữ số có nét viết tương tự nhau; phân bố không hoàn toàn bằng nhau.'
    },
    'fashion': {
        'title':'Phân loại trang phục', 'short':'Fashion-MNIST', 'emoji':'👕',
        'kaggle':'zalando-research/fashionmnist',
        'url':'https://www.kaggle.com/datasets/zalando-research/fashionmnist',
        'train':'fashion-mnist_train.csv','test':'fashion-mnist_test.csv','n_train':60000,'n_test':10000,
        'description':'Ảnh xám các loại quần áo và phụ kiện thuộc 10 lớp; thuận tiện so sánh với MNIST.',
        'classes':['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot'],
        'note':'Mỗi lớp trong tập huấn luyện chuẩn có 6.000 mẫu, nhưng các lớp áo có hình dáng giống nhau.'
    },
    'sign_language': {
        'title':'Nhận diện ký hiệu tay ASL', 'short':'Sign Language MNIST', 'emoji':'🤟',
        'kaggle':'datamunge/sign-language-mnist',
        'url':'https://www.kaggle.com/datasets/datamunge/sign-language-mnist',
        'train':'sign_mnist_train.csv','test':'sign_mnist_test.csv','n_train':27455,'n_test':7172,
        'description':'Ảnh xám cử chỉ tay biểu diễn 24 chữ cái ASL; loại trừ J và Z cần chuyển động.',
        'classes':[chr(65+i) for i in range(26) if i not in (9,25)],
        'note':'24 nhãn gốc nằm trong 0–25 nhưng không gồm 9/J và 25/Z; cần ánh xạ nhãn nhất quán.'
    }
}

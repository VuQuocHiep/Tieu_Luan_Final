# Kết quả RNN trên dữ liệu Kaggle

| Dataset        | Method   |   Accuracy |   Macro-F1 |   Precision |   Recall |   N test |
|:---------------|:---------|-----------:|-----------:|------------:|---------:|---------:|
| SMS Spam       | scratch  |     0.9880 |     0.9734 |      0.9890 |   0.9591 |      836 |
| SMS Spam       | keras    |     0.9856 |     0.9678 |      0.9876 |   0.9502 |      836 |
| SMS Spam       | pytorch  |     0.9821 |     0.9618 |      0.9567 |   0.9670 |      836 |
| IMDB Sentiment | scratch  |     0.5175 |     0.5170 |      0.5176 |   0.5175 |     1200 |
| IMDB Sentiment | keras    |     0.5067 |     0.5065 |      0.5067 |   0.5067 |     1200 |
| IMDB Sentiment | pytorch  |     0.5008 |     0.4999 |      0.5008 |   0.5008 |     1200 |
| Emotion        | scratch  |     0.4025 |     0.3631 |      0.3899 |   0.3495 |     1200 |
| Emotion        | keras    |     0.5950 |     0.4238 |      0.4236 |   0.4267 |     1200 |
| Emotion        | pytorch  |     0.4025 |     0.2102 |      0.2475 |   0.2277 |     1200 |

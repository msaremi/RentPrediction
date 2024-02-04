## Colab Usage

Clone the repository and install the necessary packages.

```shell
!git clone https://github.com/msaremi/RentPrediction
!pip install -q opendatasets
!python -m nltk.downloader stopwords
```

Download the [dataset](https://www.kaggle.com/datasets/corrieaar/apartment-rental-offers-in-germany/).

```python
import opendatasets as od
od.download('https://www.kaggle.com/datasets/corrieaar/apartment-rental-offers-in-germany/')
```
Move the dataset to the [data](data) folder.

```shell
%cd RentPrediction/
!cp ../apartment-rental-offers-in-germany/immo_data.csv data/immo_data.csv
```

Start training.

```shell
!python src/train.py -batch-size 96 -epochs=5
```

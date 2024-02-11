# Rent Prediction in Germany

Based on the [apartment rental offers in Germany dataset](https://www.kaggle.com/datasets/corrieaar/apartment-rental-offers-in-germany/).
> The data was scraped from Immoscout24, the biggest real estate platform in Germany. Immoscout24 has listings for both rental properties and homes for sale, however, the data only contains offers for rental properties.

The model has been pretrained on the [German Wikipedia Text Corpus](https://github.com/t-systems-on-site-services-gmbh/german-wikipedia-text-corpus). The pretraining has been done by [dvm1983@huggingface](https://huggingface.co/dvm1983/).
> [The German Wikipedia Text Corpus] is cleaned, preprocessed and sentence splitted. Its purpose is to train NLP embeddings like fastText or ELMo Deep contextualized word representations.

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

## References

The neural network models takes advantage of the following:
* Jiao, X., Yin, Y., Shang, L., Jiang, X., Chen, X., Li, L., ... & Liu, Q. (2019). [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351). _arXiv preprint arXiv:1909.10351._
* Dupont, E., Teh, Y. W., & Doucet, A. (2021). [Generative models as distributions of functions](https://arxiv.org/abs/2102.04776). _arXiv preprint arXiv:2102.04776._
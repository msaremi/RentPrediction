import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, random_split
from torch.utils.data.dataset import T_co
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, MinMaxScaler, Normalizer
from sklearn.impute import SimpleImputer
from collections import namedtuple
from warnings import simplefilter

X = namedtuple('XType', [
    'miscellaneous', 'firing_types', 'heating_type', 'condition',
    'interior_qual', 'type_of_flat', 'address', 'description', 'facilities', 'is_missing'
])

Y = namedtuple('YType', [
    'base_rent', 'total_rent', 'base_rent_per_sqmeter', 'total_rent_per_sqmeter'
])

XY = namedtuple('XY', ['x', 'y'])

TOK = namedtuple('TOK', ['ids', 'mask'])


class RentalDataset(Dataset):
    __bert_model_name = "dvm1983/TinyBERT_General_4L_312D_de"

    __nullable_cols = [
        'serviceCharge', 'telekomUploadSpeed', 'telekomHybridUploadSpeed',
        'yearConstructed', 'yearConstructedRange', 'lastRefurbish', 'floor', 'numberOfFloors',
        'noParkSpaces', 'thermalChar', 'heatingCosts', 'electricityBasePrice',
        'electricityKwhPrice', 'petsAllowed', 'firingTypes', 'heatingType', 'condition', 'interiorQual', 'typeOfFlat'
    ]

    __ourlier_cols = [
        'baseRent', 'totalRent', 'baseRentPerSQMeter', 'totalRentPerSQMeter',
    ]

    __max_length = 256

    def __init__(self, file_path=None, data_frame=None):
        if file_path:
            path = Path(file_path).absolute()
            sanitized_path = path.parent / (path.stem + "_sanitized.pkl")

            if sanitized_path.exists():
                print("Loading cached sanitized data...")
                self.data = pd.read_pickle(sanitized_path)
            else:
                print("Sanitizing data... This might take several minutes.")
                geoloc_path = path.parent / "address_geoloc.csv"
                raw_data = pd.read_csv(path)
                geo_data = pd.read_csv(geoloc_path, index_col='address')
                self.data = self._sanitize(raw_data, geo_data)
                pd.to_pickle(self.data, sanitized_path)
        else:
            self.data = data_frame

    @staticmethod
    def __sanitize_address(raw_data, geo_data):
        def get_address(row):
            house_number = None if pd.isna(row['houseNumber']) else str(row['houseNumber'])
            street = None if pd.isna(row['streetPlain']) else str(row['streetPlain'])
            regio3 = None if pd.isna(row['regio3']) else str(row['regio3'])
            regio2 = None if pd.isna(row['regio2']) else str(row['regio2'])
            regio1 = None if pd.isna(row['regio1']) else str(row['regio1'])
            geo_plz = None if pd.isna(row['geo_plz']) else str(row['geo_plz'])

            return ", ".join(filter(None, [
                house_number, street, regio3, regio2, regio1, geo_plz, "Deutschland"
            ])).replace("_", " ")

        addresses = raw_data.apply(get_address, axis=1)
        geo_dict = addresses.map(geo_data.to_dict('index'))
        data = pd.DataFrame(geo_dict.to_list(), index=geo_dict.index)
        return data

    @classmethod
    def __sanitize_text(cls, raw_data):
        import re
        from nltk.corpus import stopwords
        german_stopwords = set(stopwords.words('german'))

        def clean_text(text):
            text = '' if pd.isna(text) else str(text)
            text = re.sub('[^A-Za-z0-9]+', ' ', text)
            text = re.sub(r"(#[\d\w\.]+)", '', text)
            text = re.sub(r"(@[\d\w\.]+)", '', text)
            text = re.sub(r"http\S+", "", text)
            text = re.sub(r'(@.*?)[\s]', ' ', text)
            text = re.sub(r'&amp;', '&', text)
            text = re.sub(r"(?:\@|https?\://)\S+", "", text)  # remove links and mentions
            text = re.sub(r'\s+', ' ', text).strip()
            text = re.sub(r'[0-9]+', '', text)
            text = text.lower()
            cleaned_text = [word for word in text.split() if len(word) > 3 and word not in german_stopwords]
            return ' '.join(cleaned_text)

        def tokenize_text(row):
            output = tokenizer(row['text'])
            row['ids'] = output['input_ids']
            row['mask'] = output['attention_mask']

        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(cls.__bert_model_name)
        texts = raw_data.map(clean_text).to_list()
        tokenized_text = tokenizer(texts, padding=False, truncation=True, max_length=cls.__max_length)
        data = pd.DataFrame()
        data['input_ids'] = tokenized_text['input_ids']
        data['input_ids'] = data['input_ids'].map(lambda x: np.array(x).astype(np.int32))
        return data

    def _sanitize(self, raw_data, geo_data):
        from datetime import datetime

        def decode_date(str_date):
            date = datetime.strptime(str_date, "%b%y")
            return date.date().year + date.date().month / 12

        def decode_list(row):
            return set() if pd.isna(row) else set(row.split(":"))

        def decode_str(row):
            return "" if pd.isna(row) else row

        simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
        data = pd.DataFrame()

        data[('serviceCharge', '')] = raw_data['serviceCharge']
        data[('telekomUploadSpeed', '')] = raw_data['telekomUploadSpeed']
        data[('telekomHybridUploadSpeed', '')] = raw_data['telekomHybridUploadSpeed']
        data[('yearConstructed', '')] = raw_data['yearConstructed']
        data[('yearConstructedRange', '')] = raw_data['yearConstructedRange']
        data[('lastRefurbish', '')] = raw_data['lastRefurbish']

        data[('pictureCount', '')] = raw_data['picturecount'].astype(np.float)
        data[('livingSpace', '')] = raw_data['livingSpace']
        data[('noRooms', '')] = raw_data['noRooms']
        data[('floor', '')] = raw_data['floor']
        data[('noFloors', '')] = raw_data['numberOfFloors']
        data[('noParkSpaces', '')] = raw_data['noParkSpaces']
        data[('thermalChar', '')] = raw_data['thermalChar']
        data[('heatingCosts', '')] = raw_data['heatingCosts']
        data[('electricityBasePrice', '')] = raw_data['electricityBasePrice']
        data[('electricityKwhPrice', '')] = raw_data['electricityKwhPrice']

        data[('hasLift', '')] = raw_data['lift'].astype(np.float)
        data[('hasCellar', '')] = raw_data['cellar'].astype(np.float)
        data[('hasGarden', '')] = raw_data['garden'].astype(np.float)
        data[('hasBalcony', '')] = raw_data['balcony'].astype(np.float)
        data[('hasKitchen', '')] = raw_data['hasKitchen'].astype(np.float)
        data[('newlyConst', '')] = raw_data['newlyConst'].astype(np.float)
        data[('petsAllowed', '')] = raw_data['petsAllowed'].map({'no': 0, 'yes': 1, 'negotiable': .5})

        mbin = MultiLabelBinarizer()
        values = mbin.fit_transform(raw_data['firingTypes'].map(decode_list)).astype(np.float)
        values[raw_data['firingTypes'].isna(), :] = np.nan
        data[[('firingTypes', x) for x in mbin.classes_]] = values

        lbin = LabelBinarizer()
        values = lbin.fit_transform(raw_data['heatingType'].map(decode_str)).astype(np.float)
        values[raw_data['heatingType'].isna(), :] = np.nan
        data[[('heatingType', x) for x in lbin.classes_[1:]]] = values[:, 1:]

        values = lbin.fit_transform(raw_data['condition'].map(decode_str)).astype(np.float)
        values[raw_data['condition'].isna(), :] = np.nan
        data[[('condition', x) for x in lbin.classes_[1:]]] = values[:, 1:]

        values = lbin.fit_transform(raw_data['interiorQual'].map(decode_str)).astype(np.float)
        values[raw_data['interiorQual'].isna(), :] = np.nan
        data[[('interiorQual', x) for x in lbin.classes_[1:]]] = values[:, 1:]

        values = lbin.fit_transform(raw_data['typeOfFlat'].map(decode_str)).astype(np.float)
        values[raw_data['typeOfFlat'].isna(), :] = np.nan
        data[[('typeOfFlat', x) for x in lbin.classes_[1:]]] = values[:, 1:]

        data[('date', '')] = raw_data['date'].map(decode_date)

        addr_data = self.__sanitize_address(raw_data, geo_data)
        data[('address', 'latitude')] = addr_data['latitude']
        data[('address', 'longitude')] = addr_data['longitude']

        data[('description', '')] = self.__sanitize_text(raw_data['description'])['input_ids']
        data[('facilities', '')] = self.__sanitize_text(raw_data['facilities'])['input_ids']

        for col in type(self).__nullable_cols:
            data[('isMissing', col)] = raw_data[col].isna()

        data[('baseRent', '')] = raw_data['baseRent']
        data[('totalRent', '')] = raw_data['totalRent']
        data[('baseRentPerSQMeter', '')] = raw_data['baseRent'] / raw_data['livingSpace']
        data[('totalRentPerSQMeter', '')] = raw_data['totalRent'] / raw_data['livingSpace']

        data.columns = pd.MultiIndex.from_tuples(data.columns)
        return data.copy()

    def remove_outliers(self, q_bottom=0, q_top=0.95, fit_on=None):
        fit_on = fit_on if fit_on else self
        self.data = self.data[~self.data['baseRent'].isna() & ~self.data['totalRent'].isna()]
        filtered = np.column_stack([
            (self.data[col] > fit_on.data[col].quantile(q_bottom)) &
            (self.data[col] < fit_on.data[col].quantile(q_top))
            for col in self.__ourlier_cols
        ])
        self.data = self.data[filtered.all(axis=1)]

    def split(self, lengths, **kwargs):
        try:
            subsets = random_split(self, lengths, **kwargs)
        except:
            from random_split import random_split as my_random_split
            subsets = my_random_split(self, lengths, **kwargs)

        return list(map(lambda x: RentalDataset(data_frame=self.data.iloc[x.indices]), subsets))

    def impute(self, fit_on=None):
        if len(self.data) == 0:
            return

        fit_on = fit_on if fit_on else self
        date_mean = fit_on.data['date'].mean()
        year_constructed_mean = fit_on.data['yearConstructed'].mean()
        last_refurbish_mean = fit_on.data['lastRefurbish'].mean()
        fit_on.data['date'] -= date_mean
        fit_on.data['yearConstructed'] -= year_constructed_mean
        fit_on.data['lastRefurbish'] -= last_refurbish_mean
        self.data['date'] -= date_mean
        self.data['yearConstructed'] -= year_constructed_mean
        self.data['lastRefurbish'] -= last_refurbish_mean

        imp = SimpleImputer()
        columns = self.data.columns.drop(['description', 'facilities'], level=0)
        imp.fit(fit_on.data[columns].to_numpy())
        self.data[columns] = imp.transform(self.data[columns].to_numpy())

    def standardize(self, range=(0.1, 0.9), fit_on=None):
        fit_on = fit_on if fit_on else self
        scalar = MinMaxScaler(feature_range=range)
        columns = self.data.columns.drop(['description', 'facilities'], level=0)
        scalar.fit(fit_on.data[columns].to_numpy())
        self.data[columns] = scalar.transform(self.data[columns].to_numpy())

    def normalize(self, quantile=0.96, fit_on=None):
        from scipy.special import ndtri
        fit_on = fit_on if fit_on else self
        normalizer = Normalizer()
        columns = self.data.columns.drop(['description', 'facilities'], level=0)
        normalizer.fit(fit_on.data[columns].to_numpy())
        coef = ndtri(quantile)
        self.data[columns] = normalizer.transform(self.data[columns].to_numpy()) / coef

    def __getitem__(self, index) -> T_co:
        def rpad(arr):
            return np.pad(arr, (0, type(self).__max_length-len(arr)), 'constant', constant_values=(0, 0))

        def mask(length):
            arr = np.zeros((type(self).__max_length,), dtype=np.int32)
            arr[:length] = 1
            return arr

        row = self.data.iloc[index]
        return XY(
            x=X(
                miscellaneous=row[[
                    'serviceCharge', 'telekomUploadSpeed', 'telekomHybridUploadSpeed', 'yearConstructed',
                    'yearConstructedRange', 'lastRefurbish', 'pictureCount', 'livingSpace', 'noRooms',
                    'floor', 'noFloors', 'noParkSpaces', 'thermalChar', 'heatingCosts', 'electricityBasePrice',
                    'electricityKwhPrice', 'hasLift', 'hasCellar', 'hasGarden', 'hasBalcony', 'hasKitchen',
                    'newlyConst', 'petsAllowed', 'date'
                ]].to_numpy(dtype=np.float),
                firing_types=row['firingTypes'].to_numpy(dtype=np.float),
                heating_type=row['heatingType'].to_numpy(dtype=np.float),
                condition=row['condition'].to_numpy(dtype=np.float),
                interior_qual=row['interiorQual'].to_numpy(dtype=np.float),
                type_of_flat=row['typeOfFlat'].to_numpy(dtype=np.float),
                address=row['address'].to_numpy(dtype=np.float),
                is_missing=row['isMissing'].to_numpy(dtype=np.float),
                description=TOK(
                    ids=rpad(row['description', '']),
                    mask=mask(len(row['description', ''])),
                ),
                facilities=TOK(
                    ids=rpad(row['facilities', '']),
                    mask=mask(len(row['facilities', ''])),
                ),
            ),
            y=Y(
                base_rent=row['baseRent'].to_numpy(dtype=np.float),
                total_rent=row['totalRent'].to_numpy(dtype=np.float),
                base_rent_per_sqmeter=row['baseRentPerSQMeter'].to_numpy(dtype=np.float),
                total_rent_per_sqmeter=row['totalRentPerSQMeter'].to_numpy(dtype=np.float),
            )
        )

    def __len__(self):
        return len(self.data)


class Collate:
    def __init__(self, device):
        self.device = device

    def fn(self, batch):
        return XY(
            x=X(
                miscellaneous=torch.stack([torch.from_numpy(item.x.miscellaneous) for item in batch]).float().to(self.device),
                firing_types=torch.stack([torch.from_numpy(item.x.firing_types) for item in batch]).float().to(self.device),
                heating_type=torch.stack([torch.from_numpy(item.x.heating_type) for item in batch]).float().to(self.device),
                condition=torch.stack([torch.from_numpy(item.x.condition) for item in batch]).float().to(self.device),
                interior_qual=torch.stack([torch.from_numpy(item.x.interior_qual) for item in batch]).float().to(self.device),
                type_of_flat=torch.stack([torch.from_numpy(item.x.type_of_flat) for item in batch]).float().to(self.device),
                address=torch.stack([torch.from_numpy(item.x.address) for item in batch]).float().to(self.device),
                is_missing=torch.stack([torch.from_numpy(item.x.is_missing) for item in batch]).float().to(self.device),
                description=TOK(
                    ids=torch.stack([torch.from_numpy(item.x.description.ids) for item in batch]).to(self.device),
                    mask=torch.stack([torch.from_numpy(item.x.description.mask) for item in batch]).to(self.device),
                ),
                facilities=TOK(
                    ids=torch.stack([torch.from_numpy(item.x.facilities.ids) for item in batch]).to(self.device),
                    mask=torch.stack([torch.from_numpy(item.x.facilities.mask) for item in batch]).to(self.device),
                ),
            ),
            y=Y(
                base_rent=torch.stack([torch.from_numpy(item.y.base_rent) for item in batch]).float().to(self.device),
                total_rent=torch.stack([torch.from_numpy(item.y.total_rent) for item in batch]).float().to(self.device),
                base_rent_per_sqmeter=torch.stack([torch.from_numpy(item.y.base_rent_per_sqmeter) for item in batch]).float().to(self.device),
                total_rent_per_sqmeter=torch.stack([torch.from_numpy(item.y.total_rent_per_sqmeter) for item in batch]).float().to(self.device),
            )
        )

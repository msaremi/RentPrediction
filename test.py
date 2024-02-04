import torch
import argparse
from pathlib import Path
from data import RentalDataset, Collate
from model import MyModel
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

# dataset = RentalDataset(Path("data", "immo_data.csv"))
# dataset.remove_outliers()
# dataset.impute()
#
# split_generator = torch.Generator().manual_seed(0)
# split = dataset.split([.8, 0.2], generator=split_generator)
# train, test = split[0], split[1]
#
# model: MyModel = torch.load(Path("models", "my_model.pt"))
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
#
# batch_size = 1
# data_loader = DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
# total_steps = len(data_loader)
#
# reals, preds = [], []
#
#

# print('MAE', mean_absolute_error(reals, preds))
# print( 'R2', r2_score(reals, preds))
# print('RMSE', mean_squared_error(reals, preds) ** 0.5)
#
# plt.plot(reals[0:30])
# plt.plot(preds[0:30])
# plt.show()


def predict(model, dset, batch_size=512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    collate = Collate(device)

    data_loader = DataLoader(dset, batch_size=batch_size, shuffle=False, collate_fn=collate.fn)
    total_steps = len(data_loader)
    reals, preds = [], []

    for step, batch in enumerate(data_loader):
        reals.append(batch.y.total_rent)
        preds.append(model(batch.x))

        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}/{total_steps}")

    return torch.vstack(preds).cpu().detach(), torch.vstack(reals).cpu().detach()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", default=Path("data", "immo_data.csv"))
    parser.add_argument("-model", default=Path("models", "my_model.pt"))
    parser.add_argument("-split-size", nargs=3, default=[0.8, 0.0, 0.2], type=float)
    parser.add_argument("-split-seed", default=0, type=int)
    parser.add_argument("-batch-size", default=768, type=int)
    args = parser.parse_args()

    dataset = RentalDataset(args.dataset)
    split_generator = torch.Generator().manual_seed(args.split_seed)
    split = dataset.split(args.split_size, generator=split_generator)
    test_set = split[2]
    test_set.remove_outliers()
    test_set.impute()

    model: MyModel = torch.load(args.model)
    preds, reals = predict(model, test_set, batch_size=args.batch_size)
    print('MAE  ', metrics.mean_absolute_error(reals, preds))
    print('MAPE ', metrics.mean_absolute_percentage_error(reals, preds))
    print('R2   ', metrics.r2_score(reals, preds))
    print('RMSE ', ((metrics.mean_squared_error(reals, preds)) ** 0.5).item())
    print('RRMSE', ((metrics.mean_squared_error(reals, preds) / torch.sum(preds ** 2)) ** 0.5).item())

    plt.plot(reals[0:40])
    plt.plot(preds[0:40])
    plt.legend(('Reals', 'Preds'), loc='upper right')
    plt.title('Prediction of the first 40 rents.')
    plt.grid()
    plt.show()
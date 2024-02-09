import os

import torch
import argparse
from pathlib import Path
from data import RentalDataset, Collate
from model import MyModel
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


def predict(model, dset, batch_size=512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    collate = Collate(device)

    data_loader = DataLoader(dset, batch_size=batch_size, shuffle=False, collate_fn=collate.fn)
    total_steps = len(data_loader)
    reals, preds = [], []

    for step, batch in enumerate(data_loader):
        reals.append(batch.y.total_rent.cpu().detach())
        preds.append(model(batch.x).cpu().detach())

        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}/{total_steps}")

    return torch.vstack(preds), torch.vstack(reals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dir = ".." if str(Path(__file__).parent.absolute()) == os.getcwd() else "."
    parser.add_argument("-dataset", default=Path(dir, "data", "immo_data.csv"))
    parser.add_argument("-model", default=Path(dir, "models", "my_model.pt"))
    parser.add_argument("-split-size", nargs=3, default=[0.85, 0.05, 0.10], type=float)
    parser.add_argument("-split-seed", default=0, type=int)
    parser.add_argument("-batch-size", default=768, type=int)
    args = parser.parse_args()

    dataset = RentalDataset(args.dataset)
    split_generator = torch.Generator().manual_seed(args.split_seed)
    split = dataset.split(args.split_size, generator=split_generator)
    train_set, test_set = split[0], split[2]
    test_set.remove_outliers(fit_on=train_set)
    train_set.remove_outliers()
    test_set.impute(train_set)

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
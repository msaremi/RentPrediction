import os

import torch
import argparse
from pathlib import Path
from data import RentalDataset, Collate
from model import MyModel
from torch.utils.data import DataLoader
from torch import functional as fn


def train(model, train_set, validation_set, batch_size=512, num_epochs=10, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    collate = Collate(device)

    train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate.fn)
    valid_data_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, collate_fn=collate.fn)
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    train_total_steps = len(train_data_loader)
    valid_total_steps = len(valid_data_loader)

    for epoch in range(num_epochs):
        train_total_loss = 0.0
        valid_total_loss = 0.0

        for step, batch in enumerate(train_data_loader):
            optimizer.zero_grad()
            outputs = model(batch.x)
            loss = fn.F.mse_loss(outputs, batch.y.total_rent)

            if step % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} | "
                      f"Train step {step + 1}/{train_total_steps} | "
                      f"Loss: {loss.item():.8f}")

            train_total_loss += loss.item()
            loss.backward()
            optimizer.step()

        for step, batch in enumerate(valid_data_loader):
            optimizer.zero_grad()
            outputs = model(batch.x)
            loss = fn.F.mse_loss(outputs, batch.y.total_rent)
            valid_total_loss += loss.item()

            if step % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} | "
                      f"Validation step {step + 1}/{valid_total_steps} | "
                      f"Loss: {loss.item():.8f}")

        train_avg_loss = train_total_loss / train_total_steps
        valid_avg_loss = valid_total_loss / valid_total_steps if valid_total_steps else float('nan')
        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train loss: {train_avg_loss:.8f} | "
              f"Validation loss: {valid_avg_loss:.8f} *")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dir = ".." if str(Path(__file__).parent.absolute()) == os.getcwd() else "."
    parser.add_argument("-dataset", default=Path(dir, "data", "immo_data.csv"))
    parser.add_argument("-model", default=Path(dir, "models", "my_model.pt"))
    parser.add_argument("-split-size", nargs=3, default=[0.85, 0.05, 0.10], type=float)
    parser.add_argument("-split-seed", default=0, type=int)
    parser.add_argument("-batch-size", default=32, type=int)
    parser.add_argument("-epochs", default=8, type=int)
    parser.add_argument("-lr", default=1e-3, type=float)
    parser.add_argument('-checkpoint', action='store_true')
    parser.add_argument('-use-text', action='store_true')
    args = parser.parse_args()

    dataset = RentalDataset(args.dataset)
    split_generator = torch.Generator().manual_seed(args.split_seed)
    split = dataset.split(args.split_size, generator=split_generator)
    train_set, valid_set = split[0], split[1]
    valid_set.remove_outliers(fit_on=train_set)
    train_set.remove_outliers()
    valid_set.impute(fit_on=train_set)
    train_set.impute()

    model: MyModel = torch.load(args.model) if args.checkpoint else MyModel(args.use_text)
    train(model, train_set, valid_set, batch_size=args.batch_size, num_epochs=args.epochs, lr=args.lr)
    torch.save(model, args.model)

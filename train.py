import torch
import argparse
from pathlib import Path
from data import RentalDataset, Collate
from model import MyModel
from torch.utils.data import DataLoader
from torch import functional as fn


def train(model, dset, batch_size=512, num_epochs=10, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    collate = Collate(device)

    data_loader = DataLoader(dset, batch_size=batch_size, shuffle=False, collate_fn=collate.fn)
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    total_steps = len(data_loader)

    for epoch in range(num_epochs):
        total_loss = 0

        for step, batch in enumerate(data_loader):
            optimizer.zero_grad()
            outputs = model(batch.x)
            loss = fn.F.mse_loss(outputs, batch.y.total_rent)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            if (step + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} | Step {step + 1}/{total_steps} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / total_steps
        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", default=Path("data", "immo_data.csv"))
    parser.add_argument("-model", default=Path("models", "my_model.pt"))
    parser.add_argument("-split-size", nargs=3, default=[0.8, 0.0, 0.2])
    parser.add_argument("-split-seed", default=0)
    parser.add_argument("-batch-size", default=512)
    parser.add_argument("-epochs", default=18)
    parser.add_argument("-lr", default=1e-3)
    parser.add_argument('-checkpoint', action='store_true')
    args = parser.parse_args()

    dataset = RentalDataset(args.dataset)
    split_generator = torch.Generator().manual_seed(args.split_seed)
    split = dataset.split(args.split_size, generator=split_generator)
    train_set = split[0]
    train_set.remove_outliers()
    train_set.impute()

    model: MyModel = torch.load(args.model) if args.checkpoint else MyModel()
    train(model, train_set, batch_size=args.batch_size, num_epochs=args.epochs, lr=args.lr)
    torch.save(model, args.model)
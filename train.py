import torch
import time
import datetime
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from data import TextFileData
from model import TextNet


def train(model, data, n_epochs, learning_rate, device='cpu', print_every=1000):
    """Train the model with data using Adam. Saves the trained model state_dict to file"""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for e in range(n_epochs):
        start = time.time()
        hn = model.init_hidden().to(device)
        running_loss = 0.0
        for i, (inp, tgt) in enumerate(data):
            # Reset the gradients
            optimizer.zero_grad()

            # Forward step
            inp = inp.to(device)
            tgt = tgt.to(device)
            out, hn = model(inp, hn)

            # Homemade cross entropy loss
            loss = -torch.log((1e-5 + torch.sum(tgt * out, dim=2))).sum()
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            if (i+1) % print_every == 0:
                print(
                    f"Epoch {e+1}, Step {i+1}/{len(data)+1}, Avg. loss {running_loss/print_every:.3f}")
                running_loss = 0.0

        end = time.time()
        print("===================================================================")
        print(f'Finished epoch in {end-start:.3f} seconds')
        print("===================================================================")

    # Save to a file with the current date and time
    filename = f'textnet_{datetime.datetime.now().strftime("%y%m%d_%H%M")}.pth'
    save_path = Path('models', filename)
    torch.save(model, save_path)


if __name__ == "__main__":
    text_data = TextFileData('data/goblet_book.txt')
    data = DataLoader(text_data, batch_size=80, pin_memory=True)

    # Model parameters
    n_features = text_data.n_features
    n_layers = 3
    n_hidden = 500

    # Training paramters
    n_epochs = 5
    learning_rate = 2.5e-4

    # Use the GPU if possible
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    model = TextNet(input_size=n_features, output_size=n_features,
                    hidden_size=n_hidden, num_layers=n_layers, hc2fc=256, dropout=0.3).to(device)
    n_params = sum([param.numel()
                    for param in model.parameters() if param.requires_grad])
    print(f"Created model with {n_params} parameters")
    print(model)

    print("\nTraining model")
    train(model, data, n_epochs, learning_rate, device, print_every=10)
    print("Finished training!")
import torch

def train_model(data_loader, model, loss_function, optimizer, device):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train Loss: {avg_loss}")

def test_model(data_loader, model, loss_function, device):

    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Test Loss: {avg_loss}")

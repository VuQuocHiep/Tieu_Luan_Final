"""wine: Logistic Regression using PyTorch autograd & SGD."""
import torch
from shared.datasets import EPOCHS, LEARNING_RATE, SEED

def train(X, y, n_classes):
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    model = torch.nn.Linear(X.shape[1], n_classes)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        logits = model(x_tensor)
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()
    model.eval()
    return model

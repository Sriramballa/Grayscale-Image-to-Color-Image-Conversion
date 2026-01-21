import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ColorizationDataset
from model import ColorizationCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ColorizationDataset("data/train")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = ColorizationCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):
    total_loss = 0
    for gray, color in loader:
        gray, color = gray.to(device), color.to(device)

        optimizer.zero_grad()
        output = model(gray)
        loss = criterion(output, color)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader)}")

torch.save(model.state_dict(), "colorization_model.pth")

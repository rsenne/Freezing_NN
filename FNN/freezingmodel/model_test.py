from model import dataset, net
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import pickle

# Check if GPU is available
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

with open('/home/ryansenne/PycharmProjects/RamiPho/test_data.pkl', 'rb') as f:
    X_test, X_train, y_test, y_train = pickle.load(f)

    y_train.reshape(-1, len(y_train))

features = torch.tensor(X_train, dtype=torch.float32).to(device)
labels = torch.tensor(y_train, dtype=torch.float32).to(device)

data = dataset(features, labels)
data_loader = DataLoader(data, batch_size=10, shuffle=True)

input_size = 45
# device = torch.device('cuda')
nn_model = net(input_size).to(device)

criterion = nn.BCELoss()
optimizer = optim.SGD(nn_model.parameters(), lr=0.005)

num_epochs = 1000

for epoch in range(num_epochs):
    for batch_features, batch_labels in data_loader:
        batch_features = batch_features.unsqueeze(1).to(device)  # Add channel dimension
        batch_labels = batch_labels.unsqueeze(1).to(device)
        # Forward pass
        outputs = nn_model(batch_features)
        loss = criterion(outputs, batch_labels.unsqueeze(1))  # unsqueeze to match the shape

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print("epoch = %4d   loss = %0.4f" % \
                  (epoch, loss))

# Evaluation
nn_model.eval()  # Set the model to evaluation mode
total_correct = 0
total_samples = 0

with torch.no_grad():
    for batch_features, batch_labels in data_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        # Forward pass
        outputs = nn_model(batch_features)
        predicted_labels = torch.round(outputs)

        # Count correct predictions
        total_correct += (predicted_labels == batch_labels.unsqueeze(1)).sum().item()
        total_samples += batch_labels.size(0)

accuracy = total_correct / total_samples
print(f"Accuracy: {accuracy * 100:.2f}%")

import argparse
import torch
import torch.nn as nn
from models import ANN, CNN, RNN, Transformer
from dataset import load_mnist_dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def train_model(model, train_loader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Experiment with different ML models on MNIST dataset.')
    parser.add_argument('--model', type=str, required=True, choices=['ann', 'cnn', 'rnn', 'transformer', 'dt', 'knn'],
                        help='Choose the model to experiment with')
    args = parser.parse_args()

    batch_size = 64
    learning_rate = 0.001
    num_epochs = 5

    train_loader, test_loader = load_mnist_dataset(batch_size)

    if args.model == 'ann':
        model = ANN()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif args.model == 'cnn':
        model = CNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif args.model == 'rnn':
        model = RNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif args.model == 'transformer':
        model = Transformer()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif args.model == 'dt':
        model = DecisionTreeClassifier(random_state=42)
    elif args.model == 'knn':
        model = KNeighborsClassifier(n_neighbors=5)

    if args.model in ['ann', 'cnn', 'rnn', 'transformer']:
        criterion = nn.CrossEntropyLoss()
        train_model(model, train_loader, optimizer, criterion, num_epochs)
        accuracy = test_model(model, test_loader)
        print(f"{args.model.upper()} Accuracy: {accuracy:.4f}")
    elif args.model in ['dt', 'knn']:
        train_data = torch.cat([data for data, _ in train_loader]).view(-1, 28 * 28).numpy()
        train_targets = torch.cat([target for _, target in train_loader]).numpy()
        test_data = torch.cat([data for data, _ in test_loader]).view(-1, 28 * 28).numpy()
        test_targets = torch.cat([target for _, target in test_loader]).numpy()

        if args.model == 'dt':
            model.fit(train_data, train_targets)
        elif args.model == 'knn':
            model.fit(train_data, train_targets)
        
        predictions = model.predict(test_data)
        accuracy = accuracy_score(test_targets, predictions)
        print(f"{args.model.upper()} Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()

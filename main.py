import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

# Datos
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

# Modelos
class MLP(nn.Module):
    def __init__(self, hidden_layers=2, hidden_size=512):
        super(MLP, self).__init__()
        layers = [nn.Flatten()]
        input_size = 28*28
        for _ in range(hidden_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 10))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),  # Output: 16x28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # Output: 16x14x14
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), # Output: 32x14x14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # Output: 32x7x7
            nn.Flatten(),
            nn.Linear(32*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# Entrenamiento
def train_model(model, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        console.insert(tk.END, f"Época [{epoch+1}/{epochs}], Pérdida: {avg_loss:.4f}\n")
        console.see(tk.END)

    # Evaluación
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    console.insert(tk.END, f"Precisión en prueba: {acc:.2f}%\nEntrenamiento terminado\n")
    console.see(tk.END)

# Interfaz
def entrenar_mlp():
    try:
        hidden_layers = int(entry_layers.get())
        hidden_size = int(entry_neurons.get())
    except:
        hidden_layers = 2
        hidden_size = 512
    console.insert(tk.END, f"\nEntrenando MLP con {hidden_layers} capas y {hidden_size} neuronas...\n")
    model = MLP(hidden_layers, hidden_size)
    train_model(model)

def entrenar_cnn():
    console.insert(tk.END, "\nEntrenando CNN...\n")
    model = CNN()
    train_model(model)

root = tk.Tk()
root.title("Entrenamiento de Red Neuronal Fashion MNIST")

frame = ttk.Frame(root, padding=10)
frame.grid(row=0, column=0, sticky="nsew")

ttk.Label(frame, text="Capas ocultas:").grid(row=0, column=0, sticky="w")
entry_layers = ttk.Entry(frame)
entry_layers.insert(0, "2")
entry_layers.grid(row=0, column=1)

ttk.Label(frame, text="Neuronas por capa:").grid(row=1, column=0, sticky="w")
entry_neurons = ttk.Entry(frame)
entry_neurons.insert(0, "512")
entry_neurons.grid(row=1, column=1)

ttk.Button(frame, text="Entrenar MLP", command=entrenar_mlp).grid(row=2, column=0, pady=10)
ttk.Button(frame, text="Entrenar CNN", command=entrenar_cnn).grid(row=2, column=1, pady=10)

console = ScrolledText(frame, width=60, height=20)
console.grid(row=3, column=0, columnspan=2, pady=10)

root.mainloop()

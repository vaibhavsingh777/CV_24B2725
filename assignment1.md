## 🧠 Artificial Neural Network for Image Classification (MNIST)

### 📋 Overview

In this assignment, I implemented an **Artificial Neural Network (ANN)** from scratch using **PyTorch** to classify handwritten digits from the **MNIST dataset**.
The aim was to understand every stage of a deep learning workflow — from data loading to training, evaluation, hyperparameter tuning, and regularization — while interpreting how each step affects model performance.

---

### 🚀 Step 1–3: Data Preparation

I began by importing key libraries like `torch`, `torchvision`, and `matplotlib`.
I normalized the MNIST dataset using `transforms.Normalize((0.5,), (0.5,))` to scale pixel values between -1 and 1, which helps faster convergence.
I used `DataLoader` to handle batch processing efficiently.

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
```

---

### 🧩 Step 4–6: Model Design & Training

I defined a 3-layer ANN with ReLU activations to learn non-linear relationships.
The network used **CrossEntropyLoss** and **Adam optimizer (lr=0.001)**.
During training, I printed loss values every 100 steps to observe convergence.
I plotted the loss curve to verify that it decreased steadily — confirming that learning was occurring properly.

```python
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
```

---

### 📊 Step 8–9: Evaluation

I evaluated the model on test data, achieving **~96.7% accuracy**.
I calculated precision, recall, F1-score, and plotted a confusion matrix to visualize misclassifications.
This analysis helped me see which digits (like 4 and 9) were harder for the network to distinguish.

```python
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')
```

---

### ⚙️ Step 10: Hyperparameter Tuning

To improve performance, I performed a grid search across learning rate, batch size, and epochs.
I realized that smaller learning rates and moderate batch sizes provided smoother convergence and higher accuracy.

---

### 🧩 Step 11: Regularization

Finally, I introduced **Dropout (p=0.5)** and light **L2 regularization** to prevent overfitting.
I observed slightly slower training but more stable validation accuracy — proving regularization improved generalization.

```python
class ANN_Regularized(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)
```

---

### 🏁 Conclusion

Through this assignment, I learned to:

* Preprocess and normalize image data,
* Build and train an ANN from scratch,
* Evaluate performance with metrics and visualization,
* Optimize hyperparameters,
* Apply dropout regularization effectively.

This hands-on process helped me connect the theoretical foundations of deep learning with practical model-building experience.

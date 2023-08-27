import torch
import torch.nn as nn
from fff import FFF

from tqdm import tqdm
import matplotlib.pyplot as plt

# parameters
batch_size = 128
entropy_effect = 0.10
n_epochs = 3

leaf_width = 16
depth = 3
activation = nn.ReLU()
leaf_dropout = 0.0

# Load the MNIST dataset
from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset_training = datasets.MNIST('data', download=True, train=True, transform=transform)
dataloader_training = torch.utils.data.DataLoader(dataset_training, batch_size=batch_size, shuffle=True)
dataset_testing = datasets.MNIST('data', download=True, train=False, transform=transform)
dataloader_testing = torch.utils.data.DataLoader(dataset_testing, batch_size=batch_size, shuffle=True)

# setup the FFF model
model = FFF(input_width=784, hidden_width=leaf_width, output_width=10, depth=depth, activation=activation, dropout=leaf_dropout)

# train the model
training_losses = []
training_accuracies = []
training_entropies = []
testing_losses = []
testing_accuracies = []

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
for epoch in range(n_epochs):
	print("Epoch", epoch)

	# train the model
	model.train()
	for batch_images, batch_labels in tqdm(dataloader_training):
		optimizer.zero_grad()

		output, node_entropies = model(batch_images.view(-1, 784), return_entropies=True)
		node_entropy_mean = node_entropies.mean()
		loss = criterion(output, batch_labels) + entropy_effect * node_entropy_mean
		accuracy = (output.argmax(dim=1) == batch_labels).detach().float().mean()

		training_losses.append(loss.item())
		training_accuracies.append(accuracy.item())
		training_entropies.append(node_entropy_mean.item())

		loss.backward()
		optimizer.step()
	
	# test the model
	model.eval()
	for batch_images, batch_labels in tqdm(dataloader_testing):
		# output = model(batch_images.view(-1, 784))
		output = model.training_forward(batch_images.view(-1, 784), use_hard_decisions=True)
		loss = criterion(output, batch_labels)
		accuracy = (output.argmax(dim=1) == batch_labels).detach().float().mean()

		testing_losses.append(loss.item())
		testing_accuracies.append(accuracy.item())

plt.plot(training_losses)
plt.plot(training_accuracies)
plt.plot(training_entropies)
plt.legend(["training loss", "training accuracy", "mean node entropy"], loc="upper right")
plt.title("The evolution of training loss, accuracy, and mean node entropy")
plt.show()
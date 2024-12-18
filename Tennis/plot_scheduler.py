import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Define the SimpleModel Class
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.f1 = nn.Linear(4, 4)  # Simple linear layer

    def forward(self, x):
        return self.f1(x)

# 2. Initialize the Model, Optimizer, and Scheduler
model = SimpleModel()

# Define the optimizer with an initial learning rate
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Define the Cosine Annealing Scheduler
# T_max is the number of epochs for a complete cycle
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)

# 3. Simulate a Training Loop and Record Learning Rates
num_epochs = 10000
learning_rates = []

for epoch in range(num_epochs):
    # Simulate a training step (forward and backward pass)
    # For visualization purposes, we'll skip actual training
    # In practice, you'd perform loss computation and optimizer.step() here

    # Record the current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)

    # Step the scheduler to update the learning rate
    scheduler.step()

    # (Optional) Print the learning rate at certain epochs
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Learning Rate: {current_lr:.6f}")

# 4. Plot the Learning Rate Schedule
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), learning_rates, label='Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Cosine Annealing Learning Rate Schedule')
plt.legend()
plt.grid(True)
plt.savefig('cosine_annealing_lr_schedule.png')

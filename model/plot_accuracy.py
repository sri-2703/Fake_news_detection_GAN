import matplotlib.pyplot as plt

# Example accuracy values (replace with actual logs if needed)
epochs = [1, 2, 3, 4, 5, 10, 15]
train_acc = [0.60, 0.68, 0.72, 0.76, 0.80, 0.83, 0.86]
val_acc = [0.58, 0.65, 0.70, 0.73, 0.77, 0.79, 0.81]

plt.figure()
plt.plot(epochs, train_acc, label="Training Accuracy")
plt.plot(epochs, val_acc, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Over Epochs")
plt.legend()
plt.grid(True)
plt.show()

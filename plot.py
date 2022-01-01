import matplotlib.pyplot as plt

# configs
configs = [
    value.split(' ') for value in open("configs.txt", "r").read().split("\n")
][:-1]
configs = {metric[0]: float(metric[1]) for metric in configs}

# average batch_times for each epoch
batch_times = list(
    map(float,
        open("batch_times.txt", "r").read().split("\n")[:-1]))
l = len(batch_times)
epochs = int(configs["num_epoch"])
batches_per_epoch = l // epochs
avg_batch_times = [
    round(sum(batch_times[i:batches_per_epoch + i]) / batches_per_epoch, 3)
    for i in range(0, l, batches_per_epoch)
]

# average loss for each epoch
losses = list(map(float, open("loss.txt", "r").read().split("\n")[:-1]))
avg_losses = [
    round(sum(losses[i:batches_per_epoch + i]) / batches_per_epoch, 5)
    for i in range(0, l, batches_per_epoch)
]

# validation accuracy for each epoch
val_acc = list(map(float, open("val_acc.txt", "r").read().split("\n")[:-1]))

# plot avg_batch_times
plt.plot(avg_batch_times)
plt.title("Average Batch Time")
plt.xlabel("Epoch")
plt.ylabel("Time (s)")
plt.show()

# plot avg_losses
plt.plot(avg_losses)
plt.title("Average Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# plot val_acc
plt.plot(val_acc)
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

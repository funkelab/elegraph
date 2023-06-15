from matplotlib.pylab import plt
import csv

iterations = []
loss = []

with open("loss.csv", "r") as f:
    reader = csv.reader(f, delimiter=' ')
    next(reader, None)
    for row in reader:
        iterations.append(int(row[0]))
        loss.append(float(row[1]))

plt.plot(iterations, loss, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.xticks(range(0, iterations[-1] + 1, 2))
plt.legend(loc='best')
plt.show()
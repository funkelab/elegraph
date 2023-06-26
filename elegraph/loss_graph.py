from matplotlib.pylab import plt
import csv

path = "/groups/funke/home/tame/experiments/test-001/train_loss.csv"

def create_loss_graph(path):
    iterations = []
    loss = []

    with open(path, "r") as f:
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
    plt.savefig("train.png")

create_loss_graph(path)
import matplotlib.pyplot as plt

def visualize_matches(matches, filename):
    """
    Creates a 2D plot visualizing actual points to their predicted points.
    Left side is actual. Right side is predicted. 0th point starts at (0,0).
    """
    x = []
    y = []
    for i in range(len(matches)):
        x.append(0)
        y.append(i)
    for j in range(len(matches)):
        x.append(3)
        y.append(matches[j])
    # plot points
    fig = plt.figure()
    plt.scatter(x, y, c='b', marker='o')
    # connect matches
    for k in range(len(matches)):
        plt.plot([x[k], x[k + len(matches)]], [y[k], y[k+ len(matches)]], c='red', label='Line Connecting Points')
    plt.show()
    plt.savefig(filename)

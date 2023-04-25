import matplotlib.pyplot as plt

def plot(scores):
    plt.gcf().clear()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.show(block=False)
    plt.pause(.1)
import numpy as np
import matplotlib.pyplot as plt
import statistics
import seaborn as sns; sns.set()

def projection_simplex_sort(v, z=1):
	# Courtesy: EdwardRaff/projection_simplex.py
    if v.sum() == z and np.alltrue(v >= 0):
        return v
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w



def logistic(w, kappa, C, N):
    """
    Calculates the logistic function of x.
    """
    return 1 / (1 + np.exp(-kappa * (w/N - C)))

def bernoulli(p):
    """
    Samples a Bernoulli random variable with probability of success p.
    """
    if np.random.rand() < p:
        return 1
    else:
        return 0
    


def plot_accuracies(accu_to_plot1, label1, accu_to_plot2, label2, N, runs, path):
    """
    plot two accuracies for comparison
    """
    fig1 = plt.figure(figsize=(6,4))
    clrs = sns.color_palette("husl", 3)

    plot_accuracies1 = np.nan_to_num(accu_to_plot1[:5])
    piters = list(range(plot_accuracies1.shape[1]))
    pmean = list(map(statistics.mean, zip(*plot_accuracies1)))
    pstdv = list(map(statistics.stdev, zip(*plot_accuracies1)))
    ax = sns.lineplot( pmean, color = clrs[0],label= label1)
    ax.fill_between(piters, np.subtract(pmean,pstdv), np.add(pmean,pstdv), alpha=0.3, facecolor=clrs[0])

    plot_accuracies2 = np.nan_to_num(accu_to_plot2[:5])
    piters = list(range(plot_accuracies2.shape[1]))
    pmean = list(map(statistics.mean, zip(*plot_accuracies2)))
    pstdv = list(map(statistics.stdev, zip(*plot_accuracies2)))
    pmean = list(map(statistics.mean, zip(*plot_accuracies2)))
    pstdv = list(map(statistics.stdev, zip(*plot_accuracies2)))
    ax = sns.lineplot( pmean, color = clrs[2],label= label2)
    ax.fill_between(piters, np.subtract(pmean,pstdv), np.add(pmean,pstdv), alpha=0.3, facecolor=clrs[2])

    plt.grid(linewidth=0.6)
    plt.gca().set(xlabel='Iterations',ylabel='L1-accuracy', )
    title='agents = {}, runs = {} '.format(N, runs)  
    plt.title(title)
    plt.show()
    fig1.savefig(path+'avg_runs_n{}.png'.format(N),bbox_inches='tight')
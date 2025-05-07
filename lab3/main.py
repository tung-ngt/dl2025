import matplotlib
matplotlib.use('Agg') 

import math 
from matplotlib import pyplot as plt


def linear(w0: float, w1: float, x2: float, X1: list[float], X2: list[float]) -> list[float]:
    return [w0 + w1*x1 + w2*x2 for x1, x2 in zip(X1, X2)]

def derivative_linear_w0(n: int) -> list[float]:
    return [1.0 for _ in range(n)]

def derivative_linear_w1(X1: list[float]) -> list[float]:
    return X1

def derivative_linear_w2(X2: list[float]) -> list[float]:
    return X2


def sigmoid(X: list[float]) -> list[float]:
    return [ 1/(1 + math.exp(-x)) for x in X]

def derivative_sigmoid(predicts: list[float]) -> list[float]:
    return [ p*(1-p) for p in predicts]


def predict(w0: float, w1: float, x2: float, X1: list[float], X2: list[float]) -> list[float]:
    return sigmoid(linear(w0, w1, w2, X1, X2))


def loss(predicts: list[float], targets: list[float]) -> float:
    return -sum([t*math.log(p) + (1-t)*math.log(1-p) for p, t in zip(predicts, targets)])/len(predicts)

def derivative_loss(predicts: list[float], targets: list[float]) -> list[float]:
    return [t/p - (1-t)/(1-p) for p, t in zip(predicts, targets)]  


def derivative_loss_w0_w1_w2(X1: list[float], X2: list[float], predicts: list[float], targets: list[float]) -> tuple[float, float, float]:
    d_L = derivative_loss(predicts, targets) 
    d_S = derivative_sigmoid(predicts) 
    d_W0 = derivative_linear_w0(len(X1))
    d_W1 = derivative_linear_w1(X1)
    d_W2 = derivative_linear_w2(X2)

    w0 = 0
    w1 = 0
    w2 = 0

    for d_l, d_s, d_w0, d_w1, d_w2 in zip(d_L, d_S, d_W0, d_W1, d_W2):
        w0 += d_l * d_s * d_w0
        w1 += d_l * d_s * d_w1
        w2 += d_l * d_s * d_w2

    w0 /= -len(predicts)
    w1 /= -len(predicts)
    w2 /= -len(predicts)

    return w0, w1, w2


def p_t(predicts: list[float], targets: list[float]) -> list[float]:
    return [t*p + (1-t)*(1-p) for p, t in zip(predicts, targets)]

def derivative_p_t_p(targets: list[float]) -> list[float]:
    return [2*t - 1 for t in targets]


def focal_loss(gamma: float, predicts: list[float], targets: list[float]) -> float:
    p_ts = p_t(predicts, targets)
    return -sum([((1-p_t)**gamma)*math.log(p_t + 5e-5) for p_t in p_ts])/len(predicts)

def derivative_focal_p_t(gamma: float, p_ts: list[float]) -> list[float]:
    return [gamma*((1-p_t)**(gamma - 1))*math.log(p_t + 5e-5) - ((1-p_t)**gamma)/(p_t + 5e-5) for p_t in p_ts]


def derivative_focal_w0_w1_w2(gamma, X1: list[float], X2: list[float], predicts: list[float], targets: list[float]) -> tuple[float, float, float]:
    p_ts = p_t(predicts, targets)
    d_F = derivative_focal_p_t(gamma, p_ts) 
    d_P_T = derivative_p_t_p(targets)
    d_S = derivative_sigmoid(predicts) 
    d_W0 = derivative_linear_w0(len(X1))
    d_W1 = derivative_linear_w1(X1)
    d_W2 = derivative_linear_w2(X2)

    w0 = 0
    w1 = 0
    w2 = 0

    for d_f, d_pt, d_s, d_w0, d_w1, d_w2 in zip(d_F, d_P_T, d_S, d_W0, d_W1, d_W2):
        w0 += d_f * d_pt * d_s * d_w0
        w1 += d_f * d_pt * d_s * d_w1
        w2 += d_f * d_pt * d_s * d_w2

    w0 /= len(predicts)
    w1 /= len(predicts)
    w2 /= len(predicts)

    return w0, w1, w2


def read_data(path: str, head: bool = False) -> tuple[list[float], list[float], list[float]]:
    with open(path, "r") as f:
        lines = f.readlines()
        X1 = []
        X2 = []
        Y = []
        if head:
            lines = lines[1:]
        for line in lines:
            parts = line.split(",")
            X1.append(float(parts[0]))
            X2.append(float(parts[1]))
            Y.append(float(parts[2]))
        return X1, X2, Y


def normalize(X1: list[float], X2: list[float]) -> tuple[list[float], list[float], list[float], float, float, float, float, float, float]:
    meanx1 = sum(X1)/len(X1)
    meanx2 = sum(X2)/len(X2)
    stdx1 = math.sqrt(sum([(x1 - meanx1)**2 for x1 in X1])/(len(X1) + 1))
    stdx2 = math.sqrt(sum([(x2 - meanx2)**2 for x2 in X2])/(len(X2) + 1))

    X1_norm = [(x1 - meanx1)/stdx1 for x1 in X1]
    X2_norm = [(x2 - meanx2)/stdx2 for x2 in X2]
    return X1_norm, X2_norm, meanx1, meanx2, stdx1, stdx2

def draw_line(min_x: float, max_x: float, w0: float, w1: float, w2: float, step: float = 0.01) -> tuple[list[float], list[float]]:
    x1_draw = [min_x]

    x_current = min_x
    while x_current < max_x:
        x_current += step
        x1_draw.append(x_current)


    x2_draw = [-(w0 + w1*x)/w2 for x in x1_draw]
    return x1_draw, x2_draw

def denorm(predicts: list[float], meany: float, stdy: float) -> list[float]:
    return [predict * stdy + meany for predict in predicts]

if __name__ == "__main__":
    X1, X2, Y = read_data("loan2.csv", True)
    # X1_norm = X1
    # X2_norm = X2
    X1_norm, X2_norm, meanx1, meanx2, stdx1, stdx2 = normalize(X1, X2)

    #--------------------------BCE loss----------------------------------
    w0 = 0.01
    w1 = -0.01
    w2 = 0.02

    epochs = 1000
    lr = 0.1

    losses = []
    for i in range(epochs):
        predicts = predict(w0, w1, w2, X1_norm, X2_norm)
        l = loss(predicts, Y)

        d_w0, d_w1, d_w2 = derivative_loss_w0_w1_w2(X1_norm, X2_norm, predicts, Y)

        w0 = w0 - lr*d_w0
        w1 = w1 - lr*d_w1
        w2 = w2 - lr*d_w2

        losses.append(l)

    plt.plot(losses)
    plt.savefig("loss.jpg")
    plt.clf()

    plt.scatter(X1_norm, X2_norm, c=Y, cmap="viridis")
    x1_draw, x2_draw = draw_line(min(X1_norm), max(X1_norm), w0, w1, w2)
    plt.plot(x1_draw, x2_draw)
    plt.savefig("output_norm.jpg")
    plt.clf()

    #--------------------------Focal loss----------------------------------
    w0 = 0.01
    w1 = -0.01
    w2 = 0.02

    epochs = 1000
    lr = 0.1
    gamma = 0.5

    losses = []
    for i in range(epochs):
        predicts = predict(w0, w1, w2, X1_norm, X2_norm)
        l = focal_loss(gamma, predicts, Y)

        d_w0, d_w1, d_w2 = derivative_focal_w0_w1_w2(gamma, X1_norm, X2_norm, predicts, Y)

        w0 = w0 - lr*d_w0
        w1 = w1 - lr*d_w1
        w2 = w2 - lr*d_w2

        losses.append(l)

    plt.plot(losses)
    plt.savefig("loss_focal.jpg")
    plt.clf()

    plt.scatter(X1_norm, X2_norm, c=Y, cmap="viridis")
    x1_draw, x2_draw = draw_line(min(X1_norm), max(X1_norm), w0, w1, w2)
    plt.plot(x1_draw, x2_draw)
    plt.savefig("output_norm_focal.jpg")
    plt.clf()

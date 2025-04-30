import matplotlib
matplotlib.use('Agg') 

from matplotlib import pyplot as plt
import math

def predict(w: float, b: float, X: list[float]) -> list[float]:
    return [w*x + b for x in X]

def loss(predicts: list[float], targets: list[float]) -> float:
    assert len(predicts) == len(targets), "must have same dimension"
    return sum([(predict - target)**2 for predict, target in zip(predicts, targets)])/len(predicts)/2

def dlw(X: list[float], predicts: list[float], targets: list[float]) -> float:
    assert len(X) == len(targets), "must have same dimension"
    assert len(predicts) == len(targets), "must have same dimension"

    return sum([x*(predict - target) for x, predict, target in zip(X, predicts, targets)])/len(X)

def dlb(predicts: list[float], targets: list[float]) -> float:
    assert len(predicts) == len(targets), "must have same dimension"

    return sum([predict - target for predict, target in zip(predicts, targets)])/len(predicts)


def read_data(path: str) -> tuple[list[float], list[float]]:
    with open(path, "r") as f:
        lines = f.readlines()
        X = []
        Y = []
        for line in lines:
            parts = line.split(",")
            X.append(float(parts[0]))
            Y.append(float(parts[1]))

        return X, Y


def normalize(X: list[float], Y: list[float]) -> tuple[list[float], list[float], float, float, float, float]:
    assert len(X) == len(Y), "must have same dimension"
    meanx = sum(X)/len(X)
    meany = sum(X)/len(Y)
    stdx = math.sqrt(sum([(x - meanx)**2 for x in X])/(len(X) + 1))
    stdy = math.sqrt(sum([(y - meany)**2 for y in Y])/(len(Y) + 1))

    X_norm = [(x - meanx)/stdx for x in X]
    Y_norm = [(y - meany)/stdy for y in Y]
    return X_norm, Y_norm, meanx, meany, stdx, stdy

def draw_line(min_x: float, max_x: float, w: float, b: float, step:float = 0.01) -> tuple[list[float], list[float]]:
    x_draw = [min_x]
    x_current = min_x
    while x_current < max_x:
        x_current += step
        x_draw.append(x_current)

    y_draw = predict(w, b, x_draw)
    return x_draw, y_draw

def denorm(predicts: list[float], meany: float, stdy: float) -> list[float]:
    return [predict * stdy + meany for predict in predicts]

if __name__ == "__main__":
    X, Y = read_data("lr.csv")
    X_norm, Y_norm, meanx, meany, stdx, stdy = normalize(X, Y)

    print(f"meanx: {meanx}, stdx: {stdx}")
    print(f"meany: {meany}, stdy: {stdy}")



    # ---------------- NO NORM -------------------
    w = 0.1
    b = -0.1

    epochs = 100
    lr = 0.1
    losses = []
    for i in range(epochs):
        predicts = predict(w, b, X)
        losses.append(loss(predicts, Y))
        dw = dlw(X_norm, predicts, Y)
        db = dlb(predicts, Y)

        w = w - lr*dw
        b = b - lr*db


    plt.plot(losses)
    plt.savefig("loss_train_no_norm_0.1lr.jpg")
    plt.clf()

    plt.scatter(X, Y)
    x_draw, y_draw = draw_line(min(X), max(X), w, b)
    plt.plot(x_draw, y_draw)
    plt.savefig("output_train_no_norm_0.1lr.jpg")
    plt.clf()

    w = 0.1
    b = -0.1

    epochs = 10000
    lr = 0.0005
    losses = []
    for i in range(epochs):
        predicts = predict(w, b, X)
        losses.append(loss(predicts, Y))
        dw = dlw(X_norm, predicts, Y)
        db = dlb(predicts, Y)

        w = w - lr*dw
        b = b - lr*db
    epochs = 100


    plt.plot(losses)
    plt.savefig("loss_train_no_norm_0.0005lr.jpg")
    plt.clf()

    plt.scatter(X, Y)
    x_draw, y_draw = draw_line(min(X), max(X), w, b)
    plt.plot(x_draw, y_draw)
    plt.savefig("output_train_no_norm_0.0005lr.jpg")
    plt.clf()

    # ---------------- WITH NORM -------------------
    w = 0.1
    b = -0.1
    lr = 0.1
    epochs = 100
    losses = []
    for i in range(epochs):
        predicts = predict(w, b, X_norm)
        losses.append(loss(predicts, Y_norm))
        dw = dlw(X_norm, predicts, Y_norm)
        db = dlb(predicts, Y_norm)

        w = w - lr*dw
        b = b - lr*db


    print(f"w: {b}, b: {b}")
    plt.plot(losses)
    plt.savefig("loss.jpg")
    plt.clf()

    plt.scatter(X_norm, Y_norm)
    x_draw, y_draw = draw_line(min(X_norm), max(X_norm), w, b)
    plt.plot(x_draw, y_draw)
    plt.savefig("output_norm.jpg")
    plt.clf()
    

    plt.scatter(X, Y)
    x_draw, y_draw = draw_line(min(X_norm), max(X_norm), w, b)
    x_draw = denorm(x_draw, meanx, stdx)
    y_draw = denorm(y_draw, meany, stdy)
    plt.plot(x_draw, y_draw)
    plt.savefig("output.jpg")
    plt.clf()

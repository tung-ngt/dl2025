import math 

def linear(w0: float, w1: float, x2: float, X1: list[float], X2: list[float]) -> list[float]:
    return [w0 + w1*x1 + w2*x2 for x1, x2 in zip(X1, X2)]

def derivative_linear_w0() -> list[float]:
    return 1

def derivative_linear_w1(X1: list[float]) -> list[float]:
    return X1

def derivative_linear_w2(X2: list[float]) -> list[float]:
    return X2


def sigmoid(X: list[float]) -> list[float]:
    return [ 1/(1 + math.exp(-x)) for x in X]

def derivative_sigmoid(predicts: list[float]) -> list[float]:
    return [ p*(1-p) for p in predicts]


def loss(predicts: list[float], targets: list[float]) -> float:
    return -sum([t*math.log(p) + (1-t)*math.log(1-p) for p, t in zip(predicts, targets)])/len(predicts)

def derivative_loss(predicts: list[float], targets: list[float]) -> list[float]:
    return [t/p - (1-t)/(1-p) for p, t in zip(predicts, targets)]  

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
if __name__ == "__main__":
    X1, X2, Y = 
    for

# Linear regression from scratch
def add(x, y):
    if len(x) != len(y):
        print("Dimention mismatch")
        exit()
    else:
        z = [x[i] + y[i] for i in range(len(x))]
    return z


def sub(x, y):
    if len(x) != len(y):
        print("Dimention mismatch")
        exit()
    else:
        z = [x[i] - y[i] for i in range(len(x))]
    return z


def mul(x, y):
    z = [x[i] * y[i] for i in range(len(x))]
    return z


def div(x, y):
    if len(x) != len(y):
        print("Dimention mismatch")
        exit()
    else:
        z = [x[i] / y[i] for i in range(len(x))]
    return z


def pypow(x, y):
    z = [x[i] ** y for i in range(len(x))]
    return z


def ones1d(n):
    z = []
    for i in range(n):
        z.append(1)
    return z


def expand(val, n):
    z = []
    for i in range(n):
        z.append(val)
    return z


def random1d(strt, end, n):
    z = []
    import random
    for i in range(n):
        val = random.randint(strt, end)
        z.append(val)
    return z


def linear_regressor(x, y, lr, niter):  # x, y both are row vector
    N = len(x)
    # W = random1d(1, 20, N)
    import random
    W = expand(random.randint(1, 20), N)
    b = []
    for i in range(N):
        b.append(0)

    for i in range(niter):
        ypred = add(mul(W, x), b)
        L = mul(div(ones1d(N), expand(N, N)), expand(sum(pypow(sub(y, ypred), 2)), N))
        # print(L)
        dL_dW = mul(div(expand(-2, N), expand(N, N)), expand(sum(mul(sub(y, ypred), x)), N))
        dL_db = mul(div(expand(-2, N), expand(N, N)), expand(sum(sub(y, ypred)), N))
        # update weight
        W = sub(W, mul(expand(lr, N), dL_dW))
        # print(W)
        b = sub(b, mul(expand(lr, N), dL_db))
        print("MSE Loss is:" + str(L[0]))
        print("Iteration:" + str(i))
    # store result
    global weight, bias
    weight = W[0]
    bias = b[0]
    return ypred


def linear_pred(x):  # prediction function
    y = add(mul(expand(weight, len(x)), x), expand(bias, len(x)))
    return y
# scratch code for logistic regression in Micropython
# Numpy-like matrix library from scratch
# Created on 7/6/2022
# Note that, matrix must be two-dimensional
#Rev01: 7/6/22
#Rev02:11/6/22
def zeros(rows, cols):
    """
    Creates a matrix filled with zeros.
        :param rows: the number of rows the matrix should have
        :param cols: the number of columns the matrix should have

        :return: list of lists that form the matrix
    """
    M = []
    while len(M) < rows:
        M.append([])
        while len(M[-1]) < cols:
            M[-1].append(0.0)

    return M


def zeros1d(x):  # 1d zero matrix
    z = [0 for i in range(len(x))]
    return z


def add1d(x, y):
    if len(x) != len(y):
        print("Dimention mismatch")
        exit()
    else:
        z = [x[i] + y[i] for i in range(len(x))]
        return z


def eye(n):
    """
    Creates and returns an identity matrix.
        :param n: the square size of the matrix

        :return: a square identity matrix
    """
    IdM = zeros(n, n)
    for i in range(n):
        IdM[i][i] = 1.0

    return IdM


def print_matrix(M, decimals=3):
    """
    Print a matrix one row at a time
        :param M: The matrix to be printed
    """
    for row in M:
        print([round(x, decimals) + 0 for x in row])


def transpose(M):
    """
    Returns a transpose of a matrix.
        :param M: The matrix to be transposed

        :return: The transpose of the given matrix
    """
    # Section 1: if a 1D array, convert to a 2D array = matrix
    if not isinstance(M[0], list):
        M = [M]

    # Section 2: Get dimensions
    rows = len(M)
    cols = len(M[0])

    # Section 3: MT is zeros matrix with transposed dimensions
    MT = zeros(cols, rows)

    # Section 4: Copy values from M to it's transpose MT
    for i in range(rows):
        for j in range(cols):
            MT[j][i] = M[i][j]

    return MT


def sub(x, y):  # 1d subtraction between two list
    if len(x) != len(y):
        print("Dimension mismatch")
        exit()
    else:
        z = [x[i] - y[i] for i in range(len(x))]
        return z


def dot(A, B):
    """
    Returns the product of the matrix A * B where A is m by n and B is n by 1 matrix
        :param A: The first matrix - ORDER MATTERS!
        :param B: The second matrix

        :return: The product of the two matrices
    """
    # Section 1: Ensure A & B dimensions are correct for multiplication
    rowsA = len(A)
    colsA = len(A[0])
    rowsB = len(B)
    colsB = 1
    if colsA != rowsB:
        raise ArithmeticError('Number of A columns must equal number of B rows.')

    # Section 2: Store matrix multiplication in a new matrix
    C = zeros(rowsA, colsB)
    for i in range(rowsA):
        total = 0
        for ii in range(colsA):
            total += A[i][ii] * B[ii]
            C[i] = total

    return C


##Sigmoid function
def sigmoid(x):
    import math
    z = [1 / (1 + math.exp(-x[kk])) for kk in range(len(x))]
    return z


def binary_loss(ytrue, ypred):
    import math
    z = [-(float(ytrue[i]) * math.log(ypred[i])) - ((1 - float(ytrue[i])) * math.log(1 - ypred[i])) for i in
         range(len(ytrue))]
    cost = (1 / len(ytrue)) * sum(z)
    return cost


def evaluate_pred(w, x, b):
    # print(len(x[0]))
    tmp = zeros1d(x[0])
    for i in range(len(x)):
        tmp = add1d(tmp, [w[i] * x[i][j] for j in range(len(x[0]))])
    yp = sigmoid([tmp[i] + b for i in range(len(tmp))])
    return yp


##Logistic regression function
def logistic_regressor(x, y, lr, epoch):  ##lr:learning rate, niter:max iteration
    import random
    # global w, b
    w = []
    b = 0
    t = []

    for k in range(len(x)):
        ww = random.random()
        w.append(ww)

    # Gradient Descent algorithm
    for niter in range(epoch):  # looping upto no of epoch
        # Main logistic func part:f=W.TX+b
        # for j in range(len(x)):  # for no of feature
        # z = [w[j] * x[j][kk] for kk in range(len(x[0]))]  # wrong
        # z = add1d(z, [w[j] * x[j][kk] for kk in range(len(x[0]))])
        # Manual coding for 4 feature-testing
        # for i in range(len(x)):
        # w0 = [w[0] * x[0][kk] for kk in range(len(x[0]))]
        # w1 = [w[1] * x[1][kk] for kk in range(len(x[0]))]
        # w2 = [w[2] * x[2][kk] for kk in range(len(x[0]))]
        # w3 = [w[3] * x[3][kk] for kk in range(len(x[0]))]
        # z = add1d(w3, add1d(w2, add1d(w0, w1)))

        # add bias term 'b'
        # yp = sigmoid([z[i] + b for i in range(len(z))])

        # yp = sigmoid([z[i] + b for i in range(len(z))])
        # yp = sigmoid(z)  # predicted y
        yp = evaluate_pred(w, x, b)
        # print(yp[:5])
        # print(yp1[:5])
        # Derivative part
        dz = (1 / len(y)) * sum([yp[j] - y[j] for j in range(len(y))])
        # print(x)
        ff = dot(x, sub(yp, y))
        # print(ff)
        dw = [(1 / len(y)) * float(ff[j]) for j in range(len(ff))]
        db = dz

        for ii in range(len(x)):  # update weights
            w[ii] -= (lr * dw[ii])

        # update bias
        b -= (lr * db)
        # calculate loss
        loss = binary_loss(y, yp)
        print("No of epoch: " + str(niter))
        print("Training loss: " + str(loss))

    return w, b, loss, yp


# Prediction using trained model


def mean(x):  # calculate mean of an array or 1D matrix
    z = sum(x) / len(x)
    return z


def stdev(x):  # calculate std deviation of 1D array
    import math
    Xmean = sum(x) / len(x)
    N = len(x)
    tmp = 0
    for i in range(N):
        tmp = tmp + (x[i] - Xmean) ** 2
        z = math.sqrt(tmp / (N - 1))
    return z


def normalize(x):  # x is a 1d array
    nx = [(x[u] - mean(x)) / stdev(x) for u in range(len(x))]
    return nx


def predict_class(ypred):
    ypred_class = [1 if i > 0.5 else 0 for i in ypred]
    return ypred_class


def classification_report(ytrue, ypred):  # print prediction results in terms of metrics and confusion matrix
    tmp = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(ytrue)):
        if ytrue[i] == ypred[i]:  # For accuracy calculation
            tmp += 1
        ##True positive and negative count
        if ytrue[i] == 1 and ypred[i] == 1:  # find true positive
            TP += 1
        if ytrue[i] == 0 and ypred[i] == 0:  # find true negative
            TN += 1
        if ytrue[i] == 0 and ypred[i] == 1:  # find false positive
            FP += 1
        if ytrue[i] == 1 and ypred[i] == 0:  # find false negative
            FN += 1
    accuracy = tmp / len(ytrue)
    conf_matrix = [[TP, FP], [FN, TN]]
    #print(TP, FP, FN, TN)

    print("Accuracy: " + str(accuracy))
    print("Confusion Matrix:")
    print(print_matrix(conf_matrix))
    
# Function to split train and test set
def train_test_split(scaled_x_data,ydata, factor):
    scaled_data = transpose(scaled_x_data)
    N = len(scaled_data)
    print(N)
    n_sample = int(factor * N)
    print(n_sample)
    xtrain_set = transpose(scaled_data[:n_sample])
    xtest_set = transpose(scaled_data[n_sample:])
    ytrain_set = ydata[:n_sample]
    ytest_set = ydata[n_sample:]
    #print(len(xtrain_set))
    return xtrain_set, xtest_set, ytrain_set, ytest_set

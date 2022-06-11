from pylab import logistic_regression as lm
def csvread(file_name):  # function for reading csv file
    f = open(file_name, 'r')
    w = []
    tmp = []
    for each in f:
        w.append(each)
        # print (each)

    # print(w)
    for i in range(len(w)):
        data = w[i].split(",")
        tmp.append(data)
        # print(data)
    file_data = transpose([[float(y) for y in x] for x in tmp])
    # file_data = [[float(y) for y in x] for x in tmp]
    return file_data
	
	####### Test function ###########################
raw_data = csvread('diabetes_pima_test.csv')
scaled_data = [lm.normalize(raw_data[i]) for i in range(len(raw_data[:8]))]
#
xtest = scaled_data
ytest = raw_data[8]
W = [0.28817001,1.04158761,-0.20889697, 0.0914167, -0.1110515, 0.68152683, 0.29103829,0.25853476]
B = -0.83505327
ypred_test = lm.predict_class(lm.evaluate_pred(W,xtest,B))
lm.classification_report(ytest, ypred_test)
##Logistic Regression in Raspberry Pi Pico-RP-2040 using Micropython
##Use small file size to avoid memory error
##Tested using only 100 row in the diabetes.csv file, check file name, make sure
#that 1st row (column header of csv file) is removed (string not supported)
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
    file_data = lm.transpose([[float(y) for y in x] for x in tmp])
    # file_data = [[float(y) for y in x] for x in tmp]
    return file_data
####### Test function ###########################
raw_data = csvread('diabetes_pima_test.csv')

# Normalize data using mean and stdev()
scaled_data = [lm.normalize(raw_data[i]) for i in range(len(raw_data[:8]))]
xtrain, xtest, ytrain, ytest = lm.train_test_split(scaled_data,raw_data[8], 0.7)#70% data used for training, can change this
# Xtrain data
#Train and Build model using Logistic Regression
w, b, loss_train, ypred_train = lm.logistic_regressor(xtrain, ytrain, 0.1, 1000)#1000 epoch, learning rate =0.1
ypred_train = lm.predict_class(ypred_train)
print(ypred_train)
lm.classification_report(ytrain, ypred_train)
#print(len(xtrain[0]))
##test set
#W = [0.37799744,1.06052058,-0.25524181,0.04704248,-0.1428577,0.73400824,0.32438074,0.20997864] 
#B = -0.82373324
ypred_test = lm.predict_class(lm.evaluate_pred(w,xtest,b))
lm.classification_report(ytest, ypred_test)


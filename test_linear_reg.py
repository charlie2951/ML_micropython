from pylab import linear_regression as ln
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [12, 23, 31, 45, 55, 61, 76, 80, 95, 104]

ypred = ln.linear_regressor(x, y, 0.005, 100)
print(ypred)
# predict new value corresponds to input
xnew = [2.5, 7.1, 5.78, 3.54, 8.02]
ynew = ln.linear_pred(xnew)
print(ynew)
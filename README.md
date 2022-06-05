# ML_micropython
MicroPython implementation of basic machine learning Algorithm
1. Linear regression model in micropython tested on RP2040 with Micropython V1.17 in Thonny IDE windows10
 
#Function:
  ypred = ln.linear_regressor(xtrain, ytrain, learning_rate, no_of_iteration)

#Description
Here xtrain, yrain is 1d Python list.
Note: keep learning rate below  0.1 otherwise Gradient-Descent algorithm might not be able to find
global minima and solution will diverge. You can check it by keep an eye on MSE error matric. It should 
reduce with iteration. No of iteration can be varied to minimize the loss i.e. MSE.

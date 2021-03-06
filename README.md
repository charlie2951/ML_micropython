# ML_micropython #
-MicroPython implementation of basic machine learning Algorithm. 
- Note: All code / functions are inside the 'pylab folder'. all test code are placed outside pylab folder. 
- Import the required module using import statement: *from pylab import linear_regression as ln*
## Available tinyML Models ##
1. Linear regression model in micropython tested on RP2040 with Micropython V1.17 in Thonny IDE windows10
2. Logistic Regression (both training and test) but limited performance due to RAM shortage in MCU
3. Logistic regression test only, training done on PC and trained model parameters are exported to Micropython.
 # Linear Regression ##
 *y = w.x +b*
 ## Function:
  - *ypred = ln.linear_regressor(xtrain, ytrain, learning_rate, no_of_iteration)*

### Description
- Here xtrain, yrain is 1d Python list. This function perform linear regression over X data and return predicted Y data. In this version, only single feature (x value is used). With minor modification it can be applied to multiple X values.
- Mean Square Error is used as Loss function. Gradient Descent algorithm is implemented to update W and b value of linear regression model.
- Note: keep learning rate below  0.1 otherwise Gradient-Descent algorithm might not be able to find
global minima and solution will diverge. You can check it by keep an eye on MSE error matric. It should 
reduce with iteration. No of iteration can be varied to minimize the loss i.e. MSE.

## Logistic Regression ##
## Functions:
- *w,b,loss,ypred = lm.logistic_regressior(xtrain,ytrain, learning_rate,epoch)*
- learning_rate typical value 0.1
- epoch : no of iteration depending upon dataset it may vary. See loss during training to decide best no of epoch
- *ypred=lm.evaluate_pred(w,x,b)*
- Used to predict y value for a given test input. W,b are the hyperparameters which comes from training or externally supplied(training on PC)
- This function gives prediction in between 0 to 1 value. To convert it to 0 or 1, use function:
- *lm.predict_class(ypred)*-> This gives either 0 or 1 (two level), a 0.5 threshold is used for decoding


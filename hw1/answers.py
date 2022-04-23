r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:** 

    1. False - we use the test-set to estimate our out-sample error and train-set to estimate our in-sample error
    
    2. False - The train-set is used to fit the parameters of the model therefore if we will take an extreme example of 
               split ratio (%1 train, %99 test) it will not be affective way to train our model.
               
    3. True  - The test-set should evaluate our model therefore if we will use it for cross-validation we will effect the 
               results of the test-set in a way that will defect our estimation of the out-sample error.
               
    4. True  - This is the way to use cross-validation, in each iteration we use the remaining fold as a test-set and in
               the end of the process we use each iteration estimation score to calculate the final average estimation
               error score. 

"""

part1_q2 = r"""
**Your answer:**

This approach is not justified! although adding regularization term is a good way to improve over-fitting on the 
training-set this is not the way to do it. Our friend used the test-set to find the best $\lambda$ but the test-set
should not be used to find hyper parameters because in this way he is doing an over-fitting to the test-set and it
should be an independent set that is used only to check our model.
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**

As seen in plotted graph increasing k improved generalization for unseen data until k=3. This means that when we 
enlarge k (means k>3) we give less influence for the close important neighbors and get affected by unrelated data, 
as we can see it damaged the model generalization. In extreme example when k=length(dataset) all new instances will get 
same label and there will no meaning for the distance and being a neighbour. This happened because that the closest 
neighbor and the farthest neighbor has the same effect.
"""

part2_q2 = r"""
**Your answer:**

1. If we would choose hyperparameter k based on our train-set accuracy we would get best results for k=1 for sure. 
   This would happen because using k=1 will give loss function zero because we compare each instance label only to 
   himself therefore the minimum will be when k=1. The problem with this behavior is that we over fit our data and 
   in this way we are making much less generalized model.

2. If we were using test-set to find the best k it will cause an over-fitting to the test-set although it
   should be an independent set that is used only to check our model. 
   This way we will get good result for test-set accuracy but our generalization would be damaged since we fit the 
   hyperparameter k to the test-set and it will effect our ability to label unseen data.
   Meaning that test-set should be an independent set from the learning and choosing the hyperparameters.
  

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
$\Delta$ represent the margin size between different classes. In SVM every sample that is inside the margin will suffer 
from some penalty. When we are choosing the hyperplane to separate between two classes we can always multiply an 
arbitrary constant to W (including bias) without changing the hyperplane it represents. Therefore changing $\Delta$ will
not change the hyperplane.

For Example:

 $\Delta_1 = 1$
 
 $\Delta_2 = 2$
 
 Two representations to the same hyperplane:
 
 $W^TX - \Delta_1 = 0$
 
 $2 W^TX - \Delta_2 = 0$

meaning we can choose $\Delta$ in an arbitrary way and still get the same separation

"""

part3_q2 = r"""
**Your answer:**

1.  As we saw in weights_as_images() function, the model is trying to understand for each label which pixels are most 
    likely to be white. Pixels in sample image will be given higher weight by the weight tensor that represent their
    label. In this way we can understand some of the mistakes, when there was white pixels in places that the model 
    has learned that its should be black in the label result.
    
2.  The difference between them is that in KNN we take in count just the K nearest neighbors but in SVM the result is
    effected from all the data.
    The similarity is that both of them predict based on geometric decisions.     

"""

part3_q3 = r"""
**Your answer:**

1.  From the way that our loss function graph converge to minimum we can understand that our learning rate was good.
    If we would choose a bigger number we might jump to far and get a graph full of spikes because we had jump over 
    the minimum. Therefore we are not converging to the minimum and just jumping from side to side.
    If we choose a small number it should take a long time to find the minimum and we might get to the epoch limit 
    before reaching to the minimum.
    
2.  We will say that our model is slightly overfitted to the training set.
    We can see in the graph that the train-set accuracy is slightly better than the validation hence we over-fit to 
    the train-set. We can see that the difference between valid and train is not too big therefore the overfitting is 
    not so bad. 

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**

The ideal pattern of the residual plot should be a random scatter of points forming a pretty symmetrically distributed, 
approximately constant minimum width band, around the 0. Residual 0 means that the guess was exactly correct, if it 
occurred on the train-set we might have over-fit.
According to the $R^2$ score that was 0.68 we can see that that result is not so good.
After using CV we can see that the result has improved, $R^2$ score on the train-set was 0.93 and with test-set was 0.86.
We can also see this result in the residual plot that after the CV the points are scatter more closely around the 0.  

"""

part4_q2 = r"""
**Your answer:**

1.  In linear regression the linear is about the parameters of the model and not the data. Therefore even when adding 
    non-linear features to the data our model remains linear model because the coefficients/weights associated with the 
    features are still linear.
    
2.  Yes we can, the features after the non linear transform can be a new data that is non-linear. Although that data has 
    changed,the model remained linear, so we can find linear regression between the transformed data (the data after the 
    non-linear function) to the coefficients of each feature (because the coefficients/weights associated with the 
    features are still linear)

3. In the new model after adding non linear features the decision boundary will be a hyperplane in the space
   created by original and new features. 
   This happens since we can look at new features as more features in higher level space. So we can find
   hyperplane which will separate our data in this higher space.
   But in relation to old parameters new decision boundary will not be linear so it will not be a hyperplane
   (also it have more dimensions then original space).   
"""

part4_q3 = r"""
**Your answer:**

1.  Lambda is the hyper-parameter that control the power of the regularization term. When using logspace we are getting 
    numbers in log scale meaning we get a distribution of more low values and less high values.
    Linescale gives numbers in equal distribution from given range.
    As we learn regularization term tend to be very small positive number therefore we would like to check more low values
    of lambda but also check some high numbers to make sure we don't miss the best one.
    So logscale is used because it fits those demands.
    
2.  We have checked 20 values of lambda, 3 values of degree and the number of folds was 3. So for each combination of 
    those values we fitted the model to the data, Therefore we get:  $ 20 * 3 * 3 = 180 $ times.

"""

# ==============

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
should be an independent set that is used only to check our model. Our friend could use a cross validation for example.
  
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

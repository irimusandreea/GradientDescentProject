# GradientDescentProject
An algorithm that can predict rental prices, for my diploma thesis.

Given a dataset of apartment renting prices, we consider the square footage and the number of rooms. The main task is the creation of a model which helps estimating the rent prices as accurately as possible, based on these two characteristics. To model the problem, the following equation can be used:

y = b0 + b1x1 + b2x2,

where y represents the predicted rent price, x1 the square footage, x2 the number of rooms, while b0, b1 and b2 are the coefficients that we want to learn. This equation can also be put in the form of a function, called the hypothesis function, which would look like this:

h(x) = θ0 + θ1x1 + θ2x2.

Here, the coefficients are θi, when i = 0, 3; they are also known as parameters. However, we will stick with the equation form.

Utilizing the gradient descent method, we will proceed to determine the optimal values of these coefficients. As previpously mentioned, this method focuses on minimizing the cost function, by iteratively updating the coefficients. The cost function gives a suggestion about how well our model fits the training data. Since this example implies linear regression, MSE will serve as the cost function.


Before implementing this algorithm, let us frame the pseudo-code for the gradient
descent function:

1. Initialize b0, b1 and b2 with some arbitrary values.
  
2. Using the current coefficients, determined the predicted values for the training
data.

3. Determine the error between the expected and the actual values.
 
4. Compute the gradients of the cost function with respect to each coefficient.
 
5. Making use of the gradients and the learning rate, update the coefficients.
 
6. The steps 2 through 5 should be repeated until convergence is reached. Then, return b0, b1 and b2 as our ”learned parameters/coefficients”.

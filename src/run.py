import numpy as np

# Model: Basic gravity simulation
def simulate_fall(initial_height, time, gravity=9.81):
    # Equation of motion: h(t) = h0 - 0.5 * g * t^2
    height = initial_height - 0.5 * gravity * time ** 2
    return max(height, 0)  # object can't fall below ground level

# Simulate the height of an object after 2 seconds
initial_height = 100  # meters
time = 2  # seconds
output = simulate_fall(initial_height, time)
print(f"Height after {time} seconds: {output:.2f} meters")



from scipy.optimize import minimize

# Model: Quadratic function to minimize
def objective(x):
    return (x - 3)**2

# Use scipy's minimize function to find the input that minimizes the function
result = minimize(objective, x0=0)  # start with an initial guess of x = 0
print(f"Optimal input: {result.x[0]:.2f}")



from sklearn.linear_model import LinearRegression
import numpy as np

# Given input (X) and output (y) data
X = np.array([[1], [2], [3], [4], [5]])  # input
y = np.array([1.5, 2.5, 3.5, 4.5, 5.5])  # output

# Model: Fit a linear regression
model = LinearRegression()
model.fit(X, y)

# Print the learned model parameters (slope and intercept)
print(f"Model: y = {model.coef_[0]:.2f} * x + {model.intercept_:.2f}")

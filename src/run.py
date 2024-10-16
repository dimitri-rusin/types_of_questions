# simulation
import math

def time_to_fall(height, gravity=9.81):
    """
    Calculate the time it takes for an object to fall from a given height.

    Parameters:
    height (float): Height in meters from which the object is dropped.
    gravity (float): Acceleration due to gravity (m/s^2), default is 9.81 m/s^2.

    Returns:
    float: Time in seconds it takes to reach the ground.
    """
    return math.sqrt((2 * height) / gravity)

# Example usage:
height = 100  # Height in meters
fall_time = time_to_fall(height)
print(f"It takes {fall_time:.2f} seconds to fall from {height} meters.")



# optimisation
from scipy.optimize import minimize

# Define the function to minimize
def objective_function(x):
    return (x - 3)**2 + 4

# Initial guess
initial_guess = 0

# Perform the optimization
result = minimize(objective_function, initial_guess)

# Extract the optimal value
optimal_x = result.x[0]
minimum_value = result.fun

print(f"The minimum value is {minimum_value:.2f} at x = {optimal_x:.2f}")



# modeling
import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data: hours studied vs. exam score
hours_studied = np.array([[1], [2], [3], [4], [5]])  # Input feature
exam_scores = np.array([65, 70, 75, 80, 85])         # Output target

# Create and train the model
model = LinearRegression()
model.fit(hours_studied, exam_scores)

# Coefficient (slope) and intercept
slope = model.coef_[0]
intercept = model.intercept_

print(f"Model equation: score = {slope:.2f} * hours_studied + {intercept:.2f}")

# Predicting the exam score for 6 hours of study
predicted_score = model.predict([[6]])
print(f"  EXAMPLE: Predicted exam score after 6 hours of study: {predicted_score[0]:.2f}")

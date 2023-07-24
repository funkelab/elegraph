import numpy as np

# 𝜅=‖𝐫′(𝑡)×𝐫″(𝑡)‖/‖𝐫′(𝑡)‖^3.
 
def curvature_3d(points, dt=1.0):
    # print(points)
    if len(points) < 3:
        raise ValueError("At least three points are required to calculate curvature.")
    
    # points_array = np.array(points)
    
    # Calculate the first and second derivatives of the position vector using NumPy's gradient
    r_prime = np.gradient(points, dt, axis=0)
    r_double_prime = np.gradient(r_prime, dt, axis=0)
    # print(r_prime)
    # print(r_double_prime)
    
    # Calculate the cross product between the first and second derivatives
    cross_product = np.cross(r_prime, r_double_prime, axis=1)
    
    # Calculate the magnitude of the cross product
    cross_product_magnitude = np.linalg.norm(cross_product, axis=1)
    
    # Calculate the magnitude of the first derivative
    r_prime_magnitude = np.linalg.norm(r_prime, axis=1)
    
    # Calculate the curvature at each point
    curvature = cross_product_magnitude / np.power(r_prime_magnitude, 3)

    return np.mean(curvature)

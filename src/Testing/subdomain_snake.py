import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def calculate_subdomain_coordinates(domain_x, domain_y, num_rows, num_columns):
    domain_x, domain_y

    subdomain_coordinates = []

    for i in range(num_rows):
        if i % 2 == 0:
            for j in range(num_columns):
                x_start = i * (domain_x / num_rows)
                x_end = (i + 1) * (domain_x / num_rows)
                y_start = j * (domain_y / num_columns)
                y_end = (j + 1) * (domain_y / num_columns)
                subdomain_coordinates.append(((x_start, x_end), (y_start, y_end)))
        else:
            for j in range(num_columns-1, -1, -1):
                x_start = i * (domain_x / num_rows)
                x_end = (i + 1) * (domain_x / num_rows)
                y_start = j * (domain_y / num_columns)
                y_end = (j + 1) * (domain_y / num_columns)
                subdomain_coordinates.append(((x_start, x_end), (y_start, y_end)))

    return subdomain_coordinates

# Set the size of the global 2D domain
domain_x = 1
domain_y = 1


# Set the number of rows and columns to split the domain into
num_rows = 5
num_columns = 5

# Call the function to get the subdomain coordinates
subdomain_coordinates = calculate_subdomain_coordinates(domain_x, domain_y, num_rows, num_columns)

# np.random.seed(0)

# # Generate random particle locations at time t1 and t2 within the domain
# num_particles = 10000
# Nc = 30
# particle_locations_t1 = np.random.rand(num_particles, 2) * np.array([domain_x, domain_y])
# particle_locations_t2 = np.random.rand(num_particles, 2) * np.array([domain_x, domain_y])


# optimal_pairings = []

# Print the start and end coordinates of each subdomain in snake order
for i, coordinates in enumerate(subdomain_coordinates):
    x_start, x_end = coordinates[0]
    y_start, y_end = coordinates[1]

    # Filter particles that fall within the current bin for both times
    # particles_in_bin_t1 = [p for p in particle_locations_t1 if x_start <= p[0] < x_end and y_start <= p[1] < y_end]
    # particles_in_bin_t2 = [p for p in particle_locations_t2 if x_start <= p[0] < x_end and y_start <= p[1] < y_end]


    print(f"Subdomain {i + 1}:")
    print(f"X Start: {x_start}, X End: {x_end}")
    print(f"Y Start: {y_start}, Y End: {y_end}")
    # print(f"Particles before: {len(particles_in_bin_t1)},{len(particles_in_bin_t2)}")
    print()

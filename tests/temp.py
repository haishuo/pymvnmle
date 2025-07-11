from pymvnmle._utils import get_starting_values, validate_input_data
from pymvnmle.datasets import apple
import numpy as np

data = validate_input_data(apple)
our_start = get_starting_values(data)

print('Our starting values:', our_start)
print('Should be close to R final solution:')
r_final = np.array([14.72227, 49.33325, -2.247310, -1.563834, 0.2120496])
print('R final params:', r_final)
print('Difference:', our_start - r_final)
print('Relative difference:', (our_start - r_final) / np.abs(r_final))
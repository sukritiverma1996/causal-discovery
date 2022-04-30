import sys
import ges
import numpy as np

data = np.loadtxt(sys.argv[1], delimiter=',')
print(data.shape)

# Run GES with the Gaussian BIC score
estimate, score = ges.fit_bic(data)

print(estimate, score)

with open('est_ges.npy', 'wb') as fp:
    np.save(fp, estimate)

print("Saved in est_ges.npy")


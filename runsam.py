import pickle
import sys
import pandas as pd
from cdt.causality.graph import SAM, SAMv1

data = pd.read_csv(sys.argv[1])
print(data.shape)

obj = SAM()
print("Running SAM v2")
output = obj.predict(data)

with open('est_sam_endtoend.pkl', 'wb') as fp:
    pickle.dump(obj, fp)

obj = SAMv1()
print("Running SAM v1")
output = obj.predict(data)

with open('est_sammod_endtoend.pkl', 'wb') as fp:
    pickle.dump(obj, fp)

print("Done.")


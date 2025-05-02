import sys
import numpy
import pandas
import sklearn
import scikeras
import tensorflow
import joblib

print("\nEnvironment versions for replication:")
print(f"Python     : {sys.version.split()[0]}")
print(f"numpy      : {numpy.__version__}")
print(f"pandas     : {pandas.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"scikeras   : {scikeras.__version__}")
print(f"tensorflow : {tensorflow.__version__}")
print(f"keras (from tensorflow.keras): {tensorflow.keras.__version__}")
print(f"joblib     : {joblib.__version__}")

print("\nTip: For full reproducibility, run 'pip freeze > requirements.txt' and share the file.")

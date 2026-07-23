import datetime as dt

start = dt.datetime.now()
from mfai import *
print(f"mfai import took {dt.datetime.now() - start}")

start = dt.datetime.now()
import mfai.pytorch.callbacks
print(f"mfai import took {dt.datetime.now() - start}")
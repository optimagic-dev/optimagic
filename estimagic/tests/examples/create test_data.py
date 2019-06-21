import os
import pickle

import pandas as pd

with open(os.path.join(os.path.dirname(__file__), "logit_fixtures.pickle"), "rb") as p:
    fix = pickle.load(p)

fix["jacobian"] = pd.DataFrame(columns=fix["params"].index, data=fix["jacobian"])

with open("logit_fixtures.pickle", "wb") as handle:
    pickle.dump(fix, handle)

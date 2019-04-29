import pandas as pd
import numpy as np
import pytest
from estimagic.optimization.reparametrize import reparametrize_to_internal
from estimagic.optimization.process_constraints import process_constraints
from os import path
from pandas.testing import assert_series_equal

dirname = path.dirname(path.abspath(__file__))
params_fixture = pd.read_csv(path.join(dirname, 'fixtures/reparametrize_fixtures.csv'))
params_fixture.set_index(['category', 'subcategory', 'name'], inplace=True)
params_fixture['lower'].fillna(- np.inf, inplace=True)
params_fixture['upper'].fillna(np.inf, inplace=True)


external = []
internal = []

for i in range(3):
    external.append(params_fixture.rename(columns={'value{}'.format(i): 'value'}))
    internal.append(params_fixture.rename(columns={'re{}'.format(i): 'value'}).dropna(subset=['value']))


def constraints(params):
    constr = [
        {'loc': ('c', 'c2'), 'type': 'probability'},
        {'loc': 'd', 'type': 'increasing'},
        {'loc': 'e', 'type': 'covariance'},
        {'loc': 'f', 'type': 'covariance'},
        {'loc': 'g', 'type': 'sum', 'value': 5},
        {'loc': 'h', 'type': 'equality'},
        {'loc': 'i', 'type': 'equality'},
        {'query': 'subcategory == "j1" | subcategory == "i1"', 'type': 'equality'}
    ]
    constr = process_constraints(constr, params)
    return constr


@pytest.mark.parametrize('params, expected_internal', zip(external, internal))
def test_reparametrize_to_internal(params, expected_internal):
    calculated = reparametrize_to_internal(params, constraints(params))

    for col in ['value']:
        assert_series_equal(calculated[col], expected_internal[col])


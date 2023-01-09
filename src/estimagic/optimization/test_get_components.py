import numpy as np

# ===============
# task
# ===============

#The next task will be to write a function that takes an arbitrary internal constraint dictionary, 
# where the fun argument may or may not be multi-dimensional, 
# and returns a list of internal constraint dictionaries for each output dimension. 
# Ignore Jacobians for now!
#For this task please start by writing a unit test with typical internal constraints, 
# where the fun argument is one or multidimensional, 
# and the output that we want. 


def get_components(constraints, n_constr): # WiP

    output = []
    
    for i in range(len(constraints)): # the number of constraint dictionaries
        
        #check for multidimensionality
        if constraints[i]['func'](x).ndim != 0: 
            
            for j in range(len(constraints[i]['func'](x))):
                duplicate = constraints[i].copy()
                duplicate['func'] = constraints[i]['func'](x)[j]
                
                if "lower_bound" in constraints[i]:
                    duplicate["lower_bound"] = constraints[i]["lower_bound"][j]
                    
                if "value" in constraints[i]:
                    duplicate["value"] = constraints[i]["value"][j]
                    
                output.append(duplicate)
        
        if constraints[i]['func'](x).ndim == 0:
            output.append(constraints[i])
    
    if len(output) != n_constr:
        raise ValueError(f'The number of constraints does not match. Expected {n_constr}, got {len(output)}')
    
    return output
            


def constraint_func_1d(x):
    return np.dot(x, x)
 
def constraint_func_2d(x):
    value = np.dot(x, x)
    return np.array([value - 1, 2 - value])


constraints_all = [
    {
        "type": "nonlinear",
        "func": constraint_func_2d,
        "lower_bound": np.arange(2)
    },
    {
        "type": "nonlinear",
        "func": constraint_func_1d,
        "value": 2
    }
 ]

constraints_1d = [
    {
        "type": "nonlinear",
        "func": constraint_func_1d,
        "value": 2
    }
 ]
constraints_2d = [
    {
        "type": "nonlinear",
        "func": constraint_func_2d,
        "lower_bound": np.arange(2)
    }
 ]

x = np.array([0, np.sqrt(2)])

def test_get_components_all():

    expected_output = [
        {
            "type": "nonlinear",
            "func": constraint_func_2d(x)[0],
            "lower_bound": 0
        },
        {
            "type": "nonlinear",
            'func': constraint_func_2d(x)[1],
            "lower_bound": 1
        },
        {
            "type": "nonlinear",
            "func": constraint_func_1d,
            "value": 2
        }
    ]
    assert get_components(constraints_all, n_constr=3) == expected_output


def test_get_components_1d():
    assert get_components(constraints_1d, n_constr=1) == constraints_1d


def test_get_components_2d():
    expected_output = [
        {
            "type": "nonlinear",
            'func': constraint_func_2d(x)[0],
            "lower_bound": 0
        },
         {
            "type": "nonlinear",
            'func': constraint_func_2d(x)[1],
            "lower_bound": 1
        },
    ]
    assert get_components(constraints_2d, n_constr=2) == expected_output

# What if the parameters are not a numpy array but a DataFrame or general pytree?
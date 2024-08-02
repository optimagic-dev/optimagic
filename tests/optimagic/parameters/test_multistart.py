from typing import get_args, get_type_hints

from optimagic.parameters.multistart import MultistartOptions, MultistartOptionsDict


def test_multistart_options_and_dict_have_same_attributes():
    types_from_multistart_options = get_type_hints(MultistartOptions)
    types_from_multistart_options_dict = {
        k: get_args(v)[0] for k, v in get_type_hints(MultistartOptionsDict).items()
    }
    assert types_from_multistart_options == types_from_multistart_options_dict

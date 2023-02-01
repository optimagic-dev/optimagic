from estimagic.optimization.tranquilo.get_component import get_component


def get_size_calculator(size_calculator, noise_options):
    func_dict = {
        "naive": get_naive_n_acceptance_points,
    }

    out = get_component(
        name_or_func=size_calculator,
        func_dict=func_dict,
        default_options={"confidence_level": noise_options.acceptance_confidence},
    )

    return out


def get_naive_n_acceptance_points(
    noise_model, accepted_fval, expected_fval, confidence_level  # noqa: ARG001
):
    return 5

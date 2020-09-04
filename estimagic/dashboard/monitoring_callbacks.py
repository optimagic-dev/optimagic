"""Callbacks for the monitoring app."""
from threading import Thread


def logscale_callback(attr, old, new, button, doc):
    """Switch between log and linear scale.

    Args:
        attr: Required by bokeh.
        old: Old state of the Button.
        new: New state of the Button.
        button (bokeh.models.Toggle)
        doc (bokeh.Document)

    """
    linear_criterion_plot = doc.get_model_by_name("linear_criterion_plot")
    log_criterion_plot = doc.get_model_by_name("log_criterion_plot")
    if new is True:
        _switch_to_log_scale(button, linear_criterion_plot, log_criterion_plot)
    else:
        _switch_to_linear_scale(button, linear_criterion_plot, log_criterion_plot)


def _switch_to_log_scale(button, linear_criterion_plot, log_criterion_plot):
    """Make the linear criterion plot invisible, the log plot visible and adjust button.

    Args:
        button (bokeh.Toggle)
        linear_criterion_plot (bokeh.Figure)
        log_criterion_plot (bokeh.Figure)

    """
    button.button_type = "primary"
    button.label = "Show criterion plot on a linear scale"
    linear_criterion_plot.visible = False
    log_criterion_plot.visible = True


def _switch_to_linear_scale(button, linear_criterion_plot, log_criterion_plot):
    """Make the log criterion plot invisible, the linear plot visible and adjust button.

    Args:
        button (bokeh.Toggle)
        linear_criterion_plot (bokeh.Figure)
        log_criterion_plot (bokeh.Figure)

    """
    button.button_type = "default"
    button.label = "Show criterion plot on a logarithmic scale"
    log_criterion_plot.visible = False
    linear_criterion_plot.visible = True


def activation_callback(
    attr,
    old,
    new,
    session_data,
    criterion_history,
    params_history,
    button,
    flag,
    update_data_partialed,
):
    """Start and reset the convergence plots and their updating.

    Args:
        attr: Required by bokeh.
        old: Old state of the Button.
        new: New state of the Button.

        session_data (dict): This app's entry of infos to be passed between and within
            apps. The keys are:
            - last_retrieved (int): last iteration currently in the ColumnDataSource
            - database_path
        criterion_history (bokeh.ColumnDataSource)
        params_history (bokeh.ColumnDataSource)
        button (bokeh.models.Toggle)
        flag (np.array): Array with one boolean element to start and end updates
            from the database. Careful, the dtype is np.bool.
        update_data_partialed (callable): function to periodically update the
            ColumnDataSources from the database

    """
    callback_dict = session_data["callbacks"]

    if new is True:
        flag[0] = True
        thread = Thread(target=update_data_partialed)
        thread.start()
        callback_dict["data_update"] = thread

        # change the button color
        button.button_type = "success"
        button.label = "Reset Plot"
    else:
        flag[0] = False
        thread = callback_dict.pop("data_update")
        thread.join()

        session_data["last_retrieved"] = 0
        _reset_column_data_sources([criterion_history, params_history])

        # change the button color
        button.button_type = "danger"
        button.label = "Restart Plot"


def _reset_column_data_sources(cds_list):
    """Empty each ColumnDataSource in a list such that it has no entries.

    Args:
        cds_list (list): list of boheh ColumnDataSources
    """
    for cds in cds_list:
        column_names = cds.data.keys()
        cds.data = {name: [] for name in column_names}

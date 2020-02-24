"""Create the master page that is shown when the dashboard is started.

This page shows which optimizations are scheduled, running, finished successfully or
failed. From here the user can monitor any running optimizations.
"""
from bokeh.layouts import Column
from bokeh.layouts import Row
from bokeh.models import ColumnDataSource
from bokeh.models import Panel
from bokeh.models import Tabs
from bokeh.models.widgets import Div

from estimagic.dashboard.utilities import dashboard_link
from estimagic.dashboard.utilities import dashboard_toggle


def master_app(doc, elements_dict):
    """Create the page with the master dashboard.

    Args:
        doc (bokeh.Document): argument required by bokeh
        database_names (list):
            list of the shortened names by which to display the different optimizations
        elements_dict (dict): nested dictionary.
            The outer keys are the shortened paths to the databases.
            The inner keys are "nice_database_name", "full_path", "db_options",
            "start_params" and the table names "criterion_history" and "params_history".
            The inner values are ColumnDataSources with the initially available data
            for the table names.

    """
    sec_to_elements = _create_section_to_elements(elements_dict=elements_dict)
    tabs = _setup_tabs(sec_to_elements=sec_to_elements)
    doc.add_root(tabs)


def _create_section_to_elements(elements_dict):
    """Map to each section the entries that belong to it.

    .. warning::
        Only one section "all" at the moment!

    Args:
        elements_dict (dict): nested dictionary.
            The outer keys are the shortened paths to the databases.
            The inner keys are "nice_database_name", "full_path", "db_options",
            "start_params" and the table names "criterion_history" and "params_history".
            The inner values are ColumnDataSources with the initially available data
            for the table names.

    Returns:
        sec_to_elements (dict): A nested dictionary. The first level keys are the
        sections ("running", "succeeded", "failed", "scheduled"). The second level keys
        are the database names and the second level values a list consisting of the
        link to the dashboard and a button to activate tha dashboard.

    """
    src_dict = {
        "all": _name_to_bokeh_row_elements(elements_dict=elements_dict),
    }
    return src_dict


def _name_to_bokeh_row_elements(elements_dict):
    """Inner part of the sec_to_elements dictionary.

    For each entry that belongs to the section create a clickable link to that
    optimization's monitoring page and a Button to start or pause that
    optimization's monitoring page.

    .. warning::
        The button does not work yet!

    Args:
        elements_dict (dict): nested dictionary.
            The outer keys are the shortened paths to the databases.
            The inner keys are "nice_database_name", "full_path", "db_options",
            "start_params" and the table names "criterion_history" and "params_history".
            The inner values are ColumnDataSources with the initially available data
            for the table names.
    """
    name_to_row = {}
    for database_name, inner_dict in elements_dict.items():
        name_to_row[database_name] = [
            dashboard_link(database_name),
            dashboard_toggle(database_name=database_name),
        ]
    return ColumnDataSource(name_to_row)


def _setup_tabs(sec_to_elements):
    """Create tabs for each section in sec_to_elements with titles.

    Args:
        sec_to_elements (dict): A nested dictionary. The first level keys are the
        sections ("running", "succeeded", "failed", "scheduled"). The second level keys
        are the database names and the second level values a list consisting of the
        link to the dashboard and a button to activate tha dashboard.
    Returns:
        tabs (bokeh.models.Tabs): a tab for every section in sec_to_elements.
    """
    tab_list = []
    for section, name_to_row in sec_to_elements.items():
        text = f"{len(name_to_row.column_names)} optimizations {section}"
        table_rows = [Row(Div(text=text, width=400), name=section + "_how_many")]
        for name, row in name_to_row.data.items():
            table_rows.append(Row(*row, name=name))
        panel = Panel(
            child=Column(*table_rows, name=section + "_col"),
            title=section.capitalize(),
            name=section + "_panel",
        )
        tab_list.append(panel)
    tabs = Tabs(tabs=tab_list, name="tabs")
    return tabs

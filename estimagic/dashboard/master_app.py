"""Create the master page that is shown when the dashboard is started.

This page shows which optimizations are scheduled, running, finished successfully or
failed. From here the user can monitor any running optimizations.
"""
from bokeh.layouts import Column
from bokeh.layouts import Row
from bokeh.models import ColumnDataSource
from bokeh.models import Panel
from bokeh.models import Tabs
from bokeh.models import Toggle
from bokeh.models.widgets import Div


def master_app(doc, database_names, databases):
    """Create the page with the master dashboard.

    Args:
        doc (bokeh.Document): argument required by bokeh
        database_names (list):
            list of the shortened names by which to display the different optimizations
        databases (list): list of paths to the databases.
    """
    sec_to_elements = _create_section_to_elements(
        database_names=database_names, databases=databases
    )

    tabs = _setup_tabs(sec_to_elements=sec_to_elements)
    doc.add_root(tabs)


def _create_section_to_elements(database_names, databases):
    """Map to each section the entries that belong to it.

    Args:
        database_names (list): list of database names
        databases (list): list of databases
    Returns:
        sec_to_elements (dict): A nested dictionary. The first level keys are the
        sections ("running", "succeeded", "failed", "scheduled"). The second level keys
        are the database names and the second level values a list consisting of the
        link to the dashboard and a button to activate tha dashboard.

    """
    src_dict = {
        "all": _name_to_bokeh_row_elements(
            database_names=database_names, databases=databases
        ),
    }
    return src_dict


def _name_to_bokeh_row_elements(database_names, databases):
    """Inner part of the sec_to_elements dictionary.

    For each entry that belongs to the section create a clickable link to that
    optimization's monitoring page and a Button to start or pause that
    optimization's monitoring page.

    .. warning::
        The button does not work yet!

    Args:
        database_names (list): list of database names
        databases (list): list of databases

    """
    name_to_row = {}
    for name in database_names:
        name_to_row[name] = [_dashboard_link(name), _dashboard_toggle(name=name)]
    return ColumnDataSource(name_to_row)


def _dashboard_link(name):
    """Create a link refering to *name*'s monitoring app."""
    div_name = f"link_{name}"
    text = f"<a href=./{name}> {name}!</a>"
    return Div(text=text, name=div_name, width=400)


def _dashboard_toggle(name):
    """Create a Button that changes color when clicked displaying its boolean state.

    .. note::
        This should be a subclass but I did not get that to work.

    """
    toggle = Toggle(
        label=" Activate",
        button_type="danger",
        width=50,
        height=30,
        name=f"toggle_{name}",
    )

    def change_button_color(attr, old, new):
        if new is True:
            toggle.button_type = "success"
            toggle.label = "Deactivate"
        else:
            toggle.button_type = "danger"
            toggle.label = "Activate"

    toggle.on_change("active", change_button_color)
    return toggle


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

from bokeh.layouts import Column
from bokeh.layouts import Row
from bokeh.models import ColumnDataSource
from bokeh.models import Panel
from bokeh.models import Tabs
from bokeh.models import Toggle
from bokeh.models.widgets import Div


def master_app(doc, database_names, databases):
    overview = _create_section_to_entries(
        database_names=database_names, databases=databases
    )

    tabs = _setup_tabs(overview=overview)
    doc.add_root(tabs)


def _create_section_to_entries(database_names, databases):
    """Map to each section the entries that belong to it.

    Args:
        database_names (list): list of database names
        databases (list): list of databases
    Returns:
        overview (dict): A nested dictionary. The first level keys are the sections
        ("running", "succeeded", "failed", "scheduled"). The second level keys are the
        database names and the second level values a list consisting of the link to the
        dashboard and a button to activate tha dashboard.

    """
    src_dict = {
        "all": _name_to_bokeh_row_elements(
            database_names=database_names, databases=databases
        ),
    }
    return src_dict


def _name_to_bokeh_row_elements(database_names, databases):
    name_to_row = {}
    for name, db in zip(database_names, databases):
        name_to_row[name] = [_dashboard_link(name), DashboardToggle(name=name)]
    return ColumnDataSource(name_to_row)


def _dashboard_link(name):
    div_name = f"link_{name}"
    text = f"<a href=./{name}> {name}!</a>"
    return Div(text=text, name=div_name, width=400)


# this should be a subclass but I did not get that to work (in reasonable time)
def DashboardToggle(name):
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


def _setup_tabs(overview):
    """Create tabs for each section in overview with titles.

    Args:
        overview (dict): A nested dictionary. The first level keys are the sections
        ("running", "succeeded", "failed", "scheduled"). The second level keys are the
        database names and the second level values a list consisting of the link to the
        dashboard and a button to activate tha dashboard.
    Returns:
        tabs (bokeh.models.Tabs): a tab for every section in overview.
    """
    tab_list = []
    for section, name_to_row in overview.items():
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

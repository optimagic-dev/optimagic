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


def master_app(doc, database_name_to_path, session_data):
    """Create the page with the master dashboard.

    Args:
        doc (bokeh.Document):
            document where the overview over the optimizations will be displayed
            by their current stage.
        database_name_to_path (dict):
            mapping from the short, unique names to the full paths to the databases.
        session_data (dict):
            infos to be passed between and within apps.
            Keys of the monitoring apps' entries are:
            - last_retrieved (int): last iteration currently in the ColumnDataSource
            - database_path

    """
    sec_to_elements = _create_section_to_elements(
        doc=doc, database_name_to_path=database_name_to_path
    )
    tabs = _setup_tabs(sec_to_elements=sec_to_elements)
    doc.add_root(tabs)


def _create_section_to_elements(doc, database_name_to_path):
    """Create a dictionary that maps sections to corresponding entries.

    The keys are the sections. They will be "running", "succeeded", "failed"
    and "scheduled" later on. The values are ColumnDataSources.
    These are basically dictionaries mapping the database name to a list of
    bokeh elements that make up the entry in the overview table.
    This consists just of a link to the dashboard at the moment.
    They are represented as ColumnDataSources to be use callbacks on them.

    Args:
        doc (bokeh Document)
        database_name_to_path (dict):
            mapping from the short, unique names to the full paths to the databases.

    Returns:
        sec_to_elements (dict):

    """
    src_dict = {
        "all": _map_dabase_name_to_bokeh_row_elements(
            doc=doc, database_name_to_path=database_name_to_path
        ),
    }
    return src_dict


def _map_dabase_name_to_bokeh_row_elements(doc, database_name_to_path):
    """Inner part of the sec_to_elements dictionary.

    For each entry that belongs to the section create a clickable link to that
    optimization's monitoring page.

    Args:
        doc (bokeh Document)
        database_name_to_path (dict): mapping from the short, unique names to the full
            paths to the databases.

    """
    name_to_row = {}
    for database_name in database_name_to_path:
        name_to_row[database_name] = [_create_dashboard_link(database_name)]
    return ColumnDataSource(name_to_row)


def _create_dashboard_link(name):
    """Create a link refering to *name*'s monitoring app.

    Args:
        name (str): Uniqe name of the database.

    Returns:
        div (bokeh.models.widgets.Div): Link to the database's monitoring page.
    """
    div_name = f"link_{name}"
    open_in_new_tab = r'target="_blank"'
    text = f"<a href=./{name} {open_in_new_tab}> {name}!</a>"
    div = Div(text=text, name=div_name, width=400)
    return div


def _setup_tabs(sec_to_elements):
    """Create tabs for each section in sec_to_elements with titles.

    Args:
        sec_to_elements (dict): A nested dictionary. The first level keys will be the
        sections ("running", "succeeded", "failed", "scheduled"). The second level keys
        are the database names and the second level values a list consisting of the
        link to the dashboard and a button to activate tha dashboard.

    Returns:
        tabs (bokeh.models.Tabs): a tab for every section in sec_to_elements.

    """
    tab_list = []
    for section, name_to_row in sec_to_elements.items():
        text = f"{len(name_to_row.column_names)} optimizations {section}"
        table_rows = [Row(Div(text=text, width=400), name=f"{section}_how_many")]
        for name, row in name_to_row.data.items():
            table_rows.append(Row(*row, name=name))
        panel = Panel(
            child=Column(*table_rows, name=f"{section}_col"),
            title=section.capitalize(),
            name=f"{section}_panel",
        )
        tab_list.append(panel)
    tabs = Tabs(tabs=tab_list, name="tabs")
    return tabs

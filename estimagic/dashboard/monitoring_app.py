from bokeh.models.widgets import Button


def monitoring_app(doc, database):
    button = Button(label="Monitor")
    doc.add_root(button)

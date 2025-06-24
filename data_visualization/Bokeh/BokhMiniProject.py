from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, Select, CustomJS
from bokeh.layouts import column
from bokeh.sampledata.iris import flowers


output_file("interactive_iris_scatter.html")

source = ColumnDataSource(data=flowers)

p = figure(title="Iris Dataset: Petal Width vs. Petal Length",
           x_axis_label='Petal Length (cm)',
           y_axis_label='Petal Width (cm)',
           tools="pan,wheel_zoom,box_zoom,reset")

colors = {'setosa': 'green', 'versicolor': 'blue', 'virginica': 'red'}
flowers['color'] = [colors[sp] for sp in flowers['species']]
source.data['color'] = flowers['color']

scatter = p.circle('petal_length', 'petal_width',
                   color='color',
                   size=8,
                   source=source,
                   legend_field='species',
                   fill_alpha=0.6)

hover = HoverTool(tooltips=[
    ("Species", "@species"),
    ("Petal Length", "@petal_length"),
    ("Petal Width", "@petal_width"),
])
p.add_tools(hover)

p.legend.title = "Species"
p.legend.location = "top_left"

species_select = Select(title="Filter by Species:", value="All",
                        options=["All"] + sorted(flowers['species'].unique().tolist()))

callback = CustomJS(args=dict(source=source, original=flowers, select=species_select), code="""
    const data = source.data;
    const all_data = original;
    const species = select.value;

    const filtered = {
        petal_length: [],
        petal_width: [],
        species: [],
        color: []
    };

    for (let i = 0; i < all_data['petal_length'].length; i++) {
        if (species === 'All' || all_data['species'][i] === species) {
            filtered['petal_length'].push(all_data['petal_length'][i]);
            filtered['petal_width'].push(all_data['petal_width'][i]);
            filtered['species'].push(all_data['species'][i]);
            filtered['color'].push(all_data['color'][i]);
        }
    }

    data['petal_length'] = filtered['petal_length'];
    data['petal_width'] = filtered['petal_width'];
    data['species'] = filtered['species'];
    data['color'] = filtered['color'];
    source.change.emit();
""")

species_select.js_on_change('value', callback)

layout = column(species_select, p)

show(layout)

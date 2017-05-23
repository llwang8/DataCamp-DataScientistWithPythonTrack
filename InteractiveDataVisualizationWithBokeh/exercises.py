# DataCamp
# Interactive Data Visualization with Bokeh

# 1. Basic plotting with Bokeh

# Plotting with glyphs

# Import figure from bokeh.plotting
from bokeh.plotting import figure
# Import output_file and show from bokeh.io
from bokeh.io import output_file, show
# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')
# Add a circle glyph to the figure p
p.circle(fertility, female_literacy)
# Call the output_file() function and specify the name of the file
output_file('fert_lit.html')
# Display the plot
show(p)


# Create the figure: p
p = figure(x_axis_label='fertility', y_axis_label='female_literacy (% population)')
# Add a circle glyph to the figure p
p.circle(fertility_latinamerica, female_literacy_latinamerica)
# Add an x glyph to the figure p
p.x(fertility_africa, female_literacy_africa)
# Specify the name of the file
output_file('fert_lit_separate.html')
# Display the plot
show(p)



# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')
# Add a blue circle glyph to the figure p
p.circle(fertility_latinamerica, female_literacy_latinamerica, color='blue', size=10, alpha=0.8)
# Add a red circle glyph to the figure p
p.circle(fertility_africa, female_literacy_africa, color='red', size=10, alpha=0.8)
# Specify the name of the file
output_file('fert_lit_separate_colors.html')
# Display the plot
show(p)



# Import figure from bokeh.plotting
from bokeh.plotting import figure
# Create a figure with x_axis_type="datetime": p
p = figure( x_axis_type = 'datetime', x_axis_label='Date', y_axis_label='US Dollars')
# Plot date along the x axis and price along the y axis
p.line(date, price)
# Specify the name of the output file and show the result
output_file('line.html')
show(p)

# Import figure from bokeh.plotting
from bokeh.plotting import figure
# Create a figure with x_axis_type='datetime': p
p = figure(x_axis_type='datetime', x_axis_label='Date', y_axis_label='US Dollars')
# Plot date along the x-axis and price along the y-axis
p.line(date, price)
# With date on the x-axis and price on the y-axis, add a white circle glyph of size 4
p.circle(date, price, fill_color='white', size=4)
# Specify the name of the output file and show the result
output_file('line.html')
show(p)



# Create a list of az_lons, co_lons, nm_lons and ut_lons: x
x = [az_lons, co_lons, nm_lons, ut_lons]
# Create a list of az_lats, co_lats, nm_lats and ut_lats: y
y = [az_lats, co_lats, nm_lats, ut_lats]
# Add patches to figure p with line_color=white for x and y
p.patches(x, y, line_color='white')
# Specify the name of the output file and show the result
output_file('four_corners.html')
show(p)


# Data format

# Import numpy as np
import numpy as np
# Create array using np.linspace: x
x = np.linspace(0, 5, 100)
# Create array using np.cos: y
y = np.cos(x)
# Add circles at x and y
p.circle(x, y)
# Specify the name of the output file and show the result
output_file('numpy.html')
show(p)


# Import pandas as pd
import pandas as pd
# Read in the CSV file: df
df = pd.read_csv('auto.csv')
# Import figure from bokeh.plotting
from bokeh.plotting import figure
# Create the figure: p
p = figure(x_axis_label='HP', y_axis_label='MPG')
# Plot mpg vs hp by color
p.circle(df['hp'], df['mpg'], color=df['color'], size=10)
# Specify the name of the output file and show the result
output_file('auto-df.html')
show(p)


# Import the ColumnDataSource class from bokeh.plotting
from bokeh.plotting import ColumnDataSource
# Create a ColumnDataSource from df: source
source = ColumnDataSource(df)
# Add circle glyphs to the figure p
p.circle(x='Year', y='Time', source=source, color='color', size=8)
# Specify the name of the output file and show the result
output_file('sprint.html')
show(p)


# Create a figure with the "box_select" tool: p
p = figure(tools='box_select', x_axis_label='Year', y_axis_label='Time')
# Add circle glyphs to the figure p with the selected and non-selected properties
p.circle(x='Year', y='Time', source=source, selection_color='red', nonselection_alpha=0.1)
# Specify the name of the output file and show the result
output_file('selection_glyph.html')
show(p)


# import the HoverTool
from bokeh.models import HoverTool
# Add circle glyphs to figure p
p.circle(x, y, size=10,
         fill_color='grey', alpha=0.1, line_color=None,
         hover_fill_color='firebrick', hover_alpha=0.5,
         hover_line_color='white')
# Create a HoverTool: hover
hover = HoverTool(tooltips=None, mode='vline')
# Add the hover tool to the figure p
p.add_tools(hover)
# Specify the name of the output file and show the result
output_file('hover_glyph.html')
show(p)


#Import CategoricalColorMapper from bokeh.models
from bokeh.models import CategoricalColorMapper
# Convert df to a ColumnDataSource: source
source = ColumnDataSource(df)
# Make a CategoricalColorMapper object: color_mapper
color_mapper = CategoricalColorMapper(factors=['Europe', 'Asia', 'US'],
                                      palette=['red', 'green', 'blue'])
# Add a circle glyph to the figure p
p.circle('weight', 'mpg', source=source,
            color=dict(field='origin', transform=color_mapper),
            legend='origin')
# Specify the name of the output file and show the result
output_file('colormap.html')
show(p)



# 2. Layouts, Interaction, and Annotation

# Creating rows of plots
# Import row from bokeh.layouts
from bokeh.layouts import row
# Create the first figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')
# Add a circle glyph to p1
p1.circle('fertility', 'female_literacy', source=source)
# Create the second figure: p2
p2 = figure(x_axis_label='population', y_axis_label='female_literacy (% population)')
# Add a circle glyph to p2
p2.circle('population', 'female_literacy', source=source)
# Put p1 and p2 into a horizontal row: layout
layout = row(p1, p2)
# Specify the name of the output_file and show the result
output_file('fert_row.html')
show(layout)


# Creating columns of plots
# Import column from the bokeh.layouts module
from bokeh.layouts import column
# Create a blank figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')
# Add circle scatter to the figure p1
p1.circle('fertility', 'female_literacy', source=source)
# Create a new blank figure: p2
p2 = figure(x_axis_label='population', y_axis_label='female_literacy (% population)')
# Add circle scatter to the figure p2
p2.circle('population', 'female_literacy', source=source)
# Put plots p1 and p2 in a column: layout
layout = column(p1, p2)
# Specify the name of the output_file and show the result
output_file('fert_column.html')
show(layout)


# Nesting rows and columns
# Import column and row from bokeh.layouts
from bokeh.layouts import row, column
# Make a column layout that will be used as the second row: row2
row2 = column([mpg_hp, mpg_weight], sizing_mode='scale_width')
# Make a row layout that includes the above column layout: layout
layout = row([avg_mpg, row2], sizing_mode='scale_width')
# Specify the name of the output_file and show the result
output_file('layout_custom.html')
show(layout)


# Creating gridded layouts

# Import gridplot from bokeh.layouts
from bokeh.layouts import gridplot
# Create a list containing plots p1 and p2: row1
row1 = [p1, p2]
# Create a list containing plots p3 and p4: row2
row2 = [p3, p4]
# Create a gridplot using row1 and row2: layout
layout = gridplot([row1, row2])
# Specify the name of the output_file and show the result
output_file('grid.html')
show(layout)


# Starting tabbed layouts
# Import Panel from bokeh.models.widgets
from bokeh.models.widgets import Panel
# Create tab1 from plot p1: tab1
tab1 = Panel(child=p1, title='Latin America')
# Create tab2 from plot p2: tab2
tab2 = Panel(child=p2, title='Africa')
# Create tab3 from plot p3: tab3
tab3 = Panel(child=p3, title='Asia')
# Create tab4 from plot p4: tab4
tab4 = Panel(child=p4, title='Europe')


# Import Tabs from bokeh.models.widgets
from bokeh.models.widgets import Tabs
# Create a Tabs layout: layout
layout = Tabs(tabs=[tab1, tab2, tab3, tab4])
# Specify the name of the output_file and show the result
output_file('tabs.html')
show(layout)


# Linked axes

# Link the x_range of p2 to p1: p2.x_range
p2.x_range = p1.x_range
# Link the y_range of p2 to p1: p2.y_range
p2.y_range = p1.y_range
# Link the x_range of p3 to p1: p3.x_range
p3.x_range = p1.x_range
# Link the y_range of p4 to p1: p4.y_range
p4.y_range = p1.y_range
# Specify the name of the output_file and show the result
output_file('linked_range.html')
show(layout)


# Linked Brushing

# Add a circle glyph to p1
p1.circle('fertility', 'female literacy', source=source)
# Create the second figure: p2
p2 = figure(x_axis_label='fertility (children per woman)', y_axis_label='population (millions)',
            source=source)
# Add a circle glyph to p2
p2.circle('fertility', 'population', source=source)
# Create row layout of figures p1 and p2: layout
layout = row(p1, p2)


# Create ColumnDataSource: source
source = ColumnDataSource(data)

# Create the first figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female literacy (% population)', tools='box_select,lasso_select')
# Add a circle glyph to p1
p1.circle('fertility', 'female literacy', source=source)
# Create the second figure: p2
p2 = figure(x_axis_label='fertility (children per woman)', y_axis_label='population (millions)', tools='box_select,lasso_select')
# Add a circle glyph to p2
p2.circle('fertility', 'population', source=source)
# Create row layout of figures p1 and p2: layout
layout = row(p1, p2)
# Specify the name of the output_file and show the result
output_file('linked_brush.html')
show(layout)


# How to create legends
# Add the first circle glyph to the figure p
p.circle('fertility', 'female_literacy', source=latin_america, size=10, color='red', legend='Latin America')
# Add the second circle glyph to the figure p
p.circle('fertility', 'female_literacy', source=africa, size=10, color='blue', legend='Africa')
# Specify the name of the output_file and show the result
output_file('fert_lit_groups.html')
show(p)


# Positioning and styling legends
# Assign the legend to the bottom left: p.legend.location
p.legend.location = 'bottom_left'
# Fill the legend background with the color 'lightgray': p.legend.background_fill_color
p.legend.background_fill_color = 'lightgray'
# Specify the name of the output_file and show the result
output_file('fert_lit_groups.html')
show(p)


# Adding a hover tooltip
# Import HoverTool from bokeh.models
from bokeh.models import HoverTool
# Create a HoverTool object: hover
hover = HoverTool(tooltips=[('Country', '@Country')])
# Add the HoverTool object to figure p
p.add_tools(hover)
# Specify the name of the output_file and show the result
output_file('hover.html')
show(p)


# High Level Graphs

# A basic histogram
# Import Histogram, output_file, and show from bokeh.charts
from bokeh.charts import Histogram, output_file, show
# Make a Histogram: p
p = Histogram(df, 'female_literacy', title='Female Literacy')
# Set the x axis label
p.xaxis.axis_label = 'Female Literacy'
# Set the y axis label
p.yaxis.axis_label = 'Frequency'
# Specify the name of the output_file and show the result
output_file('histogram.html')
show(p)


# Controlling the number of bins

# Import Histogram, output_file, and show from bokeh.charts
from bokeh.charts import Histogram, output_file, show
# Make the Histogram: p
p = Histogram(df, 'female_literacy', title='Female Literacy', bins=40)
# Set axis labels
p.xaxis.axis_label = 'Female Literacy (% population)'
p.yaxis.axis_label = 'Number of Countries'
# Specify the name of the output_file and show the result
output_file('histogram.html')
show(p)


# Generating multiple histograms at once
# Import Histogram, output_file, and show from bokeh.charts
from bokeh.charts import Histogram, output_file, show
# Make a Histogram: p
p = Histogram(df, 'female_literacy', title='Female Literacy',
              color='Continent', legend='top_left')
# Set axis labels
p.xaxis.axis_label = 'Female Literacy (% population)'
p.yaxis.axis_label = 'Number of Countries'
# Specify the name of the output_file and show the result
output_file('hist_bins.html')
show(p)


# A basic box plot
# Import BoxPlot, output_file, and show from bokeh.charts
from bokeh.charts import BoxPlot, output_file, show
# Make a box plot: p
p = BoxPlot(df, values='female_literacy', label='Continent',
            title='Female Literacy (grouped by Continent)', legend='bottom_right')
# Set the y axis label
p.yaxis.axis_label = 'Female literacy (% population)'
# Specify the name of the output_file and show the result
output_file('boxplot.html')
show(p)


# Color different groups differently
# Import BoxPlot, output_file, and show
from bokeh.charts import BoxPlot, output_file, show
# Make a box plot: p
p = BoxPlot(df, values='female_literacy', label='Continent', color='Continent',
            title='Female Literacy (grouped by Continent)', legend='bottom_right')
# Set y-axis label
p.yaxis.axis_label = 'Female literacy (% population)'
# Specify the name of the output_file and show the result
output_file('boxplot.html')
show(p)


# A basic scatter plot
# Import Scatter, output_file, and show from bokeh.charts
from bokeh.charts import Scatter, output_file, show
# Make a scatter plot: p
p = Scatter(df, x='population', y='female_literacy',
            title='Female Literacy vs Population')
# Set the x-axis label
p.xaxis.axis_label = 'population'
# Set the y-axis label
p.yaxis.axis_label = 'female_literacy'
# Specify the name of the output_file and show the result
output_file('scatterplot.html')
show(p)


# Using colors to group data
# Import Scatter, output_file, and show from bokeh.charts
from bokeh.charts import Scatter, output_file, show
# Make a scatter plot such that each circle is colored by its continent: p
p = Scatter(df, x='population', y='female_literacy', color='Continent',
            title='Female Literacy vs Population')
# Set x-axis and y-axis labels
p.xaxis.axis_label = 'Population (millions)'
p.yaxis.axis_label = 'Female literacy (% population)'
# Specify the name of the output_file and show the result
output_file('scatterplot.html')
show(p)


# Using shapes to group data
# Import Scatter, output_file, and show from bokeh.charts
from bokeh.charts import Scatter, output_file, show
# Make a scatter plot such that each continent has a different marker type: p
p = Scatter(df, x='population', y='female_literacy', color='Continent', marker='Continent', title='Female Literacy vs Population')
# Set x-axis and y-axis labels
p.xaxis.axis_label = 'Population (millions)'
p.yaxis.axis_label = 'Female literacy (% population)'
# Specify the name of the output_file and show the result
output_file('scatterplot.html')
show(p)


# 4. Building interactive apps with Bokeh Server

# Using the current document
# Perform necessary imports
from bokeh.io import curdoc
from bokeh.plotting import figure
# Create a new plot: plot
plot = figure()
# Add a line to the plot
plot.line([1,2,3,4,5], [2,5,4,6,7])
# Add the plot to the current document
curdoc().add_root(plot)


# Add a single slider
# Perform the necessary imports
from bokeh.io import curdoc
from bokeh.layouts import widgetbox
from bokeh.models import Slider
# Create a slider: slider
slider = Slider(title='my slider', start=0, end=10, step=0.1, value=2)
# Create a widgetbox layout: layout
layout = widgetbox(slider)
# Add the layout to the current document
curdoc().add_root(layout)


# Multiple sliders in one document
# Perform necessary imports
from bokeh.io import curdoc
from bokeh.layouts import widgetbox
from bokeh.models import Slider
# Create first slider: slider1
slider1 = Slider(title='slider1', start=0, end=10, step=0.1, value=2)
# Create second slider: slider2
slider2 = Slider(title='slider2', start=10, end=100, step=1, value=20)
# Add slider1 and slider2 to a widgetbox
layout = widgetbox(slider1, slider2)
# Add the layout to the current document
curdoc().add_root(layout)


# How to combine Bokeh models into layouts
# Create ColumnDataSource: source
source = ColumnDataSource(data={'x': x, 'y': y})
# Add a line to the plot
plot.line(x='x', y='y', source=source)
# Create a column layout: layout
layout = column(widgetbox(slider), plot)
# Add the layout to the current document
curdoc().add_root(layout)


# learn about widgetbox callbacks
# Define a callback function: callback
def callback(attr, old, new):
    # Read the current value of the slider: scale
    scale = slider.value
    # Compute the updated y using np.sin(scale/x): new_y
    new_y = np.sin(scale/x)
    # Update source with the new data values
    source.data = {'x': x, 'y': new_y}
# Attach the callback to the 'value' property of slider
slider.on_change('value', callback)


# updating data sources from dropdown callbacks

# Perform necessary imports
from bokeh.models import ColumnDataSource, Select
# Create ColumnDataSource: source
source = ColumnDataSource(data={
    'x' : fertility,
    'y' : female_literacy
})
# Create a new plot: plot
plot = figure()
# Add circles to the plot
plot.circle('x', 'y', source=source)
# Define a callback function: update_plot
def update_plot(attr, old, new):
    # If the new Selection is 'female_literacy', update 'y' to female_literacy
    if new == 'female_literacy':
        source.data = {
            'x' : fertility,
            'y' : female_literacy
        }
    # Else, update 'y' to population
    else:
        source.data = {
            'x' : fertility,
            'y' : population
        }
# Create a dropdown Select widget: select
select = Select(title="distribution", options=['female_literacy', 'population'], value='female_literacy')
# Attach the update_plot callback to the 'value' property of select
select.on_change('value', update_plot)
# Create layout and add to current document
layout = row(select, plot)
curdoc().add_root(layout)


# Synthronoze two dropdowns
# Create two dropdown Select widgets: select1, select2
select1 = Select(title='First', options=['A', 'B'], value='A')
select2 = Select(title='Second', options=['1', '2', '3'], value='1')

# Define a callback function: callback
def callback(attr, old, new):
    # If select1 is 'A'
    if select1.value == 'A':
        # Set select2 options to ['1', '2', '3']
        select2.options = ['1', '2', '3']

        # Set select2 value to '1'
        select2.value = '1'
    else:
        # Set select2 options to ['100', '200', '300']
        select2.options = ['100', '200', '300']
        # Set select2 value to '100'
        select2.value = '100'

# Attach the callback to the 'value' property of select1
select1.on_change('value', callback)

# Create layout and add to current document
layout = widgetbox(select1, select2)
curdoc().add_root(layout)


# Button widgets
# Create a Button with label 'Update Data'
button = Button(label='Update Data')
# Define an update callback with no arguments: update
def update():
    # Compute new y values: y
    y = np.sin(x) + np.random.random(N)
    # Update the ColumnDataSource data dictionary
    source.data['y'] = y
# Add the update callback to the button
button.on_click(update)



# Button styles
# Import CheckboxGroup, RadioGroup, Toggle from bokeh.models
from bokeh.models import CheckboxGroup, RadioGroup, Toggle
# Add a Toggle: toggle
toggle = Toggle(label='Toggle button', button_type='success')
# Add a CheckboxGroup: checkbox
checkbox = CheckboxGroup(labels=['Option 1', 'Option 2', 'Option 3'])
# Add a RadioGroup: radio
radio = RadioGroup(labels=['Option 1', 'Option 2', 'Option 3'])
# Add widgetbox(toggle, checkbox, radio) to the current document
curdoc().add_root(widgetbox(toggle, checkbox, radio))



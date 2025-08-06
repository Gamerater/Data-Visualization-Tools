import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from wordcloud import WordCloud
import folium
from streamlit_folium import folium_static
import altair as alt
import hvplot.pandas
import holoviews as hv
from holoviews import opts
import warnings
warnings.filterwarnings('ignore')

# Import advanced charts
from advanced_charts import *

# Set page config
st.set_page_config(
    page_title="Comprehensive Data Visualization Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📊 Comprehensive Data Visualization Tool")
st.markdown("---")

# Sidebar for data upload and chart selection
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=['csv'],
        help="Upload a CSV file to visualize your data"
    )
    
    st.header("📈 Chart Selection")
    chart_types = [
        "Area Graph", "Bar Chart", "Box & Whisker Plot", "Brainstorm",
        "Bubble Chart", "Bubble Map", "Bullet Graph", "Calendar",
        "Candlestick Chart", "Chord Diagram", "Choropleth Map", "Circle Packing",
        "Connection Map", "Density Plot", "Donut Chart", "Dot Map",
        "Dot Matrix Chart", "Error Bars", "Flow Chart", "Flow Map",
        "Gantt Chart", "Heatmap", "Histogram", "Illustration Diagram",
        "Kagi Chart", "Line Graph", "Marimekko Chart", "Multi-set Bar Chart",
        "Network Diagram", "Nightingale Rose Chart", "Non-ribbon Chord Diagram",
        "Open-high-low-close Chart", "Parallel Coordinates Plot", "Parallel Sets",
        "Pictogram Chart", "Pie Chart", "Point & Figure Chart", "Population Pyramid",
        "Proportional Area Chart", "Radar Chart", "Radial Bar Chart", "Radial Column Chart",
        "Sankey Diagram", "Scatterplot", "Span Chart", "Spiral Plot",
        "Stacked Area Graph", "Stacked Bar Graph", "Stem & Leaf Plot", "Stream Graph",
        "Sunburst Diagram", "Tally Chart", "Timeline", "Timetable",
        "Tree Diagram", "Treemap", "Venn Diagram", "Violin Plot", "Word Cloud"
    ]
    
    selected_chart = st.selectbox("Select Chart Type", chart_types)

# Sample data generation if no file is uploaded
def generate_sample_data():
    np.random.seed(42)
    n = 100
    
    # Sample data for various chart types
    data = {
        'date': pd.date_range('2023-01-01', periods=n, freq='D'),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n),
        'subcategory': np.random.choice(['X', 'Y', 'Z'], n),
        'value': np.random.normal(100, 20, n),
        'value2': np.random.normal(50, 10, n),
        'size': np.random.uniform(10, 100, n),
        'color': np.random.choice(['Red', 'Blue', 'Green', 'Yellow'], n),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'open': np.random.normal(100, 5, n),
        'high': np.random.normal(105, 5, n),
        'low': np.random.normal(95, 5, n),
        'close': np.random.normal(102, 5, n),
        'volume': np.random.randint(1000, 10000, n),
        'text': ['Sample text ' + str(i) for i in range(n)]
    }
    
    return pd.DataFrame(data)

# Load data
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        df = generate_sample_data()
        st.info("📊 Using sample data instead")
else:
    df = generate_sample_data()
    st.info("📊 Using sample data. Upload a CSV file to use your own data.")

# Display data info
with st.expander("📋 Data Preview"):
    st.dataframe(df.head())
    st.write(f"**Data Shape:** {df.shape}")
    st.write(f"**Columns:** {list(df.columns)}")

# Chart rendering functions
def render_area_graph(df):
    fig = px.area(df, x='date', y='value', color='category', 
                   title="Area Graph", template="plotly_white")
    return fig

def render_bar_chart(df):
    fig = px.bar(df.groupby('category')['value'].mean().reset_index(), 
                  x='category', y='value', title="Bar Chart", template="plotly_white")
    return fig

def render_box_whisker(df):
    fig = px.box(df, x='category', y='value', color='subcategory',
                  title="Box & Whisker Plot", template="plotly_white")
    return fig

def render_bubble_chart(df):
    fig = px.scatter(df, x='value', y='value2', size='size', color='category',
                     title="Bubble Chart", template="plotly_white")
    return fig

def render_heatmap(df):
    pivot_table = df.pivot_table(values='value', index='category', columns='subcategory', aggfunc='mean')
    fig = px.imshow(pivot_table, title="Heatmap", template="plotly_white")
    return fig

def render_histogram(df):
    fig = px.histogram(df, x='value', color='category', nbins=20,
                       title="Histogram", template="plotly_white")
    return fig

def render_line_graph(df):
    fig = px.line(df, x='date', y='value', color='category',
                   title="Line Graph", template="plotly_white")
    return fig

def render_pie_chart(df):
    fig = px.pie(df.groupby('category')['value'].sum().reset_index(), 
                  values='value', names='category', title="Pie Chart")
    return fig

def render_donut_chart(df):
    fig = px.pie(df.groupby('category')['value'].sum().reset_index(), 
                  values='value', names='category', title="Donut Chart")
    fig.update_traces(hole=0.4)
    return fig

def render_scatterplot(df):
    fig = px.scatter(df, x='value', y='value2', color='category', size='size',
                     title="Scatterplot", template="plotly_white")
    return fig

def render_violin_plot(df):
    fig = px.violin(df, x='category', y='value', color='subcategory',
                     title="Violin Plot", template="plotly_white")
    return fig

def render_radar_chart(df):
    categories = df['category'].unique()
    values = [df[df['category'] == cat]['value'].mean() for cat in categories]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Values'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(values) * 1.2])),
        showlegend=False,
        title="Radar Chart"
    )
    return fig

def render_sunburst_diagram(df):
    fig = px.sunburst(df, path=['category', 'subcategory'], values='value',
                       title="Sunburst Diagram")
    return fig

def render_treemap(df):
    fig = px.treemap(df, path=['category', 'subcategory'], values='value',
                      title="Treemap")
    return fig

def render_sankey_diagram(df):
    # Create source and target for Sankey
    categories = df['category'].unique()
    subcategories = df['subcategory'].unique()
    
    source = []
    target = []
    value = []
    
    for cat in categories:
        for subcat in subcategories:
            val = df[(df['category'] == cat) & (df['subcategory'] == subcat)]['value'].sum()
            if val > 0:
                source.append(cat)
                target.append(subcat)
                value.append(val)
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(label=list(set(source + target))),
        link=dict(source=[list(set(source + target)).index(s) for s in source],
                  target=[list(set(source + target)).index(t) for t in target],
                  value=value)
    )])
    fig.update_layout(title="Sankey Diagram")
    return fig

def render_parallel_coordinates(df):
    fig = px.parallel_coordinates(df, color='category',
                                  title="Parallel Coordinates Plot")
    return fig

def render_3d_scatter(df):
    fig = px.scatter_3d(df, x='value', y='value2', z='size', color='category',
                         title="3D Scatter Plot")
    return fig

def render_waterfall_chart(df):
    categories = df['category'].unique()
    values = [df[df['category'] == cat]['value'].sum() for cat in categories]
    
    fig = go.Figure(go.Waterfall(
        name="Values",
        orientation="h",
        measure=["relative"] * len(categories),
        x=values,
        textposition="outside",
        text=values,
        y=categories,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    fig.update_layout(title="Waterfall Chart")
    return fig

def render_funnel_chart(df):
    categories = df['category'].unique()
    values = [df[df['category'] == cat]['value'].sum() for cat in categories]
    
    fig = go.Figure(go.Funnel(
        y=categories,
        x=values,
        textinfo="value+percent initial"
    ))
    fig.update_layout(title="Funnel Chart")
    return fig

def render_candlestick_chart(df):
    fig = go.Figure(data=[go.Candlestick(x=df['date'],
                                         open=df['open'],
                                         high=df['high'],
                                         low=df['low'],
                                         close=df['close'])])
    fig.update_layout(title="Candlestick Chart")
    return fig

def render_word_cloud(df):
    # Create text for word cloud
    text = ' '.join(df['text'].astype(str))
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud')
    
    return fig

def render_network_diagram(df):
    # Create a simple network
    G = nx.Graph()
    
    # Add nodes and edges based on categories
    categories = df['category'].unique()
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            if i != j:
                weight = np.random.random()
                G.add_edge(cat1, cat2, weight=weight)
    
    pos = nx.spring_layout(G)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=1000, font_size=10, font_weight='bold')
    plt.title('Network Diagram')
    
    return fig

def render_choropleth_map(df):
    # Create sample geographic data
    countries = ['USA', 'Canada', 'UK', 'Germany', 'France', 'Japan', 'Australia']
    values = np.random.normal(100, 20, len(countries))
    
    fig = px.choropleth(
        locations=countries,
        locationmode='country names',
        color=values,
        title="Choropleth Map",
        color_continuous_scale='Viridis'
    )
    return fig

def render_bubble_map(df):
    # Create sample geographic data
    cities = ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
    lats = [40.7128, 51.5074, 35.6762, 48.8566, -33.8688]
    lons = [-74.0060, -0.1278, 139.6503, 2.3522, 151.2093]
    sizes = np.random.uniform(10, 100, len(cities))
    
    fig = px.scatter_mapbox(
        lat=lats,
        lon=lons,
        size=sizes,
        hover_name=cities,
        title="Bubble Map",
        mapbox_style="carto-positron"
    )
    return fig

def render_gantt_chart(df):
    # Create sample project data
    tasks = ['Task A', 'Task B', 'Task C', 'Task D']
    start_dates = ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15']
    end_dates = ['2023-01-14', '2023-01-31', '2023-02-14', '2023-03-01']
    
    fig = px.timeline(
        x_start=start_dates,
        x_end=end_dates,
        y=tasks,
        title="Gantt Chart"
    )
    return fig

def render_calendar_heatmap(df):
    # Create calendar data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    values = np.random.normal(100, 20, len(dates))
    
    calendar_data = pd.DataFrame({
        'date': dates,
        'value': values
    })
    
    fig = px.density_heatmap(
        calendar_data,
        x=calendar_data['date'].dt.day,
        y=calendar_data['date'].dt.month,
        z='value',
        title="Calendar Heatmap"
    )
    return fig

def render_population_pyramid(df):
    # Create sample age/gender data
    ages = list(range(0, 101, 10))
    male_pop = np.random.normal(1000, 200, len(ages))
    female_pop = np.random.normal(1000, 200, len(ages))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=ages,
        x=male_pop,
        name='Male',
        orientation='h',
        marker_color='blue'
    ))
    
    fig.add_trace(go.Bar(
        y=ages,
        x=-female_pop,
        name='Female',
        orientation='h',
        marker_color='pink'
    ))
    
    fig.update_layout(
        title="Population Pyramid",
        barmode='overlay',
        xaxis_title="Population"
    )
    return fig

def render_stream_graph(df):
    # Create sample time series data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='M')
    categories = ['A', 'B', 'C', 'D']
    
    stream_data = []
    for cat in categories:
        values = np.random.normal(100, 20, len(dates))
        for date, value in zip(dates, values):
            stream_data.append({'date': date, 'category': cat, 'value': value})
    
    stream_df = pd.DataFrame(stream_data)
    
    fig = px.area(stream_df, x='date', y='value', color='category',
                   title="Stream Graph")
    return fig

def render_spiral_plot(df):
    # Create spiral data
    theta = np.linspace(0, 8*np.pi, 100)
    r = theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Spiral'))
    fig.update_layout(title="Spiral Plot")
    return fig

def render_venn_diagram(df):
    # Create sample set data
    set_a = set(['A', 'B', 'C', 'D'])
    set_b = set(['C', 'D', 'E', 'F'])
    set_c = set(['D', 'E', 'G', 'H'])
    
    # Create Venn diagram using matplotlib
    from matplotlib_venn import venn3
    
    fig, ax = plt.subplots(figsize=(10, 8))
    venn3([set_a, set_b, set_c], ('Set A', 'Set B', 'Set C'), ax=ax)
    plt.title('Venn Diagram')
    
    return fig

# Chart mapping
chart_functions = {
    "Area Graph": render_area_graph,
    "Bar Chart": render_bar_chart,
    "Box & Whisker Plot": render_box_whisker,
    "Brainstorm": render_brainstorm,
    "Bubble Chart": render_bubble_chart,
    "Bubble Map": render_bubble_map,
    "Bullet Graph": render_bullet_graph,
    "Calendar": render_calendar_heatmap,
    "Candlestick Chart": render_candlestick_chart,
    "Chord Diagram": render_chord_diagram,
    "Choropleth Map": render_choropleth_map,
    "Circle Packing": render_circle_packing,
    "Connection Map": render_connection_map,
    "Density Plot": render_density_plot,
    "Donut Chart": render_donut_chart,
    "Dot Map": render_dot_map,
    "Dot Matrix Chart": render_dot_matrix_chart,
    "Error Bars": render_error_bars,
    "Flow Chart": render_flow_chart,
    "Flow Map": render_flow_map,
    "Gantt Chart": render_gantt_chart,
    "Heatmap": render_heatmap,
    "Histogram": render_histogram,
    "Illustration Diagram": render_illustration_diagram,
    "Kagi Chart": render_kagi_chart,
    "Line Graph": render_line_graph,
    "Marimekko Chart": render_marimekko_chart,
    "Multi-set Bar Chart": render_multi_set_bar_chart,
    "Network Diagram": render_network_diagram,
    "Nightingale Rose Chart": render_nightingale_rose_chart,
    "Non-ribbon Chord Diagram": render_non_ribbon_chord_diagram,
    "Open-high-low-close Chart": render_open_high_low_close_chart,
    "Parallel Coordinates Plot": render_parallel_coordinates,
    "Parallel Sets": render_parallel_sets,
    "Pictogram Chart": render_pictogram_chart,
    "Pie Chart": render_pie_chart,
    "Point & Figure Chart": render_point_figure_chart,
    "Population Pyramid": render_population_pyramid,
    "Proportional Area Chart": render_proportional_area_chart,
    "Radar Chart": render_radar_chart,
    "Radial Bar Chart": render_radial_bar_chart,
    "Radial Column Chart": render_radial_column_chart,
    "Sankey Diagram": render_sankey_diagram,
    "Scatterplot": render_scatterplot,
    "Span Chart": render_span_chart,
    "Spiral Plot": render_spiral_plot,
    "Stacked Area Graph": render_stacked_area_graph,
    "Stacked Bar Graph": render_stacked_bar_graph,
    "Stem & Leaf Plot": render_stem_leaf_plot,
    "Stream Graph": render_stream_graph,
    "Sunburst Diagram": render_sunburst_diagram,
    "Tally Chart": render_tally_chart,
    "Timeline": render_timeline,
    "Timetable": render_timetable,
    "Tree Diagram": render_tree_diagram,
    "Treemap": render_treemap,
    "Venn Diagram": render_venn_diagram,
    "Violin Plot": render_violin_plot,
    "Word Cloud": render_word_cloud
}

# Render the selected chart
st.header(f"📊 {selected_chart}")

if selected_chart in chart_functions:
    try:
        result = chart_functions[selected_chart](df)
        
        if isinstance(result, str):  # Text-based chart
            st.text(result)
        elif hasattr(result, 'show'):  # Plotly figure
            st.plotly_chart(result, use_container_width=True)
        else:  # Matplotlib figure
            st.pyplot(result)
            
    except Exception as e:
        st.error(f"❌ Error rendering {selected_chart}: {e}")
        st.info("This chart type is not yet implemented or requires specific data format.")
else:
    st.info("🚧 This chart type is not yet implemented. Coming soon!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📊 Comprehensive Data Visualization Tool | Built with Streamlit & Plotly</p>
</div>
""", unsafe_allow_html=True) 
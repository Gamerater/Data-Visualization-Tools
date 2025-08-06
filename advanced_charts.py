import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3
import networkx as nx
from wordcloud import WordCloud
import folium
from streamlit_folium import folium_static

def render_stacked_area_graph(df):
    """Render stacked area graph"""
    fig = px.area(df, x='date', y='value', color='category', 
                   title="Stacked Area Graph", template="plotly_white")
    return fig

def render_stacked_bar_graph(df):
    """Render stacked bar graph"""
    fig = px.bar(df, x='category', y='value', color='subcategory',
                  title="Stacked Bar Graph", template="plotly_white")
    return fig

def render_multi_set_bar_chart(df):
    """Render multi-set bar chart"""
    fig = px.bar(df, x='category', y='value', color='subcategory',
                  barmode='group', title="Multi-set Bar Chart", template="plotly_white")
    return fig

def render_radial_bar_chart(df):
    """Render radial bar chart"""
    categories = df['category'].unique()
    values = [df[df['category'] == cat]['value'].mean() for cat in categories]
    
    fig = go.Figure()
    fig.add_trace(go.Barpolar(
        r=values,
        theta=categories,
        width=0.5,
        marker_color=px.colors.qualitative.Set3,
        marker_line_color="white",
        marker_line_width=2,
        opacity=0.8
    ))
    fig.update_layout(
        title="Radial Bar Chart",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.2]
            )),
        showlegend=False
    )
    return fig

def render_radial_column_chart(df):
    """Render radial column chart"""
    categories = df['category'].unique()
    values = [df[df['category'] == cat]['value'].mean() for cat in categories]
    
    fig = go.Figure()
    fig.add_trace(go.Barpolar(
        r=values,
        theta=categories,
        width=0.8,
        marker_color=px.colors.qualitative.Set1,
        marker_line_color="white",
        marker_line_width=2,
        opacity=0.8
    ))
    fig.update_layout(
        title="Radial Column Chart",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.2]
            )),
        showlegend=False
    )
    return fig

def render_nightingale_rose_chart(df):
    """Render Nightingale Rose Chart"""
    categories = df['category'].unique()
    values = [df[df['category'] == cat]['value'].mean() for cat in categories]
    
    fig = go.Figure()
    fig.add_trace(go.Barpolar(
        r=values,
        theta=categories,
        width=2*np.pi/len(categories),
        marker_color=px.colors.qualitative.Set3,
        marker_line_color="white",
        marker_line_width=2,
        opacity=0.8
    ))
    fig.update_layout(
        title="Nightingale Rose Chart",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.2]
            )),
        showlegend=False
    )
    return fig

def render_proportional_area_chart(df):
    """Render proportional area chart"""
    categories = df['category'].unique()
    values = [df[df['category'] == cat]['value'].sum() for cat in categories]
    
    # Calculate circle areas proportional to values
    max_value = max(values)
    areas = [(v/max_value) * 100 for v in values]
    
    fig = go.Figure()
    for i, (cat, area) in enumerate(zip(categories, areas)):
        fig.add_trace(go.Scatter(
            x=[i],
            y=[area],
            mode='markers',
            marker=dict(
                size=area * 2,
                color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)],
                opacity=0.7
            ),
            name=cat,
            text=f"{cat}: {values[i]:.1f}",
            hovertemplate="%{text}<extra></extra>"
        ))
    
    fig.update_layout(
        title="Proportional Area Chart",
        xaxis_title="Categories",
        yaxis_title="Proportional Area",
        showlegend=True
    )
    return fig

def render_bullet_graph(df):
    """Render bullet graph"""
    categories = df['category'].unique()
    values = [df[df['category'] == cat]['value'].mean() for cat in categories]
    
    fig = go.Figure()
    
    # Add background bars
    fig.add_trace(go.Bar(
        x=categories,
        y=[max(values) * 1.2] * len(categories),
        name='Background',
        marker_color='lightgray',
        opacity=0.3
    ))
    
    # Add value bars
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        name='Values',
        marker_color='blue',
        opacity=0.8
    ))
    
    fig.update_layout(
        title="Bullet Graph",
        barmode='overlay',
        yaxis_title="Values"
    )
    return fig

def render_error_bars(df):
    """Render error bars chart"""
    categories = df['category'].unique()
    means = [df[df['category'] == cat]['value'].mean() for cat in categories]
    stds = [df[df['category'] == cat]['value'].std() for cat in categories]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=categories,
        y=means,
        mode='markers+lines',
        error_y=dict(type='data', array=stds, visible=True),
        name='Mean ± Std',
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title="Error Bars Chart",
        xaxis_title="Categories",
        yaxis_title="Values"
    )
    return fig

def render_density_plot(df):
    """Render density plot"""
    fig = px.histogram(df, x='value', color='category', 
                       marginal='box', title="Density Plot", template="plotly_white")
    return fig

def render_stem_leaf_plot(df):
    """Render stem and leaf plot"""
    values = df['value'].values
    
    # Create stem and leaf plot
    stems = []
    leaves = []
    
    for value in sorted(values):
        stem = int(value // 10)
        leaf = int(value % 10)
        stems.append(stem)
        leaves.append(leaf)
    
    # Group by stem
    stem_leaf_data = {}
    for stem, leaf in zip(stems, leaves):
        if stem not in stem_leaf_data:
            stem_leaf_data[stem] = []
        stem_leaf_data[stem].append(leaf)
    
    # Create text representation
    plot_text = "Stem and Leaf Plot\n"
    plot_text += "=" * 20 + "\n"
    
    for stem in sorted(stem_leaf_data.keys()):
        leaves_str = ''.join(map(str, sorted(stem_leaf_data[stem])))
        plot_text += f"{stem} | {leaves_str}\n"
    
    return plot_text

def render_kagi_chart(df):
    """Render Kagi chart"""
    # Sort by date
    df_sorted = df.sort_values('date')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sorted['date'],
        y=df_sorted['close'],
        mode='lines+markers',
        name='Close Price',
        line=dict(width=2)
    ))
    
    fig.update_layout(
        title="Kagi Chart",
        xaxis_title="Date",
        yaxis_title="Close Price"
    )
    return fig

def render_point_figure_chart(df):
    """Render Point and Figure chart"""
    # Create sample price data
    prices = df['close'].values
    dates = df['date'].values
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode='lines+markers',
        name='Price',
        line=dict(width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Point and Figure Chart",
        xaxis_title="Date",
        yaxis_title="Price"
    )
    return fig

def render_marimekko_chart(df):
    """Render Marimekko chart"""
    pivot_table = df.pivot_table(values='value', index='category', columns='subcategory', aggfunc='sum')
    
    fig = go.Figure()
    
    for col in pivot_table.columns:
        fig.add_trace(go.Bar(
            name=col,
            x=pivot_table.index,
            y=pivot_table[col],
            width=0.8
        ))
    
    fig.update_layout(
        title="Marimekko Chart",
        barmode='stack',
        xaxis_title="Categories",
        yaxis_title="Values"
    )
    return fig

def render_pictogram_chart(df):
    """Render pictogram chart"""
    categories = df['category'].unique()
    values = [df[df['category'] == cat]['value'].sum() for cat in categories]
    
    # Create pictogram using emoji
    emojis = ['📊', '📈', '📉', '💰', '🎯', '📋', '📌', '📍']
    
    fig = go.Figure()
    for i, (cat, value) in enumerate(zip(categories, values)):
        fig.add_trace(go.Bar(
            x=[cat],
            y=[value],
            name=cat,
            text=[emojis[i % len(emojis)] * int(value/10)],
            textposition='inside',
            marker_color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
        ))
    
    fig.update_layout(
        title="Pictogram Chart",
        showlegend=False
    )
    return fig

def render_span_chart(df):
    """Render span chart"""
    fig = go.Figure()
    
    for category in df['category'].unique():
        cat_data = df[df['category'] == category]
        fig.add_trace(go.Scatter(
            x=cat_data['date'],
            y=cat_data['value'],
            mode='lines',
            name=category,
            fill='tonexty' if category == df['category'].iloc[0] else 'tonexty'
        ))
    
    fig.update_layout(
        title="Span Chart",
        xaxis_title="Date",
        yaxis_title="Value"
    )
    return fig

def render_tally_chart(df):
    """Render tally chart"""
    categories = df['category'].unique()
    counts = [len(df[df['category'] == cat]) for cat in categories]
    
    # Create tally marks
    tally_marks = []
    for count in counts:
        groups_of_five = count // 5
        remainder = count % 5
        tally = '||||' * groups_of_five + '|' * remainder
        tally_marks.append(tally)
    
    # Create text representation
    chart_text = "Tally Chart\n"
    chart_text += "=" * 20 + "\n"
    
    for cat, count, tally in zip(categories, counts, tally_marks):
        chart_text += f"{cat}: {tally} ({count})\n"
    
    return chart_text

def render_timeline(df):
    """Render timeline"""
    # Create sample timeline data
    events = ['Event A', 'Event B', 'Event C', 'Event D']
    dates = ['2023-01-01', '2023-03-15', '2023-06-30', '2023-12-31']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=events,
        mode='markers+lines',
        marker=dict(size=15, symbol='diamond'),
        line=dict(width=3),
        name='Timeline'
    ))
    
    fig.update_layout(
        title="Timeline",
        xaxis_title="Date",
        yaxis_title="Events"
    )
    return fig

def render_timetable(df):
    """Render timetable"""
    # Create sample timetable data
    times = ['09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00']
    activities = ['Meeting A', 'Break', 'Meeting B', 'Lunch', 'Meeting C', 'Break', 'Meeting D', 'Wrap-up']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=times,
        y=[1] * len(times),
        text=activities,
        textposition='inside',
        marker_color=px.colors.qualitative.Set3
    ))
    
    fig.update_layout(
        title="Timetable",
        xaxis_title="Time",
        yaxis_title="",
        showlegend=False
    )
    return fig

def render_tree_diagram(df):
    """Render tree diagram"""
    # Create hierarchical data
    categories = df['category'].unique()
    
    fig = go.Figure()
    
    # Add root
    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode='markers+text',
        marker=dict(size=20, color='red'),
        text=['Root'],
        textposition='middle center',
        name='Root'
    ))
    
    # Add categories
    for i, cat in enumerate(categories):
        angle = 2 * np.pi * i / len(categories)
        x = 1 * np.cos(angle)
        y = 1 * np.sin(angle)
        
        fig.add_trace(go.Scatter(
            x=[0, x],
            y=[0, y],
            mode='lines+markers+text',
            marker=dict(size=15, color='blue'),
            text=['', cat],
            textposition='middle center',
            name=cat
        ))
    
    fig.update_layout(
        title="Tree Diagram",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False
    )
    return fig

def render_illustration_diagram(df):
    """Render illustration diagram"""
    # Create a simple flowchart
    fig = go.Figure()
    
    # Add nodes
    nodes = ['Start', 'Process A', 'Process B', 'End']
    x_pos = [0, 1, 2, 3]
    y_pos = [0, 0, 0, 0]
    
    fig.add_trace(go.Scatter(
        x=x_pos,
        y=y_pos,
        mode='markers+text',
        marker=dict(size=30, color=['green', 'blue', 'blue', 'red']),
        text=nodes,
        textposition='middle center',
        name='Process'
    ))
    
    # Add connections
    for i in range(len(nodes) - 1):
        fig.add_trace(go.Scatter(
            x=[x_pos[i], x_pos[i + 1]],
            y=[y_pos[i], y_pos[i + 1]],
            mode='lines',
            line=dict(width=3, color='gray'),
            showlegend=False
        ))
    
    fig.update_layout(
        title="Illustration Diagram",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False
    )
    return fig

def render_flow_chart(df):
    """Render flow chart"""
    # Create a simple flow chart
    fig = go.Figure()
    
    # Define flow chart elements
    elements = ['Start', 'Decision', 'Process', 'End']
    x_pos = [0, 1, 2, 3]
    y_pos = [0, 0, 0, 0]
    
    # Different shapes for different elements
    shapes = ['circle', 'diamond', 'rectangle', 'circle']
    colors = ['green', 'yellow', 'blue', 'red']
    
    for i, (element, x, y, shape, color) in enumerate(zip(elements, x_pos, y_pos, shapes, colors)):
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            marker=dict(size=30, color=color, symbol=shape),
            text=[element],
            textposition='middle center',
            name=element
        ))
    
    # Add connections
    for i in range(len(elements) - 1):
        fig.add_trace(go.Scatter(
            x=[x_pos[i], x_pos[i + 1]],
            y=[y_pos[i], y_pos[i + 1]],
            mode='lines',
            line=dict(width=3, color='gray'),
            showlegend=False
        ))
    
    fig.update_layout(
        title="Flow Chart",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False
    )
    return fig

def render_connection_map(df):
    """Render connection map"""
    # Create connection data
    categories = df['category'].unique()
    
    fig = go.Figure()
    
    # Add nodes
    for i, cat in enumerate(categories):
        angle = 2 * np.pi * i / len(categories)
        x = 2 * np.cos(angle)
        y = 2 * np.sin(angle)
        
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            marker=dict(size=20, color='blue'),
            text=[cat],
            textposition='middle center',
            name=cat
        ))
    
    # Add connections
    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            angle1 = 2 * np.pi * i / len(categories)
            angle2 = 2 * np.pi * j / len(categories)
            x1 = 2 * np.cos(angle1)
            y1 = 2 * np.sin(angle1)
            x2 = 2 * np.cos(angle2)
            y2 = 2 * np.sin(angle2)
            
            fig.add_trace(go.Scatter(
                x=[x1, x2],
                y=[y1, y2],
                mode='lines',
                line=dict(width=1, color='gray'),
                showlegend=False
            ))
    
    fig.update_layout(
        title="Connection Map",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False
    )
    return fig

def render_dot_map(df):
    """Render dot map"""
    # Create sample geographic data
    cities = ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
    lats = [40.7128, 51.5074, 35.6762, 48.8566, -33.8688]
    lons = [-74.0060, -0.1278, 139.6503, 2.3522, 151.2093]
    sizes = np.random.uniform(5, 20, len(cities))
    
    fig = px.scatter_mapbox(
        lat=lats,
        lon=lons,
        size=sizes,
        hover_name=cities,
        title="Dot Map",
        mapbox_style="carto-positron"
    )
    return fig

def render_dot_matrix_chart(df):
    """Render dot matrix chart"""
    categories = df['category'].unique()
    subcategories = df['subcategory'].unique()
    
    # Create matrix data
    matrix_data = []
    for cat in categories:
        for subcat in subcategories:
            value = df[(df['category'] == cat) & (df['subcategory'] == subcat)]['value'].sum()
            matrix_data.append({'category': cat, 'subcategory': subcat, 'value': value})
    
    matrix_df = pd.DataFrame(matrix_data)
    
    fig = px.scatter(matrix_df, x='category', y='subcategory', size='value',
                     title="Dot Matrix Chart", template="plotly_white")
    return fig

def render_circle_packing(df):
    """Render circle packing"""
    categories = df['category'].unique()
    values = [df[df['category'] == cat]['value'].sum() for cat in categories]
    
    fig = go.Figure()
    
    # Create circles with areas proportional to values
    max_value = max(values)
    for i, (cat, value) in enumerate(zip(categories, values)):
        radius = np.sqrt(value / max_value) * 0.5
        fig.add_shape(
            type="circle",
            x0=i-radius, y0=-radius,
            x1=i+radius, y1=radius,
            fillcolor=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)],
            opacity=0.7,
            line=dict(color="white", width=2)
        )
        
        fig.add_annotation(
            x=i, y=0,
            text=cat,
            showarrow=False,
            font=dict(size=10, color="white")
        )
    
    fig.update_layout(
        title="Circle Packing",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False
    )
    return fig

def render_chord_diagram(df):
    """Render chord diagram"""
    # Create sample chord diagram data
    categories = ['A', 'B', 'C', 'D']
    matrix = np.random.rand(4, 4)
    np.fill_diagonal(matrix, 0)
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(label=categories),
        link=dict(
            source=[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            target=[1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
            value=matrix.flatten()
        )
    )])
    
    fig.update_layout(title="Chord Diagram")
    return fig

def render_non_ribbon_chord_diagram(df):
    """Render non-ribbon chord diagram"""
    # Similar to chord diagram but with different styling
    categories = ['A', 'B', 'C', 'D']
    matrix = np.random.rand(4, 4)
    np.fill_diagonal(matrix, 0)
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(label=categories, color=px.colors.qualitative.Set1),
        link=dict(
            source=[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            target=[1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
            value=matrix.flatten(),
            color=['rgba(0,0,0,0.2)'] * 12
        )
    )])
    
    fig.update_layout(title="Non-ribbon Chord Diagram")
    return fig

def render_open_high_low_close_chart(df):
    """Render OHLC chart"""
    fig = go.Figure(data=[go.Ohlc(x=df['date'],
                                   open=df['open'],
                                   high=df['high'],
                                   low=df['low'],
                                   close=df['close'])])
    fig.update_layout(title="Open-High-Low-Close Chart")
    return fig

def render_parallel_sets(df):
    """Render parallel sets"""
    fig = px.parallel_categories(df, dimensions=['category', 'subcategory', 'color'],
                                 title="Parallel Sets")
    return fig

def render_brainstorm(df):
    """Render brainstorm diagram"""
    # Create a simple mind map
    central_topic = "Data Analysis"
    branches = ['Category A', 'Category B', 'Category C', 'Category D']
    
    fig = go.Figure()
    
    # Central topic
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers+text',
        marker=dict(size=30, color='red'),
        text=[central_topic],
        textposition='middle center',
        name='Central Topic'
    ))
    
    # Branches
    for i, branch in enumerate(branches):
        angle = 2 * np.pi * i / len(branches)
        x = 2 * np.cos(angle)
        y = 2 * np.sin(angle)
        
        fig.add_trace(go.Scatter(
            x=[0, x], y=[0, y],
            mode='lines+markers+text',
            marker=dict(size=20, color='blue'),
            text=['', branch],
            textposition='middle center',
            name=branch
        ))
    
    fig.update_layout(
        title="Brainstorm Diagram",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False
    )
    return fig

def render_flow_map(df):
    """Render flow map"""
    # Create sample flow data
    origins = ['City A', 'City B', 'City C']
    destinations = ['City D', 'City E', 'City F']
    flows = np.random.randint(10, 100, len(origins))
    
    fig = go.Figure()
    
    # Add flow lines
    for i, (origin, dest, flow) in enumerate(zip(origins, destinations, flows)):
        fig.add_trace(go.Scatter(
            x=[i, i+1],
            y=[0, 1],
            mode='lines',
            line=dict(width=flow/10, color='blue'),
            name=f'{origin} → {dest}',
            showlegend=True
        ))
    
    fig.update_layout(
        title="Flow Map",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=True
    )
    return fig 
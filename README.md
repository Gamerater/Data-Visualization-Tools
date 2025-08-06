# 📊 Comprehensive Data Visualization Tool

A powerful Python-based data visualization tool that supports **60+ different chart types** including area graphs, bar charts, heatmaps, network diagrams, geographic visualizations, and much more.

## 🚀 Features

### Chart Types Supported

**Basic Charts:**
- Area Graph
- Bar Chart
- Line Graph
- Scatterplot
- Pie Chart
- Donut Chart
- Histogram
- Box & Whisker Plot

**Advanced Charts:**
- Heatmap
- Violin Plot
- Radar Chart
- Sunburst Diagram
- Treemap
- Sankey Diagram
- Parallel Coordinates Plot

**Geographic Visualizations:**
- Choropleth Map
- Bubble Map
- Dot Map
- Flow Map

**Network & Relationship Charts:**
- Network Diagram
- Connection Map
- Chord Diagram
- Non-ribbon Chord Diagram

**Specialized Charts:**
- Candlestick Chart
- OHLC Chart
- Gantt Chart
- Timeline
- Population Pyramid
- Word Cloud
- Venn Diagram

**Radial & Polar Charts:**
- Radial Bar Chart
- Radial Column Chart
- Nightingale Rose Chart
- Spiral Plot

**Statistical Charts:**
- Error Bars
- Density Plot
- Stem & Leaf Plot
- Kagi Chart
- Point & Figure Chart

**Hierarchical Charts:**
- Tree Diagram
- Circle Packing
- Brainstorm Diagram
- Flow Chart
- Illustration Diagram

**Time-based Charts:**
- Calendar Heatmap
- Timetable
- Stream Graph
- Span Chart

**Special Purpose Charts:**
- Bullet Graph
- Marimekko Chart
- Pictogram Chart
- Tally Chart
- Proportional Area Chart

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📁 Project Structure

```
Data Visualization Tools/
├── app.py                 # Main Streamlit application
├── advanced_charts.py     # Additional chart implementations
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🎯 Usage

### 1. **Data Upload**
- Upload your CSV file using the file uploader in the sidebar
- The tool will automatically generate sample data if no file is uploaded
- Supported format: CSV files

### 2. **Chart Selection**
- Choose from 60+ different chart types from the dropdown menu
- Each chart type is optimized for specific data patterns

### 3. **Data Preview**
- View your data structure and content in the expandable "Data Preview" section
- Check data shape and column information

### 4. **Interactive Visualizations**
- All charts are interactive and responsive
- Hover over elements for detailed information
- Zoom, pan, and explore your data

## 📊 Sample Data Structure

The tool works with CSV files containing columns like:
- `date`: Time series data
- `category`: Categorical data
- `subcategory`: Sub-categories
- `value`: Numerical values
- `value2`: Secondary numerical values
- `size`: Size values for bubble charts
- `color`: Color categories
- `region`: Geographic regions
- `open`, `high`, `low`, `close`: OHLC data
- `volume`: Volume data
- `text`: Text data for word clouds

## 🔧 Customization

### Adding New Chart Types
1. Create a new function in `advanced_charts.py`
2. Add the function to the `chart_functions` dictionary in `app.py`
3. Update the chart types list in the sidebar

### Data Format Requirements
- **Time Series Charts**: Require `date` column
- **Geographic Charts**: Require latitude/longitude or country names
- **Network Charts**: Work with relationship data
- **Statistical Charts**: Require numerical data

## 🎨 Chart Categories

### **Basic Analytics**
- Bar, Line, Area, Scatter plots
- Histograms, Box plots
- Pie and Donut charts

### **Advanced Analytics**
- Heatmaps, Violin plots
- Radar charts, Sunburst diagrams
- Parallel coordinates, Sankey diagrams

### **Geographic Visualization**
- Choropleth maps
- Bubble maps, Dot maps
- Flow maps

### **Network Analysis**
- Network diagrams
- Connection maps
- Chord diagrams

### **Time Series**
- Candlestick charts
- Gantt charts
- Calendar heatmaps

### **Statistical Analysis**
- Error bars
- Density plots
- Stem & leaf plots

## 🚀 Advanced Features

- **Interactive Charts**: All charts support zoom, pan, and hover interactions
- **Responsive Design**: Works on desktop and mobile devices
- **Data Export**: Download charts as images
- **Real-time Updates**: Charts update automatically when data changes
- **Multiple Chart Types**: Switch between different visualizations instantly

## 🔍 Troubleshooting

### Common Issues:

1. **Chart not rendering**: Check if your data has the required columns
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Port already in use**: Change the port with `streamlit run app.py --server.port 8502`

### Data Requirements:
- Ensure your CSV file has the necessary columns
- Check data types (dates should be in datetime format)
- Remove any missing values if needed

## 📈 Examples

### Sample CSV Format:
```csv
date,category,subcategory,value,value2,size,color,region
2023-01-01,A,X,100,50,25,Red,North
2023-01-02,B,Y,120,60,30,Blue,South
2023-01-03,C,Z,90,45,20,Green,East
```

## 🤝 Contributing

Feel free to contribute by:
- Adding new chart types
- Improving existing visualizations
- Enhancing the user interface
- Adding new data processing features

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the data format requirements
3. Ensure all dependencies are installed

---

**Built with ❤️ using Streamlit, Plotly, and Python** 
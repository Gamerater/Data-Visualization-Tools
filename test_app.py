#!/usr/bin/env python3
"""
Test script for the Data Visualization Tool
"""

import sys
import pandas as pd
import numpy as np

def test_imports():
    """Test if all required modules can be imported"""
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ Plotly imported successfully")
        
        import pandas as pd
        import numpy as np
        print("✅ Pandas and NumPy imported successfully")
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("✅ Matplotlib and Seaborn imported successfully")
        
        import networkx as nx
        print("✅ NetworkX imported successfully")
        
        from wordcloud import WordCloud
        print("✅ WordCloud imported successfully")
        
        import folium
        from streamlit_folium import folium_static
        print("✅ Folium imported successfully")
        
        import altair as alt
        print("✅ Altair imported successfully")
        
        import hvplot.pandas
        import holoviews as hv
        print("✅ HvPlot and HoloViews imported successfully")
        
        from matplotlib_venn import venn3
        print("✅ Matplotlib-Venn imported successfully")
        
        # Test advanced charts import
        from advanced_charts import render_stacked_bar_graph, render_stacked_area_graph, render_multi_set_bar_chart
        print("✅ Advanced charts imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_generation():
    """Test data generation functionality"""
    try:
        # Generate sample data
        np.random.seed(42)
        n = 100
        
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
        
        df = pd.DataFrame(data)
        print(f"✅ Sample data generated successfully: {df.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Data generation error: {e}")
        return False

def test_chart_functions():
    """Test a few chart functions"""
    try:
        from advanced_charts import render_stacked_bar_graph, render_stacked_area_graph, render_multi_set_bar_chart
        
        # Generate test data
        np.random.seed(42)
        n = 50
        data = {
            'date': pd.date_range('2023-01-01', periods=n, freq='D'),
            'category': np.random.choice(['A', 'B', 'C'], n),
            'subcategory': np.random.choice(['X', 'Y'], n),
            'value': np.random.normal(100, 20, n),
            'value2': np.random.normal(50, 10, n),
            'size': np.random.uniform(10, 100, n),
            'color': np.random.choice(['Red', 'Blue', 'Green'], n),
            'region': np.random.choice(['North', 'South'], n),
            'open': np.random.normal(100, 5, n),
            'high': np.random.normal(105, 5, n),
            'low': np.random.normal(95, 5, n),
            'close': np.random.normal(102, 5, n),
            'volume': np.random.randint(1000, 10000, n),
            'text': ['Sample text ' + str(i) for i in range(n)]
        }
        
        df = pd.DataFrame(data)
        
        # Test chart functions
        fig1 = render_stacked_bar_graph(df)
        print("✅ Bar chart function works")
        
        fig2 = render_stacked_area_graph(df)
        print("✅ Area graph function works")
        
        fig3 = render_multi_set_bar_chart(df)
        print("✅ Multi-set bar chart function works")
        
        return True
        
    except Exception as e:
        print(f"❌ Chart function error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Data Visualization Tool...")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Data Generation Tests", test_data_generation),
        ("Chart Function Tests", test_chart_functions)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application should work correctly.")
        print("\n🚀 To run the application:")
        print("   streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
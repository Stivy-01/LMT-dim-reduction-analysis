#!/usr/bin/env python3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
import os

def load_data(file_path=None):
    """Load and preprocess data for visualization"""
    if file_path is None:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=str(Path(__file__).parent.parent.parent / "data")
        )
        if not file_path:
            raise ValueError("No file selected")

    df = pd.read_csv(file_path)
    return df

def create_3d_scatter(df, color_by='ID_Dim_4'):
    """Create an interactive 3D scatter plot of identity space with 4th dimension as color"""
    
    # Check if we have all identity dimensions
    id_dims = ['ID_Dim_1', 'ID_Dim_2', 'ID_Dim_3', 'ID_Dim_4']
    if not all(col in df.columns for col in id_dims):
        raise ValueError("All four identity dimensions not found in the data")
    
    # Create the 3D scatter plot with 4th dimension as color
    fig = px.scatter_3d(
        df,
        x='ID_Dim_1', 
        y='ID_Dim_2', 
        z='ID_Dim_3',
        color='ID_Dim_4',
        color_continuous_scale='viridis',  # You can change this to other scales like 'plasma', 'inferno', etc.
        title='Mouse Identity Space Visualization (4D)',
        labels={
            'ID_Dim_1': 'Identity Dimension 1',
            'ID_Dim_2': 'Identity Dimension 2',
            'ID_Dim_3': 'Identity Dimension 3',
            'ID_Dim_4': 'Identity Dimension 4 (color)'
        }
    )
    
    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis_title='Identity Dimension 1',
            yaxis_title='Identity Dimension 2',
            zaxis_title='Identity Dimension 3'
        ),
        coloraxis_colorbar_title='Identity Dimension 4',
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

def add_trajectory(fig, df, mouse_id):
    """Add trajectory lines for a specific mouse"""
    mouse_data = df[df['mouse_id'] == mouse_id].copy()
    
    # Add line trace with color gradient based on ID_Dim_4
    fig.add_trace(
        go.Scatter3d(
            x=mouse_data['ID_Dim_1'],
            y=mouse_data['ID_Dim_2'],
            z=mouse_data['ID_Dim_3'],
            mode='lines',
            name=f'Mouse {mouse_id} trajectory',
            line=dict(
                width=2,
                dash='dot',
                color=mouse_data['ID_Dim_4'],  # Color the trajectory based on ID_Dim_4
                colorscale='viridis'
            ),
            showlegend=True
        )
    )
    
    return fig

def main():
    """Main function to create and display the visualization"""
    try:
        # Load data
        df = load_data()
        
        # Create 4D scatter plot (3D + color)
        fig = create_3d_scatter(df)
        
        # Add trajectories for each mouse
        if 'mouse_id' in df.columns:
            for mouse_id in df['mouse_id'].unique():
                fig = add_trajectory(fig, df, mouse_id)
        
        # Create output directory if it doesn't exist
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the plot as HTML for interactivity
        output_path = output_dir / "identity_space_4d_visualization.html"
        fig.write_html(str(output_path))
        print(f"\n✅ 4D Visualization saved to: {output_path}")
        
        # Show the plot in browser
        fig.show()
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main() 
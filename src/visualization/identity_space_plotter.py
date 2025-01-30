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
from plotly.subplots import make_subplots

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
    df.name = file_path  # Store the file path for later use
    return df

def create_3d_scatter(df, color_by='ID_Component_4'):
    """Create an interactive 3D scatter plot with 2D projections of identity space"""
    
    # Check if we have all identity dimensions
    id_dims = ['ID_Component_1', 'ID_Component_2', 'ID_Component_3', 'ID_Component_4']
    if not all(col in df.columns for col in id_dims):
        raise ValueError("All four identity components not found in the data")
    
    # Define a list of valid marker symbols
    marker_symbols = ['circle', 'circle-open', 'cross', 'diamond',
                     'diamond-open', 'square', 'square-open', 'x']
    
    # Create figure with subplots (3D view and 2D projections)
    fig = go.Figure()
    
    # Create subplot layout
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'xy'}],
               [{'type': 'xy'}, {'type': 'xy'}]],
        subplot_titles=('3D View', 'Components 1 vs 2',
                       'Components 1 vs 3', 'Components 2 vs 3')
    )
    
    # Add traces for each mouse with different symbols
    for i, mouse_id in enumerate(sorted(df['mouse_id'].unique())):
        mouse_data = df[df['mouse_id'] == mouse_id]
        symbol = marker_symbols[i % len(marker_symbols)]
        
        # 3D scatter plot
        fig.add_trace(
            go.Scatter3d(
                x=mouse_data['ID_Component_1'],
                y=mouse_data['ID_Component_2'],
                z=mouse_data['ID_Component_3'],
                mode='markers+lines',
                name=f'Mouse {mouse_id}',
                marker=dict(
                    size=8,
                    symbol=symbol,
                    color=mouse_data['ID_Component_4'],
                    colorscale='viridis',
                    showscale=True if i == 0 else False,
                    colorbar=dict(title='Identity Component 4'),
                    opacity=0.8 if 'open' in symbol else 1
                ),
                line=dict(
                    color='rgba(100,100,100,0.2)',
                    width=1
                ),
                hovertemplate=(
                    "Mouse ID: %{customdata[0]}<br>" +
                    "Component 1: %{x:.3f}<br>" +
                    "Component 2: %{y:.3f}<br>" +
                    "Component 3: %{z:.3f}<br>" +
                    "Component 4: %{marker.color:.3f}<br>" +
                    "<extra></extra>"
                ),
                customdata=mouse_data[['mouse_id']]
            ),
            row=1, col=1
        )
        
        # 2D projections
        # Components 1 vs 2
        fig.add_trace(
            go.Scatter(
                x=mouse_data['ID_Component_1'],
                y=mouse_data['ID_Component_2'],
                mode='markers+lines',
                name=f'Mouse {mouse_id}',
                marker=dict(
                    size=8,
                    symbol=symbol,
                    color=mouse_data['ID_Component_4'],
                    colorscale='viridis',
                    showscale=False
                ),
                line=dict(color='rgba(100,100,100,0.2)', width=1),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Components 1 vs 3
        fig.add_trace(
            go.Scatter(
                x=mouse_data['ID_Component_1'],
                y=mouse_data['ID_Component_3'],
                mode='markers+lines',
                name=f'Mouse {mouse_id}',
                marker=dict(
                    size=8,
                    symbol=symbol,
                    color=mouse_data['ID_Component_4'],
                    colorscale='viridis',
                    showscale=False
                ),
                line=dict(color='rgba(100,100,100,0.2)', width=1),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Components 2 vs 3
        fig.add_trace(
            go.Scatter(
                x=mouse_data['ID_Component_2'],
                y=mouse_data['ID_Component_3'],
                mode='markers+lines',
                name=f'Mouse {mouse_id}',
                marker=dict(
                    size=8,
                    symbol=symbol,
                    color=mouse_data['ID_Component_4'],
                    colorscale='viridis',
                    showscale=False
                ),
                line=dict(color='rgba(100,100,100,0.2)', width=1),
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title='Mouse Identity Space Visualization',
        scene=dict(
            xaxis_title='ID Component 1',
            yaxis_title='ID Component 2',
            zaxis_title='ID Component 3',
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            zaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            bgcolor='rgba(245,245,245,1)'
        ),
        xaxis2=dict(title='ID Component 1'),
        yaxis2=dict(title='ID Component 2'),
        xaxis3=dict(title='ID Component 1'),
        yaxis3=dict(title='ID Component 3'),
        xaxis4=dict(title='ID Component 2'),
        yaxis4=dict(title='ID Component 3'),
        showlegend=True,
        template='plotly_white',
        legend=dict(
            title='Mouse IDs',
            itemsizing='constant',
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='lightgray',
            borderwidth=1
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=1000,  # Larger figure to accommodate subplots
        width=1200
    )
    
    return fig

def add_trajectory(fig, df, mouse_id):
    """Add trajectory lines for a specific mouse with temporal color gradient"""
    mouse_data = df[df['mouse_id'] == mouse_id].copy()
    
    # Create a temporal color gradient from blue to red
    n_points = len(mouse_data)
    colors = [f'rgba(0,0,255,{i/n_points})' if i < n_points/2 else f'rgba(255,0,0,{i/n_points})' for i in range(n_points)]
    
    # Add line trace showing temporal progression
    fig.add_trace(
        go.Scatter3d(
            x=mouse_data['ID_Component_1'],
            y=mouse_data['ID_Component_2'],
            z=mouse_data['ID_Component_3'],
            mode='lines',
            name=f'Mouse {mouse_id} trajectory',
            line=dict(
                width=6,
                color=colors,  # Blue to red temporal gradient
            ),
            showlegend=False  # Hide trajectory from legend to avoid clutter
        )
    )
    
    return fig

def main():
    """Main function to create and display the visualization"""
    try:
        # Load data
        df = load_data()
        
        # Get the directory of the input CSV file
        input_file_path = Path(df.name if hasattr(df, 'name') else filedialog.askopenfilename())
        output_dir = input_file_path.parent
        base_name = input_file_path.stem
        
        # Create 4D scatter plot (3D + color)
        fig = create_3d_scatter(df)
        
        # Add trajectories for each mouse
        if 'mouse_id' in df.columns:
            for mouse_id in df['mouse_id'].unique():
                fig = add_trajectory(fig, df, mouse_id)
        
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the plot as HTML in the same directory as the input CSV
        output_path = output_dir / f"{base_name}_visualization_{timestamp}.html"
        fig.write_html(str(output_path))
        print(f"\n✅ 4D Visualization saved to: {output_path}")
        
        # Show the plot in browser
        fig.show()
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
import streamlit as st
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import plotly.graph_objects as go
import plotly.express as px
import warnings
import time

# Skyfield setup
from skyfield.api import load, Topos
from scipy.spatial import ConvexHull
from scipy.linalg import svd
from scipy.signal import find_peaks

# --- Configuration & Global Setup ---
# Suppress specific warnings if needed (e.g., from skyfield or matplotlib)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Set page config
st.set_page_config(
    page_title="Inner Solar System Collinearity Visualizer",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stSelectbox label, .stDateInput label {
        color: #1E3A8A;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("Inner Solar System Collinearity Visualizer")
st.markdown("""
    This application visualizes the collinearity of Mercury, Venus, Earth, and the Moon.
    Select a date range and calculate the collinearity index to find interesting alignments.
""")

# Initialize session state for storing calculation results
if 'calculation_results' not in st.session_state:
    st.session_state.calculation_results = {}
    
if 'selected_extrema_time' not in st.session_state:
    st.session_state.selected_extrema_time = None

# Load Skyfield data (do this once at the start)
@st.cache_resource
def load_skyfield_data():
    try:
        ts = load.timescale()
        eph = load('de421.bsp')  # Load ephemeris data
        sun = eph['sun']
        mercury = eph['mercury']
        venus = eph['venus']
        earth = eph['earth']
        moon = eph['moon']
        return ts, eph, sun, mercury, venus, earth, moon
    except Exception as e:
        st.error(f"Failed to load Skyfield ephemeris data (de421.bsp).\nPlease download it or check your internet connection.\nError: {e}")
        st.stop()

# Load data
ts, eph, sun, mercury, venus, earth, moon = load_skyfield_data()

# Plotting Aesthetics
body_colors = {'Mercury': 'gray', 'Venus': 'orange', 'Earth': 'blue', 'Moon': 'lightgray', 'Sun': 'yellow'}
body_sizes = {'Mercury': 5, 'Venus': 8, 'Earth': 8, 'Moon': 3, 'Sun': 15}
# Scale sizes for plotting visibility
plot_body_sizes = {k: v * 20 for k, v in body_sizes.items()}

# --- Core Calculation Logic ---

def calculate_collinearity_index(points_2d):
    """Calculates the collinearity index C for four 2D points."""
    if not isinstance(points_2d, np.ndarray) or points_2d.shape != (4, 2):
        raise ValueError("Input must be a 4x2 NumPy array.")

    # Check for Coincident Points
    pairwise_distances = []
    for i in range(4):
        for j in range(i + 1, 4):
            dist = np.linalg.norm(points_2d[i] - points_2d[j])
            pairwise_distances.append(dist)
    max_dist = np.max(pairwise_distances) if pairwise_distances else 0.0
    epsilon = 1e-9

    if max_dist < epsilon: return 1.0

    # Proximity Component (P) - Area-Based
    try:
        unique_points = np.unique(points_2d, axis=0)
        if unique_points.shape[0] < 3: area = 0.0
        else: hull = ConvexHull(unique_points); area = hull.volume
    except Exception: area = 0.0

    if max_dist > epsilon: P = max(0.0, 1.0 - 2.0 * area / (max_dist**2))
    else: P = 1.0
    if area < epsilon: P = 1.0

    # Evenness Component (E) - Projection-Based
    centroid = np.mean(points_2d, axis=0)
    centered_points = points_2d - centroid
    try:
        U, S, Vt = svd(centered_points.T @ centered_points)
        v1 = U[:, 0]
    except np.linalg.LinAlgError: E = 1.0; C = P * E; return max(0.0, min(1.0, C)) # PCA fails, assume even

    projected_vals = points_2d @ v1
    sorted_indices = np.argsort(projected_vals)
    sorted_proj = projected_vals[sorted_indices]
    proj_range = sorted_proj[-1] - sorted_proj[0]

    if proj_range < epsilon: E = 1.0
    else:
        spacings = np.diff(sorted_proj)
        if len(spacings) != 3: # Handle degenerate projection cases (e.g., points overlap on projection)
            unique_proj = np.unique(sorted_proj)
            if len(unique_proj) <= 2: E = 0.0 # Highly clustered -> uneven
            else: E = 0.5 # Intermediate case, fallback
        else: # Standard case
            mean_spacing = proj_range / 3.0
            if mean_spacing < epsilon: E = 1.0
            else:
                std_dev_spacing = np.std(spacings, ddof=0)
                max_norm_dev = mean_spacing * np.sqrt(2.0)
                if max_norm_dev < epsilon: E = 1.0
                else: norm_dev_ratio = min(std_dev_spacing / max_norm_dev, 1.0); E = max(0.0, 1.0 - norm_dev_ratio)

    C = P * E
    return max(0.0, min(1.0, C))


def get_heliocentric_ecliptic_coords(skyfield_time):
    """Calculates Heliocentric Ecliptic XY coords for Mercury, Venus, Earth, Moon."""
    coords = {}
    earth_pos_vec = sun.at(skyfield_time).observe(earth).ecliptic_xyz().au
    coords['Earth'] = earth_pos_vec[:2]
    coords['Mercury'] = sun.at(skyfield_time).observe(mercury).ecliptic_xyz().au[:2]
    coords['Venus'] = sun.at(skyfield_time).observe(venus).ecliptic_xyz().au[:2]
    moon_geo_pos_vec = earth.at(skyfield_time).observe(moon).ecliptic_xyz().au
    coords['Moon'] = (earth_pos_vec + moon_geo_pos_vec)[:2]
    return np.array([coords['Mercury'], coords['Venus'], coords['Earth'], coords['Moon']])


def calculate_collinearity_over_range(start_dt_utc, end_dt_utc, num_steps=300, progress_bar=None):
    """Calculates collinearity index over a time range, with progress updates."""
    times_utc = pd.date_range(start_dt_utc, end_dt_utc, periods=num_steps)
    times_ts = ts.from_datetimes(times_utc) # Use pandas DatetimeIndex directly

    collinearity_values = []
    for i, t in enumerate(times_ts):
        points_2d = get_heliocentric_ecliptic_coords(t)
        c_index = calculate_collinearity_index(points_2d)
        collinearity_values.append(c_index)
        if progress_bar and (i + 1) % 10 == 0: # Update progress every 10 steps
            progress_bar.progress((i + 1) / num_steps, text=f"Calculating... {(i + 1) / num_steps * 100:.0f}%")

    if progress_bar:
        progress_bar.progress(1.0, text="Calculation complete!")
        
    return times_utc, np.array(collinearity_values)


def find_extrema(times, values, distance_factor=0.02):
    """Finds maxima and minima in the collinearity index data."""
    num_points = len(values)
    distance = max(1, int(num_points * distance_factor)) # Dynamic distance based on range size

    max_indices, _ = find_peaks(values, distance=distance)
    min_indices, _ = find_peaks(-values, distance=distance)

    maxima = pd.DataFrame({'Time': times[max_indices], 'Index': values[max_indices], 'Type': 'Max'})
    minima = pd.DataFrame({'Time': times[min_indices], 'Index': values[min_indices], 'Type': 'Min'})

    extrema_df = pd.concat([maxima, minima]).sort_values('Time').reset_index(drop=True)
    return extrema_df


def plot_index_results(times, values, extrema_df):
    """Creates the index plot with calculation results using Plotly for interactive hover."""
    # Create a Plotly figure
    fig = go.Figure()
    
    # Add main line trace
    fig.add_trace(go.Scatter(
        x=times.to_pydatetime(),
        y=values,
        mode='lines',
        name='Collinearity Index (C)',
        line=dict(color='blue', width=2),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d %H:%M}<br><b>Value:</b> %{y:.4f}<extra></extra>'
    ))
    
    # Add extrema markers if available
    if not extrema_df.empty:
        # Add maxima points
        maxima = extrema_df[extrema_df['Type'] == 'Max']
        if not maxima.empty:
            fig.add_trace(go.Scatter(
                x=maxima['Time'].dt.to_pydatetime(),
                y=maxima['Index'],
                mode='markers',
                name='Maxima',
                marker=dict(color='red', size=10, symbol='triangle-up'),
                hovertemplate='<b>Maximum</b><br><b>Date:</b> %{x|%Y-%m-%d %H:%M}<br><b>Value:</b> %{y:.4f}<extra></extra>'
            ))
        
        # Add minima points
        minima = extrema_df[extrema_df['Type'] == 'Min']
        if not minima.empty:
            fig.add_trace(go.Scatter(
                x=minima['Time'].dt.to_pydatetime(),
                y=minima['Index'],
                mode='markers',
                name='Minima',
                marker=dict(color='green', size=10, symbol='triangle-down'),
                hovertemplate='<b>Minimum</b><br><b>Date:</b> %{x|%Y-%m-%d %H:%M}<br><b>Value:</b> %{y:.4f}<extra></extra>'
            ))
    
    # Update layout
    fig.update_layout(
        title="Collinearity Index (C) vs. Time",
        xaxis_title="Time (UTC)",
        yaxis_title="Index Value (0 to 1)",
        yaxis=dict(range=[-0.05, 1.05]),
        hovermode='closest',
        legend=dict(font=dict(size=10)),
        margin=dict(l=20, r=20, t=40, b=20),
        height=500,
        grid=dict(rows=1, columns=1)
    )
    
    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    
    return fig


def plot_system_view(time_dt_utc):
    """Creates the 2D solar system view for the given UTC time using Plotly for interactivity."""
    # Convert pandas Timestamp or python datetime to Skyfield time object
    time_ts = ts.from_datetime(time_dt_utc)

    points_2d = get_heliocentric_ecliptic_coords(time_ts)
    # Recalculate index for this specific time for accuracy in title
    current_c_index = calculate_collinearity_index(points_2d)

    # Create a Plotly figure
    fig = go.Figure()
    
    # Add Sun at the center
    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode='markers',
        name='Sun',
        marker=dict(
            size=30,
            color=body_colors['Sun'],
            line=dict(width=1, color='black')
        ),
        hovertemplate='<b>Sun</b><br>Position: (0, 0)<extra></extra>'
    ))

    # Add planets and Moon
    labels = ['Mercury', 'Venus', 'Earth', 'Moon']
    for i, label in enumerate(labels):
        fig.add_trace(go.Scatter(
            x=[points_2d[i, 0]],
            y=[points_2d[i, 1]],
            mode='markers',
            name=label,
            marker=dict(
                size=plot_body_sizes[label]/10,  # Adjust size for Plotly
                color=body_colors[label],
                line=dict(width=1, color='black')
            ),
            hovertemplate=f'<b>{label}</b><br>X: %{{x:.3f}} AU<br>Y: %{{y:.3f}} AU<extra></extra>'
        ))
    
    # Determine plot limits dynamically
    max_range = np.max(np.abs(points_2d)) * 1.2  # Add some padding
    max_range = max(max_range, 1.5)  # Ensure minimum range (e.g., Earth's orbit)
    
    # Update layout
    fig.update_layout(
        title=f"System at {time_dt_utc.strftime('%Y-%m-%d %H:%M')} UTC<br>C = {current_c_index:.4f}",
        xaxis_title="X (AU) - Ecliptic Plane",
        yaxis_title="Y (AU) - Ecliptic Plane",
        xaxis=dict(range=[-max_range, max_range], zeroline=True, zerolinewidth=1, zerolinecolor='grey'),
        yaxis=dict(range=[-max_range, max_range], zeroline=True, zerolinewidth=1, zerolinecolor='grey', scaleanchor="x", scaleratio=1),
        showlegend=True,
        legend=dict(font=dict(size=10)),
        hovermode='closest',
        margin=dict(l=20, r=20, t=60, b=20),
        height=500,
        grid=dict(rows=1, columns=1)
    )
    
    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    
    return fig


# --- Streamlit UI Layout ---

# Sidebar for controls
with st.sidebar:
    st.header("Calculation Controls")
    
    # Date selection
    st.subheader("Date Range")
    
    # Set default dates (+/- 6 months)
    today = datetime.date.today()
    default_start = today - datetime.timedelta(days=180)
    default_end = today + datetime.timedelta(days=180)
    
    start_date = st.date_input("Start Date", value=default_start)
    end_date = st.date_input("End Date", value=default_end)
    
    # Number of steps slider
    num_steps = st.slider("Resolution (data points)", min_value=100, max_value=1000, value=500, step=100,
                         help="Higher values give more detailed results but take longer to calculate")
    
    # Calculate button
    calculate_button = st.button("Calculate & Plot", type="primary", use_container_width=True)
    
    # Information section
    st.markdown("---")
    st.subheader("About")
    st.markdown("""
    This application visualizes the collinearity of Mercury, Venus, Earth, and the Moon in the solar system.
    
    The collinearity index (C) ranges from 0 to 1:
    - 1.0 = Perfect alignment
    - 0.0 = No alignment
    
    Select a date range and click Calculate to find interesting alignments.
    """)

# Main content area - use columns for layout
col1, col2 = st.columns([3, 2])

# Run calculation when button is clicked
if calculate_button:
    if start_date >= end_date:
        st.error("Start date must be before end date.")
    else:
        try:
            # Convert dates to timezone-aware datetimes for Skyfield
            start_dt_utc = pd.Timestamp(start_date).tz_localize('UTC')
            end_dt_utc = pd.Timestamp(end_date).tz_localize('UTC')
            
            # Show progress bar
            progress_bar = st.progress(0, text="Initializing calculation...")
            
            # Run calculation
            times, values = calculate_collinearity_over_range(
                start_dt_utc, end_dt_utc, num_steps, progress_bar
            )
            
            # Find extrema
            extrema_df = find_extrema(times, values)
            
            # Store results in session state
            st.session_state.calculation_results = {
                'times': times,
                'values': values,
                'extrema_df': extrema_df
            }
            
            # Reset selected extrema
            st.session_state.selected_extrema_time = None
            
            # Add a small delay to ensure progress bar is seen
            time.sleep(0.5)
            
        except Exception as e:
            st.error(f"An error occurred during calculation: {e}")

# Display results if available
if st.session_state.calculation_results and 'times' in st.session_state.calculation_results:
    results = st.session_state.calculation_results
    display_df = pd.DataFrame() # Initialize display_df
    
    with col1:
        # Plot the collinearity index
        st.subheader("Collinearity Index Over Time")
        index_fig = plot_index_results(results['times'], results['values'], results['extrema_df'])
        st.plotly_chart(index_fig, use_container_width=True)
        
        # Display extrema in a table
        st.subheader("Extreme Collinearity Events")
        if results['extrema_df'].empty:
            st.info("No significant extrema found in the selected date range.")
        else:
            # Format the extrema dataframe for display
            display_df = results['extrema_df'].copy()
            display_df['Time (UTC)'] = display_df['Time'].dt.strftime('%Y-%m-%d %H:%M')
            display_df['Collinearity Index'] = display_df['Index'].round(4)
            display_df['Event Type'] = display_df['Type']
            
            # Create a styled dataframe with color coding for Max and Min events
            def color_event_type(val):
                color = 'red' if val == 'Max' else 'green' if val == 'Min' else ''
                return f'color: {color}; font-weight: bold'
            
            # Apply styling to the dataframe
            styled_df = display_df[['Time (UTC)', 'Collinearity Index', 'Event Type']].style.applymap(
                color_event_type, subset=['Event Type']
            )
            
            # Display as an interactive table
            st.write("Click on a row to view the solar system configuration:")
            selected_rows = st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Event Type": st.column_config.TextColumn(
                        "Event Type",
                        help="Max (red) = maximum collinearity, Min (green) = minimum collinearity"
                    )
                }
            ).selected_rows
            
            # Handle row selection from the dataframe
            if selected_rows and isinstance(selected_rows, list) and len(selected_rows) > 0:
                try:
                    # In Streamlit, selected_rows returns a list of dictionaries with the row data
                    selected_row_data = selected_rows[0]  # Get the first selected row
                    
                    # Debug information to see what's in the selected row
                    with st.expander("Debug info (click to expand)", expanded=False):
                        st.write("Selected row data:", selected_row_data)
                    
                    # Extract the time string from the selected row
                    if 'Time (UTC)' in selected_row_data:
                        selected_time_str = selected_row_data['Time (UTC)']
                        
                        # Find the matching index in the original dataframe
                        matching_rows = display_df[display_df['Time (UTC)'] == selected_time_str]
                        
                        if not matching_rows.empty:
                            selected_index = matching_rows.index[0]
                            # Update the session state with the selected time
                            st.session_state.selected_extrema_time = results['extrema_df'].loc[selected_index, 'Time']
                            
                            # Force a rerun to update the visualization immediately
                            st.rerun()
                except (TypeError, AttributeError, IndexError) as e:
                    st.warning(f"Could not process selection: {e}. Please try clicking again or use the navigation controls.")
                    # Don't stop execution, let the user try again or use the navigation controls
    
    with col2:
        # System view plot
        st.subheader("Solar System View")
       
        # Display system view if a time is selected
        if st.session_state.selected_extrema_time is not None:
            # Plot the system at the selected time
            system_fig = plot_system_view(st.session_state.selected_extrema_time)
            st.plotly_chart(system_fig, use_container_width=True)

            # Show the collinearity value
            time_ts = ts.from_datetime(st.session_state.selected_extrema_time)
            points_2d = get_heliocentric_ecliptic_coords(time_ts)
            c_index = calculate_collinearity_index(points_2d)

            st.metric(
                "Collinearity Index",
                f"{c_index:.4f}",
                delta=None,
                delta_color="normal"
            )

            # Add explanation of what we're seeing (Note: This is similar to the markdown at line 480)
            # Consider removing one of them if redundant after testing.

        else:
            st.info("Select an event from the table or use the navigation controls to view the solar system configuration.")

        # Navigation controls for extrema - moved below the system view
        if not results['extrema_df'].empty:
            st.subheader("Navigation Controls")

            # Determine current index or default to first item
            current_index = 0
            if st.session_state.selected_extrema_time is not None:
                matching_rows = results['extrema_df'][results['extrema_df']['Time'] == st.session_state.selected_extrema_time]
                if not matching_rows.empty:
                    current_index = matching_rows.index[0]

            total_extrema = len(results['extrema_df'])

            # Create a row with three columns for the navigation buttons
            nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])

            # Previous button
            with nav_col1:
                if st.button("⬅️ Previous", use_container_width=True):
                    # Go to previous extrema (wrap around if at the beginning)
                    prev_index = (current_index - 1) % total_extrema
                    st.session_state.selected_extrema_time = results['extrema_df'].loc[prev_index, 'Time']
                    st.rerun()

            # Next button
            with nav_col2:
                if st.button("➡️ Next", use_container_width=True):
                    # Go to next extrema (wrap around if at the end)
                    next_index = (current_index + 1) % total_extrema
                    st.session_state.selected_extrema_time = results['extrema_df'].loc[next_index, 'Time']
                    st.rerun()

            # Reset button
            with nav_col3:
                if st.button("🔄 Reset", use_container_width=True):
                    st.session_state.selected_extrema_time = None
                    st.rerun()

            # Dropdown for direct selection on a new row
            selected_extrema = st.selectbox(
                "Select an event:",
                options=display_df.index,
                index=int(current_index),
                    format_func=lambda i: f"{display_df.loc[i, 'Event Type']} at {display_df.loc[i, 'Time (UTC)']} (C={display_df.loc[i, 'Collinearity Index']})"
                )

            if selected_extrema is not None and selected_extrema != current_index:
                st.session_state.selected_extrema_time = results['extrema_df'].loc[selected_extrema, 'Time']
                st.rerun()
            
            st.info("""
            This view shows the positions of Mercury, Venus, Earth, and the Moon from above the ecliptic plane.
            The Sun is at the center (0,0). Perfect alignment would show all bodies in a straight line.
            """)

else:
    # Initial state - no calculation yet
    with col1:
        st.info("Select a date range and click 'Calculate & Plot' to begin.")
    
    with col2:
        st.info("Solar system view will appear here after calculation.")
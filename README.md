# Inner Solar System Collinearity Visualizer (Streamlit Version)

A Streamlit web application that visualizes and analyzes the collinearity of Mercury, Venus, Earth, and the Moon in the inner solar system. This tool helps researchers and astronomy enthusiasts identify and study periods when these celestial bodies align in interesting configurations using interactive plots.

## Features

- **Web-Based Interface**: Accessible via a web browser using Streamlit.
- **Interactive Date Selection**: Choose any time period to analyze collinearity patterns using a calendar interface.
- **Collinearity Index Calculation**: Computes a normalized index (0-1) based on:
  - **Proximity (P)**: How close the bodies are to forming a straight line (area-based).
  - **Evenness (E)**: How evenly the bodies are distributed along that line (projection-based).
- **Extrema Detection**: Automatically identifies maxima (highest collinearity) and minima (lowest collinearity) within the selected range.
- **Interactive Visual Analysis**:
  - Time-series plot of the collinearity index using Plotly, with hover details.
  - Interactive 2D solar system view (heliocentric ecliptic plane) using Plotly, showing planetary positions at selected times.
  - Table displaying detected extrema events (maxima/minima). Clicking a row updates the solar system view to that specific time.
- **Responsive UI**: Adapts to different screen sizes.
- **Progress Indicator**: Shows calculation progress for longer date ranges.

## Installation

1.  Ensure you have Python 3.7+ installed.
2.  Clone or download this repository.
3.  Navigate to the repository directory in your terminal.
4.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

5.  The application requires the Skyfield ephemeris data file `de421.bsp`. It will attempt to download this automatically on the first run if you have an internet connection. Alternatively, you can download it manually from [JPL's website](https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/) and place it in the application's root directory.

## Usage

1.  Run the Streamlit application from your terminal:

    ```bash
    streamlit run solar_collinearity_streamlit.py
    ```

2.  The application will open in your default web browser.
3.  Use the sidebar to select a start and end date for your analysis period.
4.  Adjust the "Resolution" slider to control the number of data points calculated (higher means more detail but longer calculation time).
5.  Click "Calculate & Plot" to process the data.
6.  Examine the interactive collinearity index plot. Hover over the line or points for details.
7.  Review the "Extreme Collinearity Events" table. Click on any row (maximum or minimum) to view the solar system configuration at that specific time in the "Solar System View" plot.
8.  Use the Plotly chart tools (hover icons) to zoom, pan, and save the plots as images.

## Understanding the Collinearity Index

The collinearity index (C) ranges from 0 to 1, where:
- **1.0**: Perfect collinearity (all bodies in a perfectly straight line and evenly spaced).
- **0.0**: Maximum non-collinearity.

The index is calculated as the product of two components:
- **Proximity (P)**: Measures how close the points are to lying on a single line. Calculated based on the ratio of the convex hull area of the points to the square of the maximum pairwise distance. A smaller area relative to the overall spread results in a higher P value.
- **Evenness (E)**: Measures how uniformly the points are distributed along the principal axis (the line of best fit). Calculated based on the standard deviation of the spacings between points when projected onto this axis. More even spacing results in a higher E value.

## Technical Details

The application uses:
- **Streamlit**: For the web application framework and UI components.
- **Skyfield**: For high-precision astronomical calculations (planetary positions).
- **NumPy/SciPy**: For numerical operations, linear algebra (SVD for PCA), convex hull calculation, and peak finding.
- **Pandas**: For time series data management and manipulation.
- **Plotly**: For creating interactive visualizations (time series and scatter plots).

## License

This project is available for educational and research purposes.

## Acknowledgments

- [Skyfield](https://rhodesmill.org/skyfield/) for astronomical calculations.
- [NASA JPL](https://ssd.jpl.nasa.gov/) for ephemeris data.
- [Streamlit](https://streamlit.io/) for the web framework.
- [Plotly](https://plotly.com/) for interactive plotting.
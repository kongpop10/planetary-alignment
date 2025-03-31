# Inner Solar System Collinearity Visualizer

A Python application that visualizes and analyzes the collinearity of Mercury, Venus, Earth, and the Moon in the inner solar system. This tool helps researchers and astronomy enthusiasts identify and study periods when these celestial bodies align in interesting configurations.

## Features

- **Interactive Date Selection**: Choose any time period to analyze collinearity patterns
- **Collinearity Index Calculation**: Computes a normalized index (0-1) based on:
  - **Proximity (P)**: How close the bodies are to forming a straight line
  - **Evenness (E)**: How evenly the bodies are distributed along that line
- **Extrema Detection**: Automatically identifies maxima and minima in collinearity
- **Visual Analysis**:
  - Time-series plot of collinearity index
  - Interactive 2D solar system view at any selected time
  - Highlighted extrema points for easy identification
- **Multi-threaded Calculation**: Performs intensive calculations without freezing the UI

## Installation

1. Ensure you have Python 3.6+ installed
2. Clone or download this repository
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. The application requires the Skyfield ephemeris data file `de421.bsp`. It will attempt to download this automatically on first run, or you can download it manually from [JPL's website](https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/) and place it in the application directory.

## Usage

1. Run the application:

```bash
python solar_collinearity_gui.py
```

2. Select a start and end date for your analysis period
3. Click "Calculate & Plot" to process the data
4. Examine the collinearity index plot to identify interesting periods
5. Click on any extrema point in the list to view the solar system configuration at that time
6. Use the matplotlib navigation tools to zoom, pan, and save plots

## Understanding the Collinearity Index

The collinearity index (C) ranges from 0 to 1, where:
- **1.0**: Perfect collinearity (all bodies in a perfectly straight line)
- **0.0**: Maximum non-collinearity

The index is calculated as the product of two components:
- **Proximity (P)**: Based on the ratio of convex hull area to maximum pairwise distance
- **Evenness (E)**: Based on the standard deviation of projected point spacings

## Technical Details

The application uses:
- **Skyfield**: For high-precision astronomical calculations
- **NumPy/SciPy**: For mathematical operations and peak finding
- **Pandas**: For time series data management
- **Matplotlib**: For visualization
- **Tkinter**: For the GUI framework

## Advanced Usage

- The system view shows the heliocentric ecliptic coordinates (X-Y plane)
- Adjust the code parameters in `calculate_collinearity_over_range()` to change resolution
- Modify `find_extrema()` parameters to adjust sensitivity to local maxima/minima

## License

This project is available for educational and research purposes.

## Acknowledgments

- [Skyfield](https://rhodesmill.org/skyfield/) for astronomical calculations
- [NASA JPL](https://ssd.jpl.nasa.gov/) for ephemeris data
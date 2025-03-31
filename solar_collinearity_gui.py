#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry # Nicer date picker
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import warnings
import threading # To prevent GUI freezing during calculation

# Skyfield setup
from skyfield.api import load, Topos
from scipy.spatial import ConvexHull
from scipy.linalg import svd
from scipy.signal import find_peaks

# --- Configuration & Global Setup ---
# Suppress specific warnings if needed (e.g., from skyfield or matplotlib)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Load Skyfield data (do this once at the start)
print("Loading Skyfield ephemeris data (de421.bsp)...")
try:
    ts = load.timescale()
    eph = load('de421.bsp') # Load ephemeris data
    sun = eph['sun']
    mercury = eph['mercury']
    venus = eph['venus']
    earth = eph['earth']
    moon = eph['moon']
    print("Skyfield data loaded successfully.")
except Exception as e:
    messagebox.showerror("Error", f"Failed to load Skyfield ephemeris data (de421.bsp).\nPlease download it or check your internet connection.\nError: {e}")
    exit()

# Define bodies for collinearity calculation
collinearity_bodies = {
    'Mercury': mercury,
    'Venus': venus,
    'Earth': earth,
}
body_names_for_calc = ['Mercury', 'Venus', 'Earth', 'Moon'] # Used in position fetching

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


def calculate_collinearity_over_range(start_dt_utc, end_dt_utc, num_steps=300, progress_callback=None):
    """Calculates collinearity index over a time range, with progress updates."""
    times_utc = pd.date_range(start_dt_utc, end_dt_utc, periods=num_steps)
    times_ts = ts.from_datetimes(times_utc) # Use pandas DatetimeIndex directly

    collinearity_values = []
    for i, t in enumerate(times_ts):
        points_2d = get_heliocentric_ecliptic_coords(t)
        c_index = calculate_collinearity_index(points_2d)
        collinearity_values.append(c_index)
        if progress_callback and (i + 1) % 10 == 0: # Update progress every 10 steps
             progress_callback((i + 1) / num_steps)

    if progress_callback: progress_callback(1.0) # Final progress update
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


# --- GUI Application Class ---

class CollinearityApp:
    def __init__(self, master):
        self.master = master
        master.title("Inner Solar System Collinearity Visualizer")
        master.geometry("1000x750") # Adjusted size

        self.calculation_results = {} # Store times, values, extrema_df
        self.extrema_data_map = {} # Map listbox index to datetime

        # --- Style ---
        style = ttk.Style()
        style.configure("TLabel", padding=3)
        style.configure("TButton", padding=3)
        style.configure("TFrame", padding=5)

        # --- Main Frames ---
        control_frame = ttk.Frame(master, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        plot_frame = ttk.Frame(master, padding="5")
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Split plot frame into two columns
        index_plot_frame = ttk.Frame(plot_frame)
        index_plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        right_frame = ttk.Frame(plot_frame) # Frame for extrema list and system plot
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        extrema_frame = ttk.Frame(right_frame)
        extrema_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        system_plot_frame = ttk.Frame(right_frame)
        system_plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)

        status_frame = ttk.Frame(master, padding="5")
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)


        # --- Control Widgets ---
        ttk.Label(control_frame, text="Start Date:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.start_date_entry = DateEntry(control_frame, width=12, background='darkblue',
                                          foreground='white', borderwidth=2, date_pattern='yyyy-mm-dd')
        self.start_date_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(control_frame, text="End Date:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.end_date_entry = DateEntry(control_frame, width=12, background='darkblue',
                                        foreground='white', borderwidth=2, date_pattern='yyyy-mm-dd')
        self.end_date_entry.grid(row=0, column=3, padx=5, pady=5)

        self.calc_button = ttk.Button(control_frame, text="Calculate & Plot", command=self.run_calculation_thread)
        self.calc_button.grid(row=0, column=4, padx=10, pady=5)

        # Set default dates (+/- 6 months)
        today = datetime.date.today()
        default_start = today - datetime.timedelta(days=180)
        default_end = today + datetime.timedelta(days=180)
        self.start_date_entry.set_date(default_start)
        self.end_date_entry.set_date(default_end)


        # --- Index Plot Area ---
        ttk.Label(index_plot_frame, text="Collinearity Index (C) vs. Time", font=("Arial", 10, "bold")).pack(pady=2)
        self.fig_index = Figure(figsize=(6, 4), dpi=100)
        self.ax_index = self.fig_index.add_subplot(111)
        self.canvas_index = FigureCanvasTkAgg(self.fig_index, master=index_plot_frame)
        self.canvas_index_widget = self.canvas_index.get_tk_widget()
        self.canvas_index_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar_index = NavigationToolbar2Tk(self.canvas_index, index_plot_frame, pack_toolbar=False)
        self.toolbar_index.update()
        self.toolbar_index.pack(side=tk.BOTTOM, fill=tk.X)
        self._initialize_index_plot()


        # --- Extrema List Area ---
        ttk.Label(extrema_frame, text="Extreme Collinearity Events:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.extrema_listbox = tk.Listbox(extrema_frame, height=8, width=50, exportselection=False) # Allow selecting while other listbox might be active
        self.extrema_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        extrema_scrollbar = ttk.Scrollbar(extrema_frame, orient=tk.VERTICAL, command=self.extrema_listbox.yview)
        extrema_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.extrema_listbox.config(yscrollcommand=extrema_scrollbar.set)
        self.extrema_listbox.bind('<<ListboxSelect>>', self.on_extrema_select)


        # --- System Plot Area ---
        ttk.Label(system_plot_frame, text="Solar System View", font=("Arial", 10, "bold")).pack(pady=2)
        self.fig_system = Figure(figsize=(5, 5), dpi=100)
        self.ax_system = self.fig_system.add_subplot(111)
        self.canvas_system = FigureCanvasTkAgg(self.fig_system, master=system_plot_frame)
        self.canvas_system_widget = self.canvas_system.get_tk_widget()
        self.canvas_system_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar_system = NavigationToolbar2Tk(self.canvas_system, system_plot_frame, pack_toolbar=False)
        self.toolbar_system.update()
        self.toolbar_system.pack(side=tk.BOTTOM, fill=tk.X)
        self._initialize_system_plot()


        # --- Status Bar ---
        self.status_label = ttk.Label(status_frame, text="Ready. Select dates and click Calculate.", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.progress_bar = ttk.Progressbar(status_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress_bar.pack(side=tk.RIGHT, padx=5)


    def _initialize_index_plot(self):
        self.ax_index.clear()
        self.ax_index.set_title("Collinearity Index (C) vs. Time")
        self.ax_index.set_xlabel("Time (UTC)")
        self.ax_index.set_ylabel("Index Value (0 to 1)")
        self.ax_index.set_ylim(-0.05, 1.05)
        self.ax_index.grid(True, linestyle='--', alpha=0.6)
        self.fig_index.tight_layout()
        self.canvas_index.draw()

    def _initialize_system_plot(self):
        self.ax_system.clear()
        self.ax_system.set_title("Solar System View at Selected Time")
        self.ax_system.set_xlabel("X (AU) - Ecliptic Plane")
        self.ax_system.set_ylabel("Y (AU) - Ecliptic Plane")
        self.ax_system.axhline(0, color='grey', lw=0.5)
        self.ax_system.axvline(0, color='grey', lw=0.5)
        self.ax_system.set_aspect('equal', adjustable='box')
        # Set initial dummy limits
        self.ax_system.set_xlim(-1.5, 1.5)
        self.ax_system.set_ylim(-1.5, 1.5)
        self.ax_system.grid(True, linestyle='--', alpha=0.6)
        self.fig_system.tight_layout()
        self.canvas_system.draw()

    def _update_status(self, message, progress=None):
        self.status_label.config(text=message)
        if progress is not None:
            self.progress_bar['value'] = progress * 100
        self.master.update_idletasks() # Force UI update

    def _set_busy_state(self, busy):
        if busy:
            self.calc_button.config(state=tk.DISABLED)
            self.master.config(cursor="watch")
            self._update_status("Calculating...", 0)
        else:
            self.calc_button.config(state=tk.NORMAL)
            self.master.config(cursor="")
            self._update_status("Calculation complete.", 100)
            # Keep progress bar full briefly
            self.master.after(2000, lambda: self.progress_bar.config(value=0))
            self.master.after(2000, lambda: self._update_status("Ready."))


    def run_calculation_thread(self):
        """Runs the main calculation in a separate thread to avoid freezing the GUI."""
        start_date = self.start_date_entry.get_date()
        end_date = self.end_date_entry.get_date()

        if start_date >= end_date:
            messagebox.showerror("Input Error", "Start date must be before end date.")
            return

        # Convert dates to timezone-aware datetimes for Skyfield
        # Assume dates are midnight UTC for simplicity
        try:
            start_dt_utc = pd.Timestamp(start_date).tz_localize('UTC')
            end_dt_utc = pd.Timestamp(end_date).tz_localize('UTC')
        except Exception as e:
             messagebox.showerror("Date Error", f"Error processing dates: {e}")
             return

        self._set_busy_state(True)
        self._initialize_index_plot() # Clear old plots
        self._initialize_system_plot()
        self.extrema_listbox.delete(0, tk.END) # Clear listbox

        # Define the target function for the thread
        def calculation_task():
            try:
                num_steps = 500 # Or make this configurable
                times, values = calculate_collinearity_over_range(
                    start_dt_utc, end_dt_utc, num_steps,
                    progress_callback=lambda p: self.master.after(0, self._update_status, f"Calculating... {p*100:.0f}%", p)
                )

                extrema_df = find_extrema(times, values)

                # Store results safely
                self.calculation_results = {'times': times, 'values': values, 'extrema_df': extrema_df}

                # Schedule UI updates back on the main thread
                self.master.after(0, self._plot_index_results)
                self.master.after(0, self._update_extrema_list)
                self.master.after(0, self._set_busy_state, False) # Pass False argument

            except Exception as e:
                # Show error in the main thread
                self.master.after(0, messagebox.showerror, "Calculation Error", f"An error occurred: {e}")
                self.master.after(0, self._set_busy_state, False)

        # Create and start the thread
        calc_thread = threading.Thread(target=calculation_task, daemon=True) # Daemon so it exits if main window closes
        calc_thread.start()


    def _plot_index_results(self):
        """Updates the index plot with calculation results. Must run in main thread."""
        if not self.calculation_results or 'times' not in self.calculation_results:
            return

        times = self.calculation_results['times']
        values = self.calculation_results['values']
        extrema_df = self.calculation_results['extrema_df']

        self.ax_index.clear() # Clear previous plot content
        self.ax_index.plot(times.to_pydatetime(), values, label='Collinearity Index (C)', zorder=1)

        # Plot extrema markers
        if not extrema_df.empty:
            maxima = extrema_df[extrema_df['Type'] == 'Max']
            minima = extrema_df[extrema_df['Type'] == 'Min']
            self.ax_index.scatter(maxima['Time'].dt.to_pydatetime(), maxima['Index'], color='red', marker='^', label='Maxima', zorder=2)
            self.ax_index.scatter(minima['Time'].dt.to_pydatetime(), minima['Index'], color='green', marker='v', label='Minima', zorder=2)

        # Reset labels, title, grid, legend
        self.ax_index.set_title(f"Collinearity Index (C) vs. Time")
        self.ax_index.set_xlabel("Time (UTC)")
        self.ax_index.set_ylabel("Index Value (0 to 1)")
        self.ax_index.set_ylim(-0.05, 1.05)
        self.ax_index.grid(True, linestyle='--', alpha=0.6)
        self.ax_index.legend(fontsize='small')
        self.fig_index.autofmt_xdate() # Improve date formatting
        self.fig_index.tight_layout()
        self.canvas_index.draw()


    def _update_extrema_list(self):
        """Populates the listbox with found extrema. Must run in main thread."""
        self.extrema_listbox.delete(0, tk.END)
        self.extrema_data_map.clear()

        if not self.calculation_results or 'extrema_df' not in self.calculation_results:
            return

        extrema_df = self.calculation_results['extrema_df']

        if extrema_df.empty:
            self.extrema_listbox.insert(tk.END, " No significant extrema found.")
            self.extrema_listbox.config(state=tk.DISABLED)
        else:
            self.extrema_listbox.config(state=tk.NORMAL)
            for index, row in extrema_df.iterrows():
                # Ensure time is in UTC before formatting
                time_utc = row['Time'].tz_convert('UTC')
                label = f"{row['Type']} at {time_utc.strftime('%Y-%m-%d %H:%M')} (C={row['Index']:.4f})"
                self.extrema_listbox.insert(tk.END, label)
                # Map the listbox index (current end) to the actual datetime object
                self.extrema_data_map[self.extrema_listbox.size() - 1] = time_utc


    def on_extrema_select(self, event):
        """Callback when an item is selected in the extrema listbox."""
        selection = self.extrema_listbox.curselection()
        if not selection:
            return # No selection

        selected_listbox_index = selection[0]

        # Retrieve the datetime object using the map
        if selected_listbox_index in self.extrema_data_map:
            selected_time_utc = self.extrema_data_map[selected_listbox_index]
            self._plot_system_view(selected_time_utc)
        else:
            # Fallback or error handling if map is inconsistent
            print(f"Warning: Could not find time data for listbox index {selected_listbox_index}")
            self._initialize_system_plot() # Clear plot if data missing


    def _plot_system_view(self, time_dt_utc):
        """Plots the 2D solar system view for the given UTC time."""
        self.ax_system.clear() # Clear previous plot

        # Convert pandas Timestamp or python datetime to Skyfield time object
        time_ts = ts.from_datetime(time_dt_utc)

        points_2d = get_heliocentric_ecliptic_coords(time_ts)
        # Recalculate index for this specific time for accuracy in title
        current_c_index = calculate_collinearity_index(points_2d)

        # Plot Sun
        self.ax_system.scatter(0, 0, s=plot_body_sizes['Sun'], c=body_colors['Sun'], label='Sun', zorder=1)

        # Plot bodies
        labels = ['Mercury', 'Venus', 'Earth', 'Moon']
        for i, label in enumerate(labels):
            self.ax_system.scatter(points_2d[i, 0], points_2d[i, 1],
                                   s=plot_body_sizes[label], c=body_colors[label], label=label, zorder=2)

        # Determine plot limits dynamically
        max_range = np.max(np.abs(points_2d)) * 1.2 # Add some padding
        max_range = max(max_range, 1.5) # Ensure minimum range (e.g., Earth's orbit)
        self.ax_system.set_xlim(-max_range, max_range)
        self.ax_system.set_ylim(-max_range, max_range)

        # Reset labels, title, grid, legend
        self.ax_system.set_xlabel("X (AU) - Ecliptic Plane")
        self.ax_system.set_ylabel("Y (AU) - Ecliptic Plane")
        self.ax_system.set_title(f"System at {time_dt_utc.strftime('%Y-%m-%d %H:%M')} UTC\nC = {current_c_index:.4f}", fontsize=9)
        self.ax_system.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        self.ax_system.set_aspect('equal', adjustable='box')
        self.ax_system.grid(True, linestyle='--', alpha=0.6)
        self.fig_system.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
        self.canvas_system.draw()


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = CollinearityApp(root)
    root.mainloop()
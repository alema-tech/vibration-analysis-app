import streamlit as st
import numpy as np
import pandas as pd
import pywt  # Corrected import for PyWavelets
import plotly.graph_objs as go
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis
import requests
import time

# Set Streamlit page configuration
st.set_page_config(
    page_title="Advanced Vibration Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Function to fetch data from NodeMCU
def fetch_data(ip_address):
    """Fetches JSON data from NodeMCU at the given IP address."""
    try:
        response = requests.get(f"http://{ip_address}/data", timeout=5)
        response.raise_for_status()
        return response.json() if response.status_code == 200 else None
    except requests.RequestException as e:
        st.error(f"Error fetching data: {e}")
    return None


# Data processing function
def process_vibration_data(raw_data, lowcut_freq, highcut_freq, wavelet_type, decomp_level, fs=1000):
    """Processes accelerometer data by filtering and wavelet transformation."""
    df = pd.DataFrame(raw_data)

    if not all(col in df.columns for col in ["Time", "X", "Y", "Z"]):
        st.error("Data format is incorrect.")
        return None

    time_data, x_data, y_data, z_data = df["Time"].values, df["X"].values, df["Y"].values, df["Z"].values

    x_filtered_data = apply_filter(x_data, lowcut_freq, highcut_freq, fs)
    y_filtered_data = apply_filter(y_data, lowcut_freq, highcut_freq, fs)
    z_filtered_data = apply_filter(z_data, lowcut_freq, highcut_freq, fs)

    # Scientific statistics
    stats = {
        'X': {"RMS": np.sqrt(np.mean(x_filtered_data ** 2)), "Kurtosis": kurtosis(x_filtered_data)},
        'Y': {"RMS": np.sqrt(np.mean(y_filtered_data ** 2)), "Kurtosis": kurtosis(y_filtered_data)},
        'Z': {"RMS": np.sqrt(np.mean(z_filtered_data ** 2)), "Kurtosis": kurtosis(z_filtered_data)},
    }
    st.write("### Data Summary Statistics")
    st.write(df.describe())
    st.write("### Scientific Metrics (RMS and Kurtosis)")
    st.json(stats)

    return time_data, x_filtered_data, y_filtered_data, z_filtered_data


# Filter function
def apply_filter(data_series, low_freq, high_freq, fs, order=5):
    """Applies low-pass and high-pass Butterworth filters to the data."""
    b_low, a_low = butter(order, low_freq / (0.5 * fs), btype='low')
    low_passed = filtfilt(b_low, a_low, data_series)
    b_high, a_high = butter(order, high_freq / (0.5 * fs), btype='high')
    return filtfilt(b_high, a_high, low_passed)


# Plotting function for wavelet analysis
def plot_wavelet_analysis(time_series, signal_series, title_text, wavelet_name, level, graph_type="3D"):
    """Plots Discrete Wavelet Transform (DWT) analysis results in 2D or 3D with Plotly."""
    coeffs = pywt.wavedec(signal_series, wavelet_name, level=level)
    num_levels = len(coeffs) - 1
    time_indices = [np.linspace(0, len(signal_series), len(c)) for c in coeffs]

    fig = go.Figure()
    if graph_type == "3D":
        for lvl in range(1, num_levels + 1):
            t_idx = time_indices[lvl]
            amplitude = np.abs(coeffs[lvl])
            fig.add_trace(go.Scatter3d(
                x=t_idx, y=[lvl] * len(t_idx), z=amplitude,
                mode='lines', name=f'Level {lvl}',
                line=dict(width=2)
            ))
        fig.update_layout(
            title=f'{title_text} - Time vs Level vs Amplitude',
            scene=dict(
                xaxis_title="Time (s)",
                yaxis_title="Decomposition Level",
                zaxis_title="Amplitude"
            )
        )
    else:
        for lvl in range(1, num_levels + 1):
            t_idx = time_indices[lvl]
            amplitude = np.abs(coeffs[lvl])
            fig.add_trace(go.Scatter(
                x=t_idx, y=amplitude,
                mode='lines', name=f'Level {lvl}'
            ))
        fig.update_layout(
            title=f'{title_text} - Time vs Amplitude',
            xaxis_title="Time (s)",
            yaxis_title="Amplitude"
        )

    st.plotly_chart(fig)


# Sidebar configuration
st.sidebar.title("Vibration Analysis Configuration")
ip_address_input = st.sidebar.text_input("NodeMCU IP Address", "192.168.1.1")
refresh_interval = st.sidebar.number_input("Data Refresh Interval (seconds)", min_value=1, value=10, step=1)
lowcut_input = st.sidebar.number_input("Low Cutoff Frequency (Hz)", min_value=0.1, value=0.5, step=0.1)
highcut_input = st.sidebar.number_input("High Cutoff Frequency (Hz)", min_value=0.1, value=20.0, step=0.1)
wavelet_type_input = st.sidebar.selectbox("Wavelet Type", pywt.wavelist(), index=0)
decomposition_level_input = st.sidebar.number_input("Decomposition Level", min_value=1, value=3, step=1)
graph_type_input = st.sidebar.radio("Graph Type", ("3D", "2D"))

# Real-time data fetching and processing
if st.sidebar.button("Fetch and Analyze Data"):
    st.info("Fetching and analyzing data. Please wait...")
    start_time = time.time()
    while True:
        node_data = fetch_data(ip_address_input)
        if node_data:
            results = process_vibration_data(
                node_data, lowcut_input, highcut_input, wavelet_type_input, decomposition_level_input
            )
            if results:
                time_series, x_filtered, y_filtered, z_filtered = results
                st.success("Data successfully fetched and processed.")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("### X-Axis Vibration Analysis")
                    plot_wavelet_analysis(time_series, x_filtered, "X-Axis Vibration", wavelet_type_input,
                                          decomposition_level_input, graph_type_input)
                with col2:
                    st.write("### Y-Axis Vibration Analysis")
                    plot_wavelet_analysis(time_series, y_filtered, "Y-Axis Vibration", wavelet_type_input,
                                          decomposition_level_input, graph_type_input)
                with col3:
                    st.write("### Z-Axis Vibration Analysis")
                    plot_wavelet_analysis(time_series, z_filtered, "Z-Axis Vibration", wavelet_type_input,
                                          decomposition_level_input, graph_type_input)

        time.sleep(refresh_interval)
        if time.time() - start_time > 60:
            break
else:
    st.info("Configure the settings in the sidebar and click 'Fetch and Analyze Data'.")

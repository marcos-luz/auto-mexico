#!/usr/bin/env python
# coding: utf-8

# # Script 01:

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import segyio
import time
import sys
import os
import re
import math
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from pygam import LinearGAM, s
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Input


# # Script 02:


def generate_converted_geometry_txt(
    station_spacing_ft=87.5,
    receiver_spacing_ft=87.5,
    n_shots=1001,
    n_receivers=180,
    output_path="converted_geometry.txt"
):

    ft_to_m = 0.3048
    station_spacing_m = station_spacing_ft * ft_to_m
    receiver_spacing_m = receiver_spacing_ft * ft_to_m

    shot_positions = np.arange(n_shots) * station_spacing_m
    receiver_positions = np.arange(n_receivers) * receiver_spacing_m

    geometry = []
    trace_id = 1  

    for shot_id, shot_pos in enumerate(shot_positions):
        for rec_id, rec_pos in enumerate(receiver_positions):
            offset = rec_pos - shot_pos
            cmp = (rec_pos + shot_pos) / 2
            geometry.append({
                "Trace": trace_id,
                "Shot_ID": shot_id,
                "Receiver_ID": rec_id,
                "Source_Position_m": round(shot_pos, 3),
                "Receiver_Position_m": round(rec_pos, 3),
                "Offset_m": round(offset, 3),
                "CMP_m": round(cmp, 3)
            })
            trace_id += 1

    df_geometry = pd.DataFrame(geometry)
    df_geometry.to_csv(output_path, sep="\t", index=False)
    return output_path

output_txt_path = "geometry.txt"
generate_converted_geometry_txt(output_path=output_txt_path)


# # Script 03:

def load_sgy_data_as_array(file_path):
    with segyio.open(file_path, "r", ignore_geometry=True) as sgy:
        data = segyio.tools.collect(sgy.trace[:])  
        twt = np.array(sgy.samples, dtype=np.float64)  
    return data.T, twt  

def save_as_npy(data, twt, output_base_name, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(os.path.join(output_dir, f"{output_base_name}_data.npy"), data)
    np.save(os.path.join(output_dir, f"{output_base_name}_twt.npy"), twt)

if __name__ == "__main__":
    input_directory = "mexico_files_sgy" 
    output_directory = os.path.join(os.getcwd(), "cdps-npy")  

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    input_files = [f for f in os.listdir(input_directory) if f.endswith(".sgy")]

    if not input_files:
        print("No .sgy file found in the input directory.")
    else:
        for filename in input_files:
            full_path = os.path.join(input_directory, filename)
            base_name = os.path.splitext(filename)[0]
            
            try:
                cdp_number = base_name.split('_')[1]  
                output_base_name = f"cdp_{cdp_number}"
            except IndexError:
                output_base_name = base_name  

            data, twt = load_sgy_data_as_array(full_path)
            save_as_npy(data, twt, output_base_name, output_directory)

            print(f"File {filename} successfully converted to .npy (base: {output_base_name})")


# Script 4: 

def process_column(column):
    n_samples = len(column)
    K = int(1 + 3.32 * np.log10(n_samples))
    total_amplitude = column.max() - column.min()
    h = total_amplitude / K
    lower_limit = np.linspace(column.min(), column.max() - h, K)
    upper_limit = lower_limit + h
    frequencies, _ = np.histogram(column, bins=np.append(lower_limit, upper_limit[-1]))
    df_classes = pd.DataFrame({
        'Class': [f"[{low:.2f}, {high:.2f})" for low, high in zip(lower_limit, upper_limit)],
        'Absolute Frequency': frequencies
    })
    return df_classes, K

def calculate_k2(df_classes, K):
    freq_count = df_classes['Absolute Frequency'].value_counts()
    zeros = freq_count.get(0, 0) + freq_count.get(1, 0)
    uniques = len({freq for freq, count in freq_count.items() if count > 1 and freq > 1})
    k1 = zeros + uniques
    k2 = K - k1
    return k2

def calculate_mean_k2(data_array):
    k2_values = []
    for i in range(data_array.shape[1]):
        column = data_array[:, i]
        df_result, K = process_column(pd.Series(column))
        k2 = calculate_k2(df_result, K)
        k2_values.append(k2)
    mean_k2 = round(np.mean(k2_values))
    return k2_values, mean_k2

if __name__ == "__main__":
    input_dir = "cdps-npy"
    output_dir = "k2-npy-output"
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith("_data.npy"):
            cdp_name = file.replace("_data.npy", "")
            input_path = os.path.join(input_dir, file)
            print(f"Processing: {cdp_name}")

            data_array = np.load(input_path)
            k2_values, mean_k2 = calculate_mean_k2(data_array)
            print(f"Number of clusters with mean k2: {mean_k2}")
            
            np.save(os.path.join(output_dir, f"{cdp_name}_k2_values.npy"), k2_values)
            np.save(os.path.join(output_dir, f"{cdp_name}_mean_k2.npy"), np.array([mean_k2]))

    print("All files have been processed and successfully saved as .npy.")
    
    
# Script 5: 


def k_means_clustering_by_trace(data_array, num_clusters):
    data_by_trace = data_array  
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=num_clusters, max_iter=9000)
    kmeans.fit(data_by_trace)
    return kmeans.cluster_centers_, kmeans.labels_

if __name__ == "__main__":
    input_data_dir = "cdps-npy"
    input_k2_dir = "k2-npy-output"
    output_dir = "kmeans-output"
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_data_dir):
        if file.endswith("_data.npy"):
            base_name = file.replace("_data.npy", "")
            data_path = os.path.join(input_data_dir, f"{base_name}_data.npy")
            k2_path = os.path.join(input_k2_dir, f"{base_name}_mean_k2.npy")

            if not os.path.exists(k2_path):
                print(f"[WARNING] Mean K2 not found for {base_name}, skipping...")
                continue

            print(f"[RUNNING K-means++] CDP: {base_name}")

            data_array = np.load(data_path)  
            num_clusters = int(np.load(k2_path)[0])  

            centers_array, labels = k_means_clustering_by_trace(data_array, num_clusters)

            np.save(os.path.join(output_dir, f"{base_name}_kmeans_centers.npy"), centers_array)
            np.save(os.path.join(output_dir, f"{base_name}_kmeans_labels.npy"), labels)

    print("K-means successfully applied to all CDPs.")


# Script 6: 

def identify_closest_traces(data_array, centers_array, twt_array):

    results = []
    n_samples, n_traces = data_array.shape

    for center in centers_array:
        center_replicated = np.tile(center, (n_samples, 1))  
        distances = np.linalg.norm(data_array - center_replicated, axis=1)
        min_index = np.argmin(distances)

        trace_index = min_index % n_traces
        sample_index = min_index // n_traces
        time_s = twt_array[sample_index] / 1000.0

        results.append({
            'Centroid_Mean': float(center.mean()),
            'Trace': trace_index + 1,
            'Time (s)': round(time_s, 6),
            'Sample': sample_index
        })

    return pd.DataFrame(results)

def build_trace_database(data_array, twt_array, centroid_df):
    trace_bank = []

    for _, row in centroid_df.iterrows():
        trace_idx = int(row['Trace']) - 1
        sample_idx = int(row['Sample'])

        amplitudes = data_array[:, trace_idx]
        nearest_time = twt_array[sample_idx]
        nearest_amplitude = data_array[sample_idx, trace_idx]

        trace_bank.append({
            'Trace': trace_idx + 1,
            'Time (ms)': twt_array.copy(),        
            'Amplitude': amplitudes.copy(),      
            'Nearest Time (ms)': nearest_time,
            'Nearest Amplitude': nearest_amplitude
        })

    return trace_bank


if __name__ == "__main__":
    input_dir = "cdps-npy"
    centers_dir = "kmeans-output"
    output_dir = "identificados-output"
    os.makedirs(output_dir, exist_ok=True)

    data_files = [f for f in os.listdir(input_dir) if f.endswith("_data.npy")]

    for file in data_files:
        base = file.replace("_data.npy", "")
        data_path = os.path.join(input_dir, f"{base}_data.npy")
        twt_path = os.path.join(input_dir, f"{base}_twt.npy")
        centers_path = os.path.join(centers_dir, f"{base}_kmeans_centers.npy")

        if not all(map(os.path.exists, [data_path, twt_path, centers_path])):
            print(f"[CDP {base}] Missing files. Skipping.")
            continue

        data_array = np.load(data_path)
        twt_array = np.load(twt_path)
        centers_array = np.load(centers_path)

        centroid_df = identify_closest_traces(data_array, centers_array, twt_array)
        np.save(os.path.join(output_dir, f"{base}_centroid_info.npy"), centroid_df.to_records(index=False))

        trace_bank = build_trace_database(data_array, twt_array, centroid_df)
        np.save(os.path.join(output_dir, f"{base}_banco_tracos.npy"), trace_bank)

        print(f"[CDP {base}] Identification and saving completed.")


# Script 7: 

input_dir = "identified-output"
output_dir = "trace_figures"
os.makedirs(output_dir, exist_ok=True)

def extract_cdp(filename):
    match = re.search(r"cdp_(\d+)_trace_bank\.npy", filename)
    return int(match.group(1)) if match else None

files = [f for f in os.listdir(input_dir) if f.endswith("_trace_bank.npy")]
cdp_numbers = sorted([extract_cdp(f) for f in files if extract_cdp(f) is not None])

if len(cdp_numbers) < 3:
    raise ValueError("At least 3 files are required to identify minimum, median, and maximum CDPs.")

cdp_min = cdp_numbers[0]
cdp_max = cdp_numbers[-1]
cdp_med = cdp_numbers[len(cdp_numbers) // 2]
selected_cdps = [cdp_min, cdp_med, cdp_max]


for cdp in selected_cdps:
    file_path = os.path.join(input_dir, f"cdp_{cdp}_trace_bank.npy")
    trace_bank = np.load(file_path, allow_pickle=True)

    cdp_output_dir = os.path.join(output_dir, f"cdp_{cdp}")
    os.makedirs(cdp_output_dir, exist_ok=True)

    for item in trace_bank:
        trace = item['Trace']
        times = item['Time (ms)']
        amplitudes = item['Amplitude']

        idx_max = np.argmax(np.abs(amplitudes))
        marked_time = times[idx_max]

        marked_amp = item['Nearest Amplitude']

        plt.figure(figsize=(8, 10))
        plt.plot(amplitudes, times, color='blue', label=f"Trace {trace}")
        plt.axhline(y=marked_time, color='red', linestyle='--', label=f"Centroid time: {marked_time:.2f} ms")
        plt.scatter([marked_amp], [marked_time], color='red', s=50, marker='o', label="Centroid amplitude")
        plt.title(f"CDP {cdp} - Trace {trace}")
        plt.xlabel("Amplitude")
        plt.ylabel("Time (ms)")
        plt.gca().invert_yaxis()
        plt.legend()
        plt.minorticks_on()
        plt.grid(which='major', linestyle='-', linewidth=0.5)
        plt.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
        plt.tight_layout()

        fig_name = os.path.join(cdp_output_dir, f"cdp_{cdp}_trace_{trace}.png")
        plt.savefig(fig_name, dpi=300)
        plt.close()

    print(f"[CDP {cdp}] Figures saved in: {cdp_output_dir}")

print("Final processing completed successfully.")



# Script 8:

def apply_pca_on_bank(trace_bank):
    matrix = []
    for trace in trace_bank:
        matrix.append(trace['Amplitude']) 

    seismic_matrix = np.array(matrix).T 
    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(seismic_matrix)
    pca_trace = pca_result.flatten()
    time_ms = trace_bank[0]['Time (ms)']
    return time_ms, pca_trace

def plot_pca_trace(pca_trace, time_ms, cdp_id, output_dir):
    plt.figure(figsize=(8, 10))
    plt.plot(pca_trace, time_ms, color='blue', label=f"PCA Trace - CDP {cdp_id}")
    plt.title(f"Principal Trace (PCA) - CDP {cdp_id}")
    plt.xlabel("Amplitude")
    plt.ylabel("Time (ms)")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth=0.5)
    plt.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"pca_trace_cdp_{cdp_id}.png"), dpi=300)
    plt.close()

input_dir = "identified-output"
output_dir = "pca-output"

plot_dir = "pca_figures"  

os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)


bank_files = sorted([f for f in os.listdir(input_dir) if f.endswith("_trace_bank.npy")])
cdp_ids = sorted([int(f.split("_")[1]) for f in bank_files])
selected = [cdp_ids[0], cdp_ids[len(cdp_ids)//2], cdp_ids[-1]]  

for f in bank_files:
    cdp_id = int(f.split("_")[1])
    bank_path = os.path.join(input_dir, f)
    trace_bank = np.load(bank_path, allow_pickle=True)

    time_ms, pca_trace = apply_pca_on_bank(trace_bank)

    np.save(os.path.join(output_dir, f"pca_trace_{cdp_id}.npy"), pca_trace)
    print(f"PCA trace saved as: pca_trace_{cdp_id}.npy.")
    np.save(os.path.join(output_dir, f"pca_time_{cdp_id}.npy"), time_ms)
    print(f"Times saved as: pca_time_{cdp_id}.npy.")


    if cdp_id in selected:
        plot_pca_trace(pca_trace, time_ms, cdp_id, plot_dir)


# Script 9:

def process_column(column):
    n_samples = len(column)
    K = int(1 + 3.32 * np.log10(n_samples))  
    total_amplitude = column.max() - column.min()
    h = total_amplitude / K
    lower_limit = np.linspace(column.min(), column.max() - h, K)
    upper_limit = lower_limit + h
    frequencies, bins = np.histogram(column, bins=np.append(lower_limit, upper_limit[-1]))
    df_classes = pd.DataFrame({
        'Class': [f"[{low:.2f}, {high:.2f})" for low, high in zip(lower_limit, upper_limit)],
        'Absolute Frequency': frequencies
    })
    return df_classes, K

def calculate_k2_absolute_values(df_classes, K):
    freq_count = df_classes['Absolute Frequency'].value_counts()
    zeros = freq_count.get(0, 0) + freq_count.get(1, 0)
    uniques = len({freq for freq, count in freq_count.items() if count > 1 and freq > 1})
    k1 = zeros + uniques
    k2 = K - k1
    return k2

def calculate_k2_pca_trace(trace_array):
    df_result, K = process_column(pd.Series(trace_array))
    k2 = calculate_k2_absolute_values(df_result, K)
    return k2, K, df_result

if __name__ == "__main__":
    input_dir = "pca-output"
    output_dir = "k2-pca-output"
    os.makedirs(output_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(input_dir) if f.startswith("pca_trace_") and f.endswith(".npy")])

    for fname in files:
        cdp_id = fname.replace("pca_trace_", "").replace(".npy", "")
        trace_path = os.path.join(input_dir, fname)

        pca_trace = np.load(trace_path)
        k2, K, df_freq = calculate_k2_pca_trace(pca_trace)
        print(f"CDP {cdp_id}")
        print(f"Total number of classes (K): {K}")
        print(f"Estimated k2 (clusters): {k2}")
        
        np.save(os.path.join(output_dir, f"{cdp_id}_k2_pca.npy"), np.array([k2]))
        np.save(os.path.join(output_dir, f"{cdp_id}_classes_freq_pca.npy"), df_freq.to_records(index=False))

print("k2 results and frequency classes saved to '.npy'.")


# Script 10:

def k_means_clustering_by_trace(data_array, num_clusters):
    data_by_trace = data_array 
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=num_clusters, max_iter=9000)
    kmeans.fit(data_by_trace)
    return kmeans.cluster_centers_, kmeans.labels_

input_data_dir = "pca-output"
input_k2_dir = "k2-pca-output"
output_dir = "kmeans-pca-output"
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_data_dir):
    if file.startswith("pca_trace_") and file.endswith(".npy"):
        cdp_id = file.replace("pca_trace_", "").replace(".npy", "")
        data_path = os.path.join(input_data_dir, file)
        k2_path = os.path.join(input_k2_dir, f"{cdp_id}_k2_pca.npy")

        if not os.path.exists(k2_path):
            print(f"[WARNING] PCA k2 not found for CDP {cdp_id}, skipping...")
            continue

        print(f"[RUNNING K-means++ PCA] CDP: {cdp_id}")

        data_array = np.load(data_path).reshape(-1, 1)  
        num_clusters = int(np.load(k2_path)[0])

        centers_array, labels = k_means_clustering_by_trace(data_array, num_clusters)

        np.save(os.path.join(output_dir, f"{cdp_id}_kmeans_centers_pca.npy"), centers_array)
        np.save(os.path.join(output_dir, f"{cdp_id}_kmeans_labels_pca.npy"), labels)

print("K-means successfully applied to PCA traces.")


# Script 11: 

def plot_pca_with_markings(cdp_id, trace_path, time_path, centroid_path, output_dir):
    pca_trace = np.load(trace_path)
    twt_array = np.load(time_path)  
    centers_array = np.load(centroid_path)  

    plt.figure(figsize=(8, 10))
    plt.plot(pca_trace, twt_array, color='blue', label=f'PCA Trace - CDP {cdp_id}')

    for center in centers_array:
        centroid_val = float(center[0])
        nearest_idx = np.argmin(np.abs(pca_trace - centroid_val))
        marked_time = twt_array[nearest_idx]
        marked_amp = pca_trace[nearest_idx]

        plt.axhline(y=marked_time, color='red', linestyle='--', linewidth=1, label=f"Time: {marked_time:.2f} ms")
        plt.scatter([marked_amp], [marked_time], color='red', s=50, marker='o', label=f"Amplitude: {marked_amp:.2f}")

    plt.title(f"CDP {cdp_id} - Trace with centroid markings")
    plt.xlabel("Amplitude")
    plt.ylabel("Time (ms)")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth=0.5)
    plt.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig_name = os.path.join(output_dir, f"cdp_{cdp_id}_pca_with_markings.png")
    plt.savefig(fig_name, dpi=300)
    plt.close()
    print(f"[CDP {cdp_id}] Figure saved to {fig_name}")


if __name__ == "__main__":
    pca_dir = "pca-output"
    centroid_dir = "kmeans-pca-output"
    output_dir = "pca_figures_marked"
    os.makedirs(output_dir, exist_ok=True)

    valid_cdp_ids = []
    for file in os.listdir(pca_dir):
        if file.startswith("pca_trace_") and file.endswith(".npy"):
            cdp_id = file.replace("pca_trace_", "").replace(".npy", "")
            trace_path = os.path.join(pca_dir, f"pca_trace_{cdp_id}.npy")
            time_path = os.path.join(pca_dir, f"pca_time_{cdp_id}.npy")
            centroid_path = os.path.join(centroid_dir, f"{cdp_id}_kmeans_centers_pca.npy")

            if all(map(os.path.exists, [trace_path, time_path, centroid_path])):
                valid_cdp_ids.append(int(cdp_id))

    valid_cdp_ids = sorted(valid_cdp_ids)

    if len(valid_cdp_ids) < 3:
        raise ValueError("At least 3 CDPs with complete data (trace, time, centroid) are required.")

    selected = [valid_cdp_ids[0], valid_cdp_ids[len(valid_cdp_ids) // 2], valid_cdp_ids[-1]]

    for cdp_id in selected:
        trace_path = os.path.join(pca_dir, f"pca_trace_{cdp_id}.npy")
        time_path = os.path.join(pca_dir, f"pca_time_{cdp_id}.npy")
        centroid_path = os.path.join(centroid_dir, f"{cdp_id}_kmeans_centers_pca.npy")

        plot_pca_with_markings(cdp_id, trace_path, time_path, centroid_path, output_dir)

    print("Plotting completed for minimum, median, and maximum CDPs.")


# Script 12: 

centers_dir = "kmeans-pca-output"
traces_dir = "identified-output"
geometry_path = "converted_geometry.txt"
output_dir = "dix_centroids_input_final"
os.makedirs(output_dir, exist_ok=True)

df_geometry = pd.read_csv(geometry_path, sep="\t")
offset_dict = dict(zip(df_geometry["Trace"], df_geometry["Offset_m"]))

center_files = sorted([f for f in os.listdir(centers_dir) if f.endswith("_kmeans_centers_pca.npy")])

processing_log = []

for fname in center_files:
    match = re.match(r"(\d+)_kmeans_centers_pca\.npy", fname)
    if not match:
        continue
    cdp_id = match.group(1)
    centers_path = os.path.join(centers_dir, fname)
    traces_path = os.path.join(traces_dir, f"cdp_{cdp_id}_trace_bank.npy")

    if not os.path.exists(traces_path):
        processing_log.append(f"[CDP {cdp_id}] Trace file missing. Skipping.")
        continue

    centers_array = np.load(centers_path)
    if centers_array.ndim == 2 and centers_array.shape[1] == 1:
        centers_array = centers_array.flatten()

    trace_bank = np.load(traces_path, allow_pickle=True)

    records = []
    for centroid in centers_array:
        best_diff = np.inf
        best_sample = None

        for trace in trace_bank:
            amplitudes = trace["Amplitude"]
            times = trace["Time (ms)"] / 1000.0  
            trace_id = trace["Trace"]

            idx = np.argmin(np.abs(amplitudes - centroid))
            diff = abs(amplitudes[idx] - centroid)

            if diff < best_diff:
                best_diff = diff
                offset = offset_dict.get(trace_id, None)
                if offset is not None:
                    best_sample = {
                        "Time (s)": round(times[idx], 6),
                        "Amplitude": round(float(amplitudes[idx]), 6),
                        "Offset_m": round(float(offset), 3)
                    }

        if best_sample:
            records.append(best_sample)

    if records:
        df_out = pd.DataFrame(records).sort_values(by="Time (s)").reset_index(drop=True)
        output_path = os.path.join(output_dir, f"centroid_dix_input_cdp_{cdp_id}.csv")
        df_out.to_csv(output_path, index=False)
        processing_log.append(f"[CDP {cdp_id}] File exported to: {output_path}")
    else:
        processing_log.append(f"[CDP {cdp_id}] No valid matches found.")

print("\n".join(processing_log))


def compute_vrms(offset, time):
    """
    Compute RMS velocity (VRMS) from offset and time:
    vrms = offset / sqrt(2 * time)
    """
    try:
        return offset / np.sqrt(2 * time) if time > 0 else np.nan
    except Exception:
        return np.nan

def compute_dix_velocities(df):
    """
    Apply the DIX formula to obtain interval velocities (Vint) and RMS (Vrms).
    """
    times = df['Time (s)'].values
    offsets = df['Offset_m'].values

    vrms_list = []
    vint_list = []

    for t, x in zip(times, offsets):
        vrms = compute_vrms(abs(x), t)
        vrms_list.append(vrms)

    for i in range(len(vrms_list)):
        if i == 0:
            vint_list.append(vrms_list[i])
        else:
            t1, t2 = times[i-1], times[i]
            v1, v2 = vrms_list[i-1], vrms_list[i]

            num = (v2**2) * t2 - (v1**2) * t1
            den = t2 - t1

            vint = np.sqrt(num / den) if den > 0 and num > 0 else np.nan
            vint_list.append(vint)

    df_out = pd.DataFrame({
        "Time (s)": times,
        "Offset_m": offsets,
        "VRMS (m/s)": vrms_list,
        "Interval Velocity (m/s)": vint_list,
        "VNMO/VRMS": vrms_list
    })

    return df_out

def export_nmo(df_result, cdp_id, output_dir, method="dix"):
    """
    Export computed data to .nmo file in the required format.
    """
    tnmo = ",".join(f"{row['Time (s)']:.3f}" for _, row in df_result.iterrows())

    vnmo = ",".join(f"{row['VNMO/VRMS']+4000:.0f}" for _, row in df_result.iterrows())

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"vnmo_{method}_{cdp_id}.nmo"), "w") as f:
        f.write(f"cdp={cdp_id}\n")
        f.write(f"vnmo={vnmo}\n")
        f.write(f"tnmo={tnmo}\n")

def run_dix_batch(input_dir, output_dir, method="dix"):
    """
    Run velocity calculation for all .csv files in the input directory.
    Only saves files with valid data: no NaN and strictly increasing tnmo.
    """
    files = sorted([
        f for f in os.listdir(input_dir)
        if f.startswith("centroid_dix_input_cdp_") and f.endswith(".csv")
    ])

    if not files:
        print("No input files found.")
        return

    for fname in files:
        match = re.search(r"cdp_(\d+)\.csv", fname)
        if not match:
            continue

        cdp_id = match.group(1)
        csv_path = os.path.join(input_dir, fname)

        try:
            df = pd.read_csv(csv_path)
            df = df.sort_values(by="Time (s)").reset_index(drop=True)

            df_result = compute_dix_velocities(df)

            if df_result[["Time (s)", "Offset_m", "VNMO/VRMS"]].isnull().any().any():
                print(f"[CDP {cdp_id}] Contains NaN in Time, Offset, or VNMO/VRMS. File skipped.")
                continue

            times = df_result["Time (s)"].values
            if not np.all(np.diff(times) > 0):
                print(f"[CDP {cdp_id}] tnmo is not strictly increasing. File skipped.")
                continue

            export_nmo(df_result, cdp_id, output_dir, method)
            print(f"[CDP {cdp_id}] .nmo file successfully generated.")
        except Exception as e:
            print(f"[ERROR - CDP {cdp_id}] {str(e)}")

if __name__ == "__main__":
    input_dir = "dix_centroids_input_final"
    output_dir = "vnmo_dix_output"
    run_dix_batch(input_dir, output_dir)


# Script 14: 

input_dir = "vnmo_dix_output"  
output_dir = "vnmo_mlp_output"
os.makedirs(output_dir, exist_ok=True)

def load_nmo(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    cdp_id = re.search(r"cdp=(\d+)", lines[0]).group(1)
    vnmo = np.array([float(x) for x in lines[1].split("=")[1].split(",")]) / 1000.0  
    tnmo = np.array([float(x) for x in lines[2].split("=")[1].split(",")])
    return cdp_id, tnmo.reshape(-1, 1), vnmo

def fit_velocity_mlp(tnmo, vnmo):
    mlp = MLPRegressor(
        hidden_layer_sizes=(10, 10),
        activation='logistic',
        solver='adam',
        max_iter=500000,
        random_state=42
    )
    mlp.fit(tnmo, vnmo)
    vnmo_fitted = mlp.predict(tnmo)
    vnmo_monotonic = np.maximum.accumulate(vnmo_fitted)
    return vnmo_monotonic

def export_nmo(cdp_id, tnmo, vnmo_adjusted):
    vnmo_str = ",".join(f"{int(v * 1000)}" for v in vnmo_adjusted)  
    tnmo_str = ",".join(f"{t[0]:.3f}" for t in tnmo)
    output_path = os.path.join(output_dir, f"vnmo_mlp_{cdp_id}.nmo")
    with open(output_path, "w") as f:
        f.write(f"cdp={cdp_id}\n")
        f.write(f"vnmo={vnmo_str}\n")
        f.write(f"tnmo={tnmo_str}\n")
    return output_path

nmo_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".nmo") and f.startswith("vnmo_dix_")])
adjusted_files = []

for filename in nmo_files:
    path = os.path.join(input_dir, filename)
    cdp_id, tnmo, vnmo = load_nmo(path)
    vnmo_adjusted = fit_velocity_mlp(tnmo, vnmo)
    output_path = export_nmo(cdp_id, tnmo, vnmo_adjusted)
    adjusted_files.append(output_path)

adjusted_files[:5]


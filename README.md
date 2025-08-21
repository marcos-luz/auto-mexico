# auto-mexico
Auto Mexico – Automated Seismic Velocity Pipeline (PCA + K-means++ + DIX + MLP)

A complete and automated pipeline to:

1. convert CDPs in SEG-Y format into `.npy`,
2. estimate **k2** (number of clusters) via histogram,
3. apply **K-means++**,
4. identify representative traces/times,
5. extract the **main trace via PCA**,
6. regroup and mark centroids in the PCA,
7. associate centroids with real **time–offset** pairs,
8. calculate **VRMS/Vint** using **Dix**,
9. **smooth and enforce monotonicity** with **MLP**,
10. consolidate **`.nmo` files** for migration/stacking.

> **Scope**: designed for Gulf of Mexico data (example), but generalizable.
> **Primary input**: `.sgy` files organized by CDP.

---

## Table of Contents

* [Pipeline Architecture](#pipeline-architecture)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Directory Structure](#directory-structure)
* [How to Run (step by step)](#how-to-run-step-by-step)
* [Key Parameters](#key-parameters)
* [Generated Outputs](#generated-outputs)
* [Best Practices and Performance](#best-practices-and-performance)
* [Troubleshooting](#troubleshooting)
* [Limitations and Next Steps](#limitations-and-next-steps)
* [Citation](#citation)
* [License](#license)

---

## Pipeline Architecture

1. **Geometry Generation**
   Creates `geometria_convertida.txt` with **Trace**, **Shot\_ID**, **Receiver\_ID**, positions, **Offset\_m** (signed), and **CMP\_m**.

2. **SEG-Y → NPY Conversion**
   Reads `.sgy`, extracts traces and samples (TWT), saves `*_data.npy` (matrix `[n_samples, n_traces]`) and `*_twt.npy`.

3. **k2 Estimation via Histogram (Sturges)**
   For each column (trace), calculates classes, frequencies, and estimates average **k2** per CDP.

4. **K-means++ (original data)**
   Applies K-means++ to traces to obtain **centroids** and **labels** per CDP.

5. **Identification of traces/samples near centroids**
   Returns the **trace** and **sample** (time/amplitude) closest to each centroid and builds a **trace bank**.

6. **Trace plotting (min/med/max CDPs)**
   Generates figures per trace, marking the time of maximum absolute amplitude and centroid point.

7. **PCA (main trace)**
   Applies **PCA (1 component)** to the trace bank → saves the **PCA trace** and **times**; plots for min/med/max CDPs.

8. **k2 Estimation on PCA**
   Repeats histogram process on the PCA 1D series to obtain **k2\_PCA**.

9. **K-means++ on PCA**
   Clusters the PCA trace (1D) with K-means++ using **k2\_PCA**.

10. **PCA plot with centroid markings**
    Plots the PCA trace and marks each centroid with a **time line** and **amplitude**.

11. **Linking centroids (PCA) to (time, offset)**
    Matches centroid amplitudes with **real trace amplitudes** and, using geometry, obtains the corresponding **offset**.
    Exports a **CSV sorted by time** for Dix input.

12. **VRMS/Vint Calculation (Dix)**
    Calculates **VRMS** for each (t, x) pair and **Vint** via **Dix** (quadratic differences). Generates `.nmo` (vnmo/tnmo).

13. **Adjustment with MLP + monotonicity**
    Fits **VNMO (m/s)** with **MLPRegressor** and enforces a **non-decreasing constraint**. Exports adjusted `.nmo`.

14. **Consolidation**
    Merges multiple `.nmo` files into a **single file** containing the CDP sequence and its `tnmo/vnmo` pairs.

---

## Prerequisites

* Python 3.10+ (recommended)
* Libraries:

  * NumPy, Pandas, Matplotlib, Seaborn
  * **segyio**
  * **scikit-learn** (KMeans, PCA, MLPRegressor)
  * SciPy
  * **pygam** (LinearGAM) *(present in script; optional in base flow)*
  * TensorFlow / Keras *(used in imports; not mandatory in standard flow)*
* System:

  * Sufficient RAM to load CDPs into a `[n_samples, n_traces]` matrix
  * Drivers/compilers compatible with `segyio`

> Tip: use a **virtual environment** to isolate dependencies.

---

## Installation

```bash
# 1) clone your repository
git clone https://github.com/<USER>/<REPO>.git
cd <REPO>

# 2) create and activate a venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) install dependencies
pip install numpy pandas matplotlib seaborn segyio scikit-learn scipy pygam tensorflow
```

> If TensorFlow is not needed, you can omit it to simplify setup:
> `pip install numpy pandas matplotlib seaborn segyio scikit-learn scipy pygam`

---

## Directory Structure

```text
<REPO>/
├─ auto_mexico.py                # main pipeline (scripts 1–15)
├─ arquivos_mexico_sgy/          # <place input .sgy files here>
├─ cdps-npy/                     # .npy outputs (data/twt)
├─ k2-npy-output/                # average k2 per CDP (original data)
├─ kmeans-output/                # centroids/labels (original data)
├─ identificados-output/         # centroid_info + trace bank per CDP
├─ figuras_tracos/               # trace figures per CDP
│  └─ cdp_<id>/
├─ pca-output/                   # pca_trace_<id>.npy, pca_tempo_<id>.npy
├─ figuras_pca/                  # PCA trace plots (min/med/max)
├─ k2-pca-output/                # k2 on PCA
├─ kmeans-pca-output/            # centroids on PCA
├─ figuras_pca_marcadas/         # PCA with centroid markings
├─ entrada_dix_centroides_final/ # CSVs time-amp-offset for Dix
├─ vnmo_dix_output/              # .nmo (Dix)
├─ vnmo_mlp_output/              # adjusted .nmo (MLP + monotonicity)
└─ vnmo-arquivo-unico/           # final consolidated .nmo file
```

---

## How to Run (step by step)

> All steps are orchestrated within the **same file** `auto_mexico.py`, in “Script N” blocks.
> Run the file according to the **main** sections (several blocks already have `if __name__ == "__main__":`).

1. **Prepare inputs**

```bash
mkdir -p arquivos_mexico_sgy
# copy your .sgy files into ./arquivos_mexico_sgy
```

2. **Generate geometry (Script 2)**

* Defaults: 1001 shots, 180 receivers, 87.5 ft spacing.
* Edit parameters in `gerar_geometria_convertida_txt(...)` if needed.
* Output: `geometria_convertida.txt`.

3. **Convert SEG-Y → NPY (Script 3)**

```bash
python auto_mexico.py
# logs will show conversion into cdps-npy/
```

4. **Estimate k2 and run K-means++ (Scripts 4–5)**

* Running the file processes everything in main blocks:

  * `k2-npy-output/` gets `*_media_k2.npy`
  * `kmeans-output/` gets centroids and labels

5. **Identify traces and generate figures (Scripts 6–7)**

* Outputs in `identificados-output/` and figures in `figuras_tracos/`.

6. **Compute PCA and plot (Script 8)**

* `pca-output/` and `figuras_pca/`.

7. **k2 on PCA + K-means++ on PCA + plots (Scripts 9–11)**

* `k2-pca-output/`, `kmeans-pca-output/`, `figuras_pca_marcadas/`.

8. **Relate PCA centroids to offsets and generate CSVs (Script 12)**

* Outputs in `entrada_dix_centroides_final/centroide_dix_input_cdp_<id>.csv`.

9. **DIX: VRMS/Vint + .nmo (Script 13)**

* Outputs in `vnmo_dix_output/`.

10. **Adjust with MLP + monotonicity (Script 14)**

* Outputs in `vnmo_mlp_output/`.

11. **Consolidate .nmo (Script 15)**

* Final output: `vnmo-arquivo-unico/vnmo_pca_consolidado.nmo`.

> **Note:** some blocks assume there are at least **3 valid CDPs** to select **minimum, median, maximum** for plotting.

---

## Key Parameters

* **Geometry (Script 2)**
  `intervalo_estacoes_ft`, `intervalo_receptores_ft`, `n_tiros`, `n_receptores`

* **Histogram (Scripts 4 and 9)**
  **Sturges**: `K = 1 + 3.32 log10(n)`

* **K-means++ (Scripts 5 and 10)**
  `n_clusters = media_k2` (original data) or `k2_PCA` (PCA)
  `random_state = num_clusters`, `max_iter = 9000`

* **PCA (Script 8)**
  `n_components = 1` (main trace)

* **DIX (Script 13)**
  `VRMS = offset / sqrt(2 * time)`,
  `Vint_n = sqrt( (V2^2 * t2 − V1^2 * t1) / (t2 − t1) )`
  (incremental calculation ordered by increasing time)

* **MLP (Script 14)**
  `hidden_layer_sizes=(10,10)`, `activation='logistic'`, `max_iter=500000`
  Post-processing: `np.maximum.accumulate(...)` for monotonicity.

---

## Generated Outputs

* **Geometry**: `geometria_convertida.txt` (TSV)
* **NPY**: `cdp_<id>_data.npy`, `cdp_<id>_twt.npy`
* **k2** (original and PCA): `*_media_k2.npy`, `*_k2_pca.npy`
* **Centroids**: `*_kmeans_centers.npy`, `*_kmeans_centers_pca.npy`
* **Identification / Trace Bank**: `*_centroid_info.npy`, `*_banco_tracos.npy`
* **Figures**: traces per CDP; simple PCA and PCA with markings
* **CSV for Dix**: `centroide_dix_input_cdp_<id>.csv`
* **`.nmo` files**:

  * Dix: `vnmo_dix_<id>.nmo`
  * Adjusted: `vnmo_mlp_<id>.nmo`
  * Consolidated: `vnmo_pca_consolidado.nmo`

---

## Best Practices and Performance

* **Memory**: be careful when loading too many CDPs; process in batches.
* **Parallelization**: current scripts are **sequential**; can be parallelized by CDP.
* **Seeds**: `random_state` set for reproducibility in K-means++.
* **Data quality**: ensure TWT is in **ms** (script uses ms in some places and converts to s when indicated).

---

## Troubleshooting

* **“No .sgy file found”**
  → Check if `.sgy` files are in `./arquivos_mexico_sgy`. They should be maximum coverage CMP panels (example rename: `cdp_17612_maxcov.sgy`).

* **“At least 3 files required …” (plots)**
  → Add more valid CDPs (minimum 3) for min/med/max plotting sections.

* **Missing files (warning messages)**
  → Ensure previous scripts were run in the correct order.

* **NaN in VNMO/VRMS or non-increasing times**
  → Script 13 **skips** CDPs with inconsistencies. Review input CSV (`entrada_dix_centroides_final/`).

* **Non-monotonic velocities after MLP**
  → Post-processing enforces monotonicity; if curves are “flat”, adjust MLP architecture, `max_iter`, or provide more samples.

* **`segyio` fails to install/open SEG-Y**
  → Check Python version, system dependencies, and read permissions.

---

## Limitations and Next Steps

* **k2 heuristic**: based on class frequencies (Sturges). Consider alternatives (e.g., **Silhouette**, **BIC/AIC**, **Gap Statistic**).
* **PCA 1D**: fast but may lose relevant variation; consider **n\_components > 1** and automatic selection.
* **Direct DIX from (t,x)**: simplified VRMS formula; could be refined with classical **NMO picking** and QC.
* **MLP**: fixed architecture; evaluate **GAM** (already imported), **monotonic spline**, or **isotonic regression**.
* **Logs**: centralize logs (e.g., `logging`) and create a **CLI** (`argparse`) for each step.

---

## Citation

If this pipeline is useful, please cite the repository/project. This README was prepared based on the script `auto_mexico.py`.



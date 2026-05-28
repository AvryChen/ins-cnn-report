# LNO327 Fig.2 Data Disk and Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all new Fig.2 datasets, chunks, and reusable preprocessed inputs live on `/data`, while making Fig.2 training avoid repeated preprocessing and avoid keeping unnecessary arrays on GPU.

**Architecture:** Keep code in `~/work/ins-cnn`, write large data to `/data/ins-cnn`. Generate raw Sunny HDF5 once, precompute the CNN-ready Fig.2 input once, then train from that cache. For large Fig.2 runs, keep the full input on CPU and move only the current batch to GPU.

**Tech Stack:** Julia/Sunny.jl data generation, HDF5, Python/h5py, NumPy, PyTorch AMP, bash queue scripts.

---

## File Structure

- Modify `scripts/lno327_generate_chunked_inverse_dataset.py`
  - Add `--chunk-dir`.
  - Add `--cleanup-chunks`.
  - Let large chunk files live under `/data/ins-cnn/chunks`.

- Create `scripts/lno327_precompute_fig2_inputs.py`
  - Read raw `figure2/maps`.
  - Apply the existing weak-peak transform once.
  - Save a compact float16 cache under `/data/ins-cnn/cache`.

- Modify `inverse_spectrum/lno327_baseline.py`
  - Add optional load flags so Fig.2-only training does not read Fig.3 arrays.

- Modify `inverse_spectrum/lno327_cnn.py`
  - Make weak-peak transform work in place.
  - Add CPU-resident input mode for large Fig.2 training.
  - Add preprocessed-cache input path.

- Modify `scripts/lno327_run_cnn_task.py`
  - Pass through `--input-cache`.
  - Pass through `--input-device cpu|cuda|auto`.

- Modify `scripts/lno327_run_fig2_queue.sh`
  - Write raw data to `/data/ins-cnn/data`.
  - Write chunks to `/data/ins-cnn/chunks`.
  - Write preprocessed inputs to `/data/ins-cnn/cache`.
  - Train n4096 first, then n32768.

- Test `tests/test_lno327_fig2_cache_and_memory.py`
  - Tiny HDF5 cache test.
  - Loader flag test.
  - In-place transform test.

---

## Task 1: Put New Data on `/data`

**Files:**
- Modify: `scripts/lno327_generate_chunked_inverse_dataset.py`
- Modify: `scripts/lno327_run_fig2_queue.sh`

- [ ] **Step 1: Add generator args**

Add these parser arguments to `scripts/lno327_generate_chunked_inverse_dataset.py`:

```python
parser.add_argument("--chunk-dir", type=Path, default=None)
parser.add_argument("--cleanup-chunks", action="store_true")
```

- [ ] **Step 2: Route chunk output**

Replace the current chunk-dir line:

```python
chunk_dir = root / "data" / f"{tag}_chunks"
```

with:

```python
chunk_dir = args.chunk_dir if args.chunk_dir is not None else root / "data" / f"{tag}_chunks"
chunk_dir = chunk_dir if chunk_dir.is_absolute() else root / chunk_dir
```

- [ ] **Step 3: Clean chunks only after successful merge**

After:

```python
tmp_out.replace(out)
```

add:

```python
if args.cleanup_chunks and chunk_dir.exists():
    shutil.rmtree(chunk_dir)
    print(f"[cleanup] removed chunks: {chunk_dir}", flush=True)
```

- [ ] **Step 4: Use `/data` in the Fig.2 queue**

At the top of `scripts/lno327_run_fig2_queue.sh`, define:

```bash
DATA_DISK_ROOT="${DATA_DISK_ROOT:-/data/ins-cnn}"
DATA_DIR="${DATA_DIR:-$DATA_DISK_ROOT/data}"
CHUNK_DIR="${CHUNK_DIR:-$DATA_DISK_ROOT/chunks}"
CACHE_DIR="${CACHE_DIR:-$DATA_DISK_ROOT/cache}"

mkdir -p "$DATA_DIR" "$CHUNK_DIR" "$CACHE_DIR"

DATA_N4096="${DATA_N4096:-$DATA_DIR/lno327_inverse_fig2fig3_fwhm10_n4096_q160_e180_h81_k121_range200.h5}"
DATA_N32768="${DATA_N32768:-$DATA_DIR/lno327_inverse_fig2fig3_fwhm10_n32768_q160_e180_h81_k121_range200.h5}"
```

- [ ] **Step 5: Pass chunk location during generation**

For n4096:

```bash
--chunk-dir "$CHUNK_DIR/lno327_inverse_fig2fig3_fwhm10_n4096_q160_e180_h81_k121_range200_chunks" \
--cleanup-chunks \
```

For n32768:

```bash
--chunk-dir "$CHUNK_DIR/lno327_inverse_fig2fig3_fwhm10_n32768_q160_e180_h81_k121_range200_chunks" \
--cleanup-chunks \
```

- [ ] **Step 6: Verify on remote**

Run:

```bash
ssh ins-cnn-cloud 'mkdir -p /data/ins-cnn/{data,chunks,cache}; df -h /data ~/work/ins-cnn'
```

Expected:

```text
/data has about 93G free
~/work/ins-cnn remains on system disk
```

---

## Task 2: Precompute Fig.2 CNN Inputs Once

**Files:**
- Create: `scripts/lno327_precompute_fig2_inputs.py`
- Modify: `scripts/lno327_run_fig2_queue.sh`

- [ ] **Step 1: Create the cache script**

Create `scripts/lno327_precompute_fig2_inputs.py` with this interface:

```python
parser.add_argument("--data", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--percentile", type=float, default=99.5)
parser.add_argument("--gamma", type=float, default=0.5)
parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
parser.add_argument("--force", action="store_true")
```

- [ ] **Step 2: Cache datasets**

The output HDF5 must contain:

```text
fig2/x                      float16 or float32, shape (N, 10, 81, 121)
paper_reference/fig2_x       float16 or float32, shape (1, 10, 81, 121)
params                       copied from source
split                        copied from source
param_names                  copied from source
figure2/energies             copied from source
figure2/h_values             copied from source
figure2/k_values             copied from source
figure2 attrs l_value        copied from source
attrs source_data            absolute source path
attrs percentile             99.5
attrs gamma                  0.5
attrs input_kind             fig2_preprocessed
```

- [ ] **Step 3: Use chunked preprocessing**

Process `figure2/maps` in blocks so Python never creates another full 13GB copy:

```python
for start in range(0, n, args.batch):
    stop = min(n, start + args.batch)
    block = np.asarray(src["figure2/maps"][start:stop], dtype=np.float32)
    block = weak_peak_transform(block, percentile=args.percentile, gamma=args.gamma)
    dst["fig2/x"][start:stop] = block.astype(out_dtype, copy=False)
```

Use `--batch 128` by default.

- [ ] **Step 4: Add cache paths to queue**

In `scripts/lno327_run_fig2_queue.sh`:

```bash
CACHE_N4096="${CACHE_N4096:-$CACHE_DIR/lno327_fig2_fwhm10_n4096_range200_p995_g050_f16.h5}"
CACHE_N32768="${CACHE_N32768:-$CACHE_DIR/lno327_fig2_fwhm10_n32768_range200_p995_g050_f16.h5}"
```

- [ ] **Step 5: Precompute before training**

After raw HDF5 validation:

```bash
"$PYTHON_BIN" scripts/lno327_precompute_fig2_inputs.py \
  --data "$DATA_N32768" \
  --out "$CACHE_N32768" \
  --percentile 99.5 \
  --gamma 0.5 \
  --dtype float16 \
  --batch 128 \
  --force
```

---

## Task 3: Train Without Keeping Full Fig.2 on GPU

**Files:**
- Modify: `inverse_spectrum/lno327_cnn.py`
- Modify: `scripts/lno327_run_cnn_task.py`

- [ ] **Step 1: Add training flags**

In `scripts/lno327_run_cnn_task.py`, add:

```python
parser.add_argument("--input-cache", default=None)
parser.add_argument("--input-device", choices=["auto", "cpu", "cuda"], default="auto")
```

Pass them into `train_cnn_baseline(...)`.

- [ ] **Step 2: Add the same function args**

In `inverse_spectrum/lno327_cnn.py`, add to `train_cnn_baseline`:

```python
input_cache: str | Path | None = None,
input_device: str = "auto",
```

- [ ] **Step 3: Choose default input location**

Use this rule:

```python
if input_device == "auto":
    n_samples = len(ds.params)
    input_device = "cpu" if mode == "fig2" and n_samples >= 8192 else device
```

For n32768 Fig.2, this means:

```text
model on GPU
input tensor on CPU
only current batch copied to GPU
```

- [ ] **Step 4: Use CPU indices when input is CPU**

Do not index CPU tensors with CUDA indices.

Use:

```python
order = np.random.permutation(indices)
for start in range(0, len(order), batch_size):
    idx_np = order[start:start + batch_size]
    fig2_batch = None if fig2 is None else fig2[idx_np].to(device, non_blocking=True)
    fig3_batch = None if fig3 is None else fig3[idx_np].to(device, non_blocking=True)
    target_batch = target[idx_np].to(device, non_blocking=True)
```

- [ ] **Step 5: Keep target small**

`params` target is tiny. It can stay on CPU or GPU. For the CPU-input path, move only the selected target batch to GPU.

- [ ] **Step 6: Keep old behavior for small runs**

For n4096 and smaller, allow `input_device=cuda` so the old fast path still works.

---

## Task 4: Make Fig.2 Queue Use Cache and Fallback Batches

**Files:**
- Modify: `scripts/lno327_run_fig2_queue.sh`

- [ ] **Step 1: n4096 training from cache**

Use:

```bash
--input-cache "$CACHE_N4096" \
--input-device auto
```

Runs:

```text
f2a_n4096_h128_e256_b128_e250
f2b_n4096_h192_e384_b96_e250
f2c_n4096_h256_e512_b64_e300
```

- [ ] **Step 2: n32768 training from cache**

Runs:

```text
f2d_n32768_h128_e256_b128_e180
f2e_n32768_h192_e384_b96_e180
f2f_n32768_h256_e512_b64_e220
```

- [ ] **Step 3: Add OOM fallback**

For each run, try larger batch first. If the log contains `CUDA out of memory`, rerun with the next batch:

```text
f2d: 192 -> 128 -> 96
f2e: 128 -> 96 -> 64
f2f: 96 -> 64 -> 48
```

- [ ] **Step 4: Never repeat completed work**

Before every stage:

```bash
if [[ -f "$out_dir/metrics.json" ]]; then
  echo "SKIP existing metrics"
  return 0
fi
```

Before every cache:

```bash
if [[ -f "$CACHE_N32768" ]]; then
  echo "SKIP existing cache"
fi
```

Before every raw dataset:

```bash
if [[ -f "$DATA_N32768" ]]; then
  echo "SKIP existing raw HDF5"
fi
```

---

## Task 5: Tests and Smoke Runs

**Files:**
- Create: `tests/test_lno327_fig2_cache_and_memory.py`

- [ ] **Step 1: Test in-place transform**

Test:

```python
arr = np.array([[[[0.0, 1.0], [4.0, 9.0]]]], dtype=np.float32)
before_id = id(arr)
out = weak_peak_transform(arr, percentile=100.0, gamma=0.5)
assert id(out) == before_id
assert out.dtype == np.float32
assert np.all(out >= 0.0)
assert np.all(out <= 1.0)
```

- [ ] **Step 2: Test Fig.2-only load skips Fig.3 arrays**

Create a tiny HDF5 with both Fig.2 and Fig.3.

Run:

```python
ds = load_lno327_inverse_dataset(path, load_figure3=False, load_figure2=True, load_figure4d=False)
assert ds.figure2_maps.shape == (4, 10, 8, 12)
assert ds.spectra.shape == (0, 0)
```

- [ ] **Step 3: Test cache shape**

Run the cache script on the tiny HDF5.

Expected:

```text
fig2/x shape = (4, 10, 8, 12)
fig2/x dtype = float16
params copied
split copied
```

- [ ] **Step 4: Remote smoke**

Run after sync:

```bash
ssh ins-cnn-cloud 'cd ~/work/ins-cnn && PYTHON_BIN=/home/linux/anaconda3/envs/torch2.4_cuda12.1/bin/python bash scripts/lno327_run_fig2_queue.sh'
```

For smoke, override:

```bash
SAMPLES=64
```

Expected:

```text
raw HDF5 generated on /data
cache generated on /data
2-epoch Fig.2 training writes metrics.json
```

---

## Execution Order

1. Implement `/data` routing first.
2. Implement Fig.2 cache second.
3. Implement CPU-batch training third.
4. Update queue fourth.
5. Run local tests.
6. Sync to remote.
7. Kill and restart only the waiting Fig.2 queue.
8. Do not stop current `b3-long`.

---

## Concrete Queue After Implementation

Final order:

```text
wait for current b3-long lock
generate /data/ins-cnn/data/Fig.2 n4096 raw HDF5
precompute /data/ins-cnn/cache/Fig.2 n4096 cache
train n4096 f2a/f2b/f2c
generate /data/ins-cnn/data/Fig.2 n32768 raw HDF5
precompute /data/ins-cnn/cache/Fig.2 n32768 cache
train n32768 f2d/f2e/f2f
write final status
```

The current code path should not repeatedly run weak-peak preprocessing for every model. It should run once per dataset and reuse the cached input.


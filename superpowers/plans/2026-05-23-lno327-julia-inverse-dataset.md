# LNO327 Julia Inverse Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first Julia-only dataset generation stage for La3Ni2O7 inverse fitting: sample exchange parameters, generate Sunny spectra, assign train/val/test splits by parameter sample, and write one reproducible HDF5 dataset.

**Architecture:** Add a small dataset layer beside `src/LNO327Reproduce.jl` instead of mixing training-dataset logic into the paper reproduction module. The new layer will reuse `LNOParams`, `LNOConfig`, `simulate_figure3_path`, and existing CIF/twin handling, then serialize spectra, parameters, split labels, and metadata into HDF5. The first dataset target is the Fig.3 high-symmetry E-Q spectrum because it is already implemented, physically meaningful, and cheaper than full Fig.2 maps.

**Tech Stack:** Julia, Sunny.jl, HDF5.jl, Test.jl, existing `src/LNO327Reproduce.jl`, shell scripts under `scripts/`.

---

## File Structure

- Create `src/LNO327InverseDataset.jl`: parameter ranges, deterministic sampling, split assignment, spectrum generation loop, HDF5 writer.
- Create `scripts/lno327_generate_inverse_dataset.jl`: command-line entrypoint for generating datasets.
- Create `tests/test_lno327_inverse_dataset.jl`: deterministic sampling, split integrity, HDF5 smoke test.
- Modify `Project.toml` only if a new dependency is truly needed. The expected implementation should use existing Julia stdlib plus HDF5/Sunny already present.
- Leave `src/LNO327Reproduce.jl` focused on forward physics/spectrum generation.

## Dataset Contract

The first HDF5 file should use this shape:

```text
params                      Float32[n_samples, 5]
param_names                 String[5]
split                       UInt8[n_samples]         # 0=train, 1=val, 2=test
figure3/spectra             Float32[n_samples, n_q, n_energy]
figure3/energies            Float64[n_energy]
figure3/axis                Float64[n_q]
figure3/tick_positions      Float64[5]
figure3/tick_labels         String[5]
paper_reference/params      Float32[5]
paper_reference/spectrum    Float32[n_q, n_energy]
```

Required attributes:

```text
seed
n_samples
train_fraction
val_fraction
test_fraction
parameter_sampling = "latin_hypercube"
target = "figure3_path"
fwhm
twin_weight
cif_path
cif_block
note = "Split is by full parameter sample, not by spectrum pixel."
```

Suggested first parameter ranges around the paper values:

```julia
Dict(
    "SJc"  => (30.0, 50.0),
    "SJ1a" => (0.5, 5.0),
    "SJ1b" => (1.0, 7.0),
    "SJ2"  => (2.0, 8.0),
    "SA"   => (-0.15, -0.005),
)
```

These are intentionally broad enough to test identifiability, but not so broad that Sunny spends the first run on unphysical spectra. Later phases can tighten or widen them after sensitivity checks.

## Task 1: Dataset Types And Sampling

**Files:**
- Create: `src/LNO327InverseDataset.jl`
- Test: `tests/test_lno327_inverse_dataset.jl`

- [ ] **Step 1: Write the failing tests**

Add this file:

```julia
include(joinpath(@__DIR__, "..", "src", "LNO327Reproduce.jl"))
include(joinpath(@__DIR__, "..", "src", "LNO327InverseDataset.jl"))

using .LNO327Reproduce
using .LNO327InverseDataset
using Test

@testset "LNO327 inverse dataset sampling" begin
    ranges = default_parameter_ranges()
    @test sort(collect(keys(ranges))) == sort(parameter_names())
    @test ranges["SJc"] == (30.0, 50.0)
    @test ranges["SA"] == (-0.15, -0.005)

    samples_a = sample_lno_params(6; seed=1234, ranges=ranges)
    samples_b = sample_lno_params(6; seed=1234, ranges=ranges)
    @test samples_a == samples_b
    @test length(samples_a) == 6
    @test all(p -> ranges["SJc"][1] <= p.SJc <= ranges["SJc"][2], samples_a)
    @test all(p -> ranges["SA"][1] <= p.SA <= ranges["SA"][2], samples_a)

    matrix = params_matrix(samples_a)
    @test size(matrix) == (6, 5)
    @test matrix[1, :] == Float32[samples_a[1].SJc, samples_a[1].SJ1a, samples_a[1].SJ1b, samples_a[1].SJ2, samples_a[1].SA]
end
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
julia --project=. tests/test_lno327_inverse_dataset.jl
```

Expected: FAIL because `src/LNO327InverseDataset.jl` does not exist or exports are missing.

- [ ] **Step 3: Implement minimal sampling module**

Create `src/LNO327InverseDataset.jl`:

```julia
module LNO327InverseDataset

using Random

using ..LNO327Reproduce

export default_parameter_ranges,
    sample_lno_params,
    params_matrix

function default_parameter_ranges()
    return Dict(
        "SJc" => (30.0, 50.0),
        "SJ1a" => (0.5, 5.0),
        "SJ1b" => (1.0, 7.0),
        "SJ2" => (2.0, 8.0),
        "SA" => (-0.15, -0.005),
    )
end

function _latin_hypercube(rng::AbstractRNG, n::Int, lo::Float64, hi::Float64)
    n > 0 || error("n must be positive, got $n")
    bins = ((0:n-1) .+ rand(rng, n)) ./ n
    shuffled = bins[randperm(rng, n)]
    return lo .+ shuffled .* (hi - lo)
end

function sample_lno_params(n::Int; seed::Integer=20260523, ranges=default_parameter_ranges())
    rng = MersenneTwister(seed)
    names = parameter_names()
    columns = Dict(name => _latin_hypercube(rng, n, Float64(ranges[name][1]), Float64(ranges[name][2])) for name in names)
    return [
        LNOParams(;
            SJc=columns["SJc"][i],
            SJ1a=columns["SJ1a"][i],
            SJ1b=columns["SJ1b"][i],
            SJ2=columns["SJ2"][i],
            SA=columns["SA"][i],
        )
        for i in 1:n
    ]
end

function params_matrix(params::AbstractVector{LNOParams})
    matrix = Matrix{Float32}(undef, length(params), length(parameter_names()))
    for (i, p) in enumerate(params)
        matrix[i, :] = Float32[p.SJc, p.SJ1a, p.SJ1b, p.SJ2, p.SA]
    end
    return matrix
end

end
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
julia --project=. tests/test_lno327_inverse_dataset.jl
```

Expected: PASS for the sampling testset.

## Task 2: Split Assignment By Parameter Sample

**Files:**
- Modify: `src/LNO327InverseDataset.jl`
- Modify: `tests/test_lno327_inverse_dataset.jl`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_lno327_inverse_dataset.jl`:

```julia
@testset "LNO327 inverse dataset splits" begin
    split = assign_splits(20; seed=7, train_fraction=0.70, val_fraction=0.15)
    @test length(split) == 20
    @test count(==(TRAIN_SPLIT), split) == 14
    @test count(==(VAL_SPLIT), split) == 3
    @test count(==(TEST_SPLIT), split) == 3
    @test split == assign_splits(20; seed=7, train_fraction=0.70, val_fraction=0.15)
end
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
julia --project=. tests/test_lno327_inverse_dataset.jl
```

Expected: FAIL because `assign_splits` and split constants are missing.

- [ ] **Step 3: Implement deterministic split assignment**

Update exports and add:

```julia
export TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    assign_splits

const TRAIN_SPLIT = UInt8(0)
const VAL_SPLIT = UInt8(1)
const TEST_SPLIT = UInt8(2)

function assign_splits(n::Int; seed::Integer=20260523, train_fraction::Float64=0.70, val_fraction::Float64=0.15)
    n > 0 || error("n must be positive, got $n")
    0 < train_fraction < 1 || error("train_fraction must be in (0, 1), got $train_fraction")
    0 <= val_fraction < 1 || error("val_fraction must be in [0, 1), got $val_fraction")
    train_fraction + val_fraction < 1 || error("train_fraction + val_fraction must be < 1")

    rng = MersenneTwister(seed)
    order = randperm(rng, n)
    n_train = round(Int, train_fraction * n)
    n_val = round(Int, val_fraction * n)
    n_test = n - n_train - n_val
    n_test > 0 || error("test split would be empty; increase n or reduce train/val fractions")

    split = fill(TEST_SPLIT, n)
    split[order[1:n_train]] .= TRAIN_SPLIT
    split[order[n_train+1:n_train+n_val]] .= VAL_SPLIT
    split[order[n_train+n_val+1:end]] .= TEST_SPLIT
    return split
end
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
julia --project=. tests/test_lno327_inverse_dataset.jl
```

Expected: PASS.

## Task 3: HDF5 Writer Smoke Test

**Files:**
- Modify: `src/LNO327InverseDataset.jl`
- Modify: `tests/test_lno327_inverse_dataset.jl`

- [ ] **Step 1: Write the failing HDF5 test**

Append:

```julia
using HDF5

@testset "LNO327 inverse dataset HDF5 smoke" begin
    out = joinpath(mktempdir(), "lno327_inverse_smoke.h5")
    cfg = LNOConfig(; n_q=5, n_energy=6, n_map_h=5, n_map_k=5, fwhm=2.0, twin_weight=0.5)

    write_inverse_dataset(out; n_samples=4, seed=99, cfg=cfg)

    h5open(out, "r") do h5
        @test size(h5["params"]) == (4, 5)
        @test size(h5["figure3/spectra"]) == (4, 5, 6)
        @test length(h5["split"][:]) == 4
        @test haskey(h5, "paper_reference/params")
        @test haskey(h5, "paper_reference/spectrum")
        @test attrs(h5)["target"] == "figure3_path"
        @test attrs(h5)["note"] == "Split is by full parameter sample, not by spectrum pixel."
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
julia --project=. tests/test_lno327_inverse_dataset.jl
```

Expected: FAIL because `write_inverse_dataset` is missing.

- [ ] **Step 3: Implement HDF5 writer**

Add `using HDF5` to the module and export `write_inverse_dataset`. Add:

```julia
export write_inverse_dataset

function write_inverse_dataset(
    out_path::AbstractString;
    n_samples::Int,
    seed::Integer=20260523,
    cfg::LNOConfig=LNOConfig(),
    ranges=default_parameter_ranges(),
    train_fraction::Float64=0.70,
    val_fraction::Float64=0.15,
)
    params = sample_lno_params(n_samples; seed=seed, ranges=ranges)
    split = assign_splits(n_samples; seed=seed + 1, train_fraction=train_fraction, val_fraction=val_fraction)

    first_spectrum, energies, ticks, labels, _ = simulate_figure3_path(params[1]; cfg=cfg)
    spectra = Array{Float32}(undef, n_samples, size(first_spectrum, 1), size(first_spectrum, 2))
    spectra[1, :, :] = first_spectrum

    for i in 2:n_samples
        spectrum, sample_energies, sample_ticks, sample_labels, _ = simulate_figure3_path(params[i]; cfg=cfg)
        size(spectrum) == size(first_spectrum) || error("sample $i spectrum size $(size(spectrum)) differs from first sample $(size(first_spectrum))")
        sample_energies == energies || error("sample $i energy axis differs from first sample")
        sample_ticks == ticks || error("sample $i tick positions differ from first sample")
        sample_labels == labels || error("sample $i tick labels differ from first sample")
        spectra[i, :, :] = spectrum
    end

    paper_spectrum, paper_energies, paper_ticks, paper_labels, _ = simulate_figure3_path(default_params(); cfg=cfg)
    paper_energies == energies || error("paper reference energy axis differs from sampled spectra")
    paper_ticks == ticks || error("paper reference tick positions differ from sampled spectra")
    paper_labels == labels || error("paper reference tick labels differ from sampled spectra")

    mkpath(dirname(out_path))
    h5open(out_path, "w") do h5
        h5["params"] = params_matrix(params)
        h5["param_names"] = parameter_names()
        h5["split"] = split
        h5["figure3/spectra"] = spectra
        h5["figure3/energies"] = energies
        h5["figure3/axis"] = Float64.(collect(range(0.0, 1.0; length=size(first_spectrum, 1))))
        h5["figure3/tick_positions"] = ticks
        h5["figure3/tick_labels"] = labels
        h5["paper_reference/params"] = params_matrix([default_params()])[1, :]
        h5["paper_reference/spectrum"] = paper_spectrum
        attrs(h5)["seed"] = seed
        attrs(h5)["n_samples"] = n_samples
        attrs(h5)["train_fraction"] = train_fraction
        attrs(h5)["val_fraction"] = val_fraction
        attrs(h5)["test_fraction"] = 1.0 - train_fraction - val_fraction
        attrs(h5)["parameter_sampling"] = "latin_hypercube"
        attrs(h5)["target"] = "figure3_path"
        attrs(h5)["fwhm"] = cfg.fwhm
        attrs(h5)["twin_weight"] = cfg.twin_weight
        attrs(h5)["cif_path"] = cfg.cif_path
        attrs(h5)["cif_block"] = cfg.cif_block
        attrs(h5)["note"] = "Split is by full parameter sample, not by spectrum pixel."
    end
    return out_path
end
```

- [ ] **Step 4: Run smoke test**

Run:

```bash
julia --project=. tests/test_lno327_inverse_dataset.jl
```

Expected: PASS. The smoke dataset should generate quickly because it uses `n_samples=4`, `n_q=5`, and `n_energy=6`.

## Task 4: CLI Dataset Generator

**Files:**
- Create: `scripts/lno327_generate_inverse_dataset.jl`
- Modify: `tests/test_lno327_inverse_dataset.jl`

- [ ] **Step 1: Add a CLI smoke test by invoking the script**

Append:

```julia
@testset "LNO327 inverse dataset CLI" begin
    out = joinpath(mktempdir(), "cli_inverse_smoke.h5")
    script = joinpath(@__DIR__, "..", "scripts", "lno327_generate_inverse_dataset.jl")
    cmd = `$(Base.julia_cmd()) --project=$(joinpath(@__DIR__, "..")) $script --out $out --n-samples 3 --seed 123 --nq 4 --ne 5 --fwhm 2 --twin-weight 0.5`
    run(cmd)
    h5open(out, "r") do h5
        @test size(h5["params"]) == (3, 5)
        @test size(h5["figure3/spectra"]) == (3, 4, 5)
        @test attrs(h5)["seed"] == 123
        @test attrs(h5)["twin_weight"] ≈ 0.5
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
julia --project=. tests/test_lno327_inverse_dataset.jl
```

Expected: FAIL because the CLI script does not exist.

- [ ] **Step 3: Create CLI script**

Create `scripts/lno327_generate_inverse_dataset.jl`:

```julia
#!/usr/bin/env julia

include(joinpath(@__DIR__, "..", "src", "LNO327Reproduce.jl"))
include(joinpath(@__DIR__, "..", "src", "LNO327InverseDataset.jl"))

using .LNO327Reproduce
using .LNO327InverseDataset

function parse_args(args)
    opts = Dict{String,String}()
    i = 1
    while i <= length(args)
        key = args[i]
        startswith(key, "--") || error("Expected --key, got $key")
        i == length(args) && error("Missing value for $key")
        opts[key[3:end]] = args[i + 1]
        i += 2
    end
    return opts
end

getint(opts, key, default) = haskey(opts, key) ? parse(Int, opts[key]) : default
getfloat(opts, key, default) = haskey(opts, key) ? parse(Float64, opts[key]) : default
getbool(opts, key, default) = haskey(opts, key) ? lowercase(opts[key]) in ("1", "true", "yes", "y") : default

opts = parse_args(ARGS)
out = get(opts, "out", joinpath(@__DIR__, "..", "data", "lno327_inverse_dataset.h5"))

cfg = LNOConfig(;
    use_cif_geometry=getbool(opts, "use-cif-geometry", true),
    cif_path=get(opts, "cif-path", LNO327Reproduce.DEFAULT_LNO327_CIF_PATH),
    cif_block=get(opts, "cif-block", "data_Neutron"),
    n_q=getint(opts, "nq", 160),
    n_energy=getint(opts, "ne", 180),
    energy_max=getfloat(opts, "energy-max", 85.0),
    fwhm=getfloat(opts, "fwhm", 1.0),
    twin_weight=getfloat(opts, "twin-weight", 0.5),
)

write_inverse_dataset(
    out;
    n_samples=getint(opts, "n-samples", 128),
    seed=getint(opts, "seed", 20260523),
    cfg=cfg,
    train_fraction=getfloat(opts, "train-fraction", 0.70),
    val_fraction=getfloat(opts, "val-fraction", 0.15),
)

println("Saved La3Ni2O7 inverse dataset to $(abspath(out))")
```

- [ ] **Step 4: Run CLI test**

Run:

```bash
julia --project=. tests/test_lno327_inverse_dataset.jl
```

Expected: PASS.

## Task 5: First Real Dataset And Sanity Report

**Files:**
- Create: `data/lno327_inverse_fig3_n128_q160_e180.h5`
- Create: `runs/lno327_inverse_dataset_sanity/report.md`

- [ ] **Step 1: Generate a modest first dataset**

Run:

```bash
julia --project=. scripts/lno327_generate_inverse_dataset.jl \
  --out data/lno327_inverse_fig3_n128_q160_e180.h5 \
  --n-samples 128 \
  --seed 20260523 \
  --nq 160 \
  --ne 180 \
  --energy-max 85 \
  --fwhm 1 \
  --twin-weight 0.5 \
  --cif-path cif/LaNiO327-cbk.cif \
  --cif-block data_Neutron
```

Expected: HDF5 exists and contains 128 spectra. This may take a while because every sample runs Sunny.

- [ ] **Step 2: Inspect split counts**

Run:

```bash
.venv/bin/python - <<'PY'
import h5py
import numpy as np
path = "data/lno327_inverse_fig3_n128_q160_e180.h5"
with h5py.File(path, "r") as h5:
    split = h5["split"][:]
    print("params", h5["params"].shape)
    print("spectra", h5["figure3/spectra"].shape)
    print("train", int(np.sum(split == 0)))
    print("val", int(np.sum(split == 1)))
    print("test", int(np.sum(split == 2)))
    print("seed", h5.attrs["seed"])
    print("target", h5.attrs["target"])
PY
```

Expected:

```text
params (128, 5)
spectra (128, 160, 180)
train 90
val 19
test 19
seed 20260523
target figure3_path
```

- [ ] **Step 3: Write a short sanity report**

Create `runs/lno327_inverse_dataset_sanity/report.md`:

```markdown
# LNO327 Inverse Dataset Sanity Report

Dataset: `data/lno327_inverse_fig3_n128_q160_e180.h5`

## What This Dataset Is

This is the first Julia/Sunny-generated inverse-fitting dataset for La3Ni2O7. Each sample is one full parameter vector and one full Fig.3-path spin-wave spectrum. The train/val/test split is assigned by sample, not by pixels.

## Split Counts

- train: 90
- val: 19
- test: 19

## Parameters

- SJc: 30.0 to 50.0 meV
- SJ1a: 0.5 to 5.0 meV
- SJ1b: 1.0 to 7.0 meV
- SJ2: 2.0 to 8.0 meV
- SA: -0.15 to -0.005 meV

## What To Check Before Training

1. The paper parameter vector is stored separately under `paper_reference`.
2. The split is reproducible from seed `20260523`.
3. Spectra are generated with twin_weight = 0.5 and FWHM = 1 meV.
4. This dataset only uses Fig.3-path spectra. Fig.2 maps should be added in a subsequent dataset after the first inverse model works.
```

## Verification

Run these after implementation:

```bash
julia --project=. tests/test_lno327_inverse_dataset.jl
julia --project=. tests/test_lno327_reproduce.jl
julia --project=. tests/test_effective_params.jl
.venv/bin/python -m compileall -q inverse_spectrum tests
.venv/bin/python -m unittest tests.test_sensitivity_plots -v
```

Expected:

- `tests/test_lno327_inverse_dataset.jl`: all new tests pass.
- `tests/test_lno327_reproduce.jl`: current La3Ni2O7 reproduction tests still pass.
- `tests/test_effective_params.jl`: existing effective-parameter tests still pass.
- Python compile/test suite still passes.

## Explicit Non-Goals For This First Stage

- Do not train the neural network in this task.
- Do not add Fig.2 maps to the training input yet.
- Do not optimize directly against experimental data yet.
- Do not change the physical Hamiltonian beyond reusing the current `LNO327Reproduce.jl` forward model.

## Next Step After This Plan

After the dataset exists and passes sanity checks, the next task should be a tiny baseline inverse model:

1. Load `figure3/spectra`.
2. Flatten or lightly normalize each spectrum.
3. Train a simple ridge regression or small MLP to predict `[SJc, SJ1a, SJ1b, SJ2, SA]`.
4. Evaluate on test split.
5. Predict parameters from `paper_reference/spectrum` and check whether it returns values near the paper parameters.

This keeps the first inverse-fitting loop honest before spending time on larger models.

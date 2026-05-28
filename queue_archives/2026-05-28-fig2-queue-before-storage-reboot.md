# Fig.2 Queue Archive Before Storage Reboot

Created: 2026-05-28 12:24 CST

Purpose: save the current waiting queue before the remote machine is rebooted for storage expansion. After reboot, use this file to restore the Fig.2 tasks.

## Current Remote State

Remote:

```text
ssh alias: ins-cnn-cloud
repo: /home/linux/work/ins-cnn
```

Disk state at archive time:

```text
/        system disk: /dev/sda2, 197 GB total, 104 GB free
/data    data disk:   /dev/sdb,   98 GB total,  93 GB free
```

GPU state at archive time:

```text
GPU utilization: 97%
GPU memory: 31290 / 32760 MiB
```

Running training at archive time:

```text
PID: 1376704
task: b3_long_n32768_h192_e384_b192_e500
data: data/lno327_inverse_fig3only_fwhm10_n32768_q160_e180_range200.h5
out:  runs/fig3_32768_range200/b3_long_n32768_h192_e384_b192_e500
mode: fig3
posterior: gaussian
epochs: 500
batch: 192
hidden: 192
embedding: 384
seed: 20260803
```

Latest observed `b3-long` training curve tail:

```text
epoch 84 train_nll=-13.428885539372763 val_nll=-14.424419953272892
epoch 85 train_nll=-13.308111862341564 val_nll=-13.674969416398268
epoch 86 train_nll=-13.41909945011139  val_nll=-14.028049579033485
epoch 87 train_nll=-13.581602509816488 val_nll=-14.282711579249455
epoch 88 train_nll=-13.472830939292908 val_nll=-13.945513211763823
```

This running task was **not cancelled** by this archive step.

## Cancelled Waiting Queue

Waiting queue process:

```text
PID: 1409115
command: bash scripts/lno327_run_fig2_queue.sh
state: WAIT shared lock: runs/fig3_32768_range200/pipeline.lock
started: 2026-05-28 12:07:53 CST
```

Important: this Fig.2 queue had not acquired the lock yet. It had not generated any Fig.2 data and had not started any Fig.2 training.

The waiting queue was cancelled so the machine can be rebooted safely for storage expansion.

Cancellation confirmation:

```text
cancelled PID: 1409115
confirmed at: 2026-05-28 12:26 CST
remote archive copied to:
  /home/linux/work/ins-cnn/docs/queue_archives/2026-05-28-fig2-queue-before-storage-reboot.md
```

## Queued Tasks That Need Restoring

The old remote queue contained these tasks:

```text
1. Generate Fig.2+Fig.3 n=4096 dataset
   output: data/lno327_inverse_fig2fig3_fwhm10_n4096_q160_e180_h81_k121_range200.h5
   FWHM: 10 meV
   parameter range:
     SJc:   10.0 to 70.0
     SJ1a:  -2.7 to 8.9
     SJ1b:  -3.25 to 11.75
     SJ2:   -3.0 to 13.0
     SA:    -0.2995 to 0.0985
   Julia workers fallback:
     12 -> 8 -> 6

2. Validate HDF5 shape
   expected:
     params: (4096, 5)
     figure2/maps: (4096, 10, 81, 121)
     figure3/spectra: (4096, 160, 180)

3. Train Fig.2-only CNN f2a
   run: f2a_n4096_h128_e256_b128_e250
   hidden: 128
   embedding: 256
   batch: 128
   epochs: 250

4. Train Fig.2-only CNN f2b
   run: f2b_n4096_h192_e384_b96_e250
   hidden: 192
   embedding: 384
   batch: 96
   epochs: 250

5. Train Fig.2-only CNN f2c
   run: f2c_n4096_h256_e512_b64_e300
   hidden: 256
   embedding: 512
   batch: 64
   epochs: 300
```

## Updated Plan After Storage Expansion

Do not simply rerun the old queue unchanged.

After reboot, first switch new large data to `/data`, then run the expanded Fig.2 queue.

Use this implementation plan:

```text
docs/superpowers/plans/2026-05-28-lno327-fig2-data-disk-memory.md
```

The intended new layout is:

```text
/data/ins-cnn/data     raw HDF5 datasets
/data/ins-cnn/chunks   temporary chunk files
/data/ins-cnn/cache    preprocessed CNN input caches
~/work/ins-cnn         code and small logs
```

The intended new queue is:

```text
1. Wait for current GPU lock.
2. Generate Fig.2 n=4096 raw HDF5 on /data.
3. Precompute Fig.2 n=4096 CNN cache on /data.
4. Train n=4096 Fig.2-only CNN: f2a, f2b, f2c.
5. Generate Fig.2 n=32768 raw HDF5 on /data.
6. Precompute Fig.2 n=32768 CNN cache on /data.
7. Train n=32768 Fig.2-only CNN: f2d, f2e, f2f.
```

Recommended n32768 runs:

```text
f2d_n32768_h128_e256
  hidden: 128
  embedding: 256
  batch fallback: 192 -> 128 -> 96
  epochs: 180

f2e_n32768_h192_e384
  hidden: 192
  embedding: 384
  batch fallback: 128 -> 96 -> 64
  epochs: 180

f2f_n32768_h256_e512
  hidden: 256
  embedding: 512
  batch fallback: 96 -> 64 -> 48
  epochs: 220
```

## Memory and GPU Rules

For n32768 Fig.2:

```text
Do not keep the full Fig.2 tensor on GPU.
Keep the full input on CPU.
Move only the current batch to GPU.
```

Precompute the weak-peak transform once:

```text
raw figure2/maps -> preprocessed cache
percentile: 99.5
gamma: 0.5
cache dtype: float16
```

This avoids doing the same preprocessing again for every model.

## Restore Checklist After Reboot

Run these checks first:

```bash
ssh ins-cnn-cloud 'hostname; date; df -h / /data; nvidia-smi'
```

Then check repo:

```bash
ssh ins-cnn-cloud 'cd ~/work/ins-cnn && git status --short && ls -lh scripts/lno327_run_fig2_queue.sh'
```

Then implement or sync the `/data` and cache-aware changes described in:

```text
docs/superpowers/plans/2026-05-28-lno327-fig2-data-disk-memory.md
```

Then launch the restored queue:

```bash
ssh ins-cnn-cloud 'cd ~/work/ins-cnn && mkdir -p runs/fig2_queue && nohup bash scripts/lno327_run_fig2_queue.sh > runs/fig2_queue/nohup.log 2>&1 & echo $! > runs/fig2_queue/queue.pid'
```

Then verify:

```bash
ssh ins-cnn-cloud 'cd ~/work/ins-cnn && cat runs/fig2_queue/queue.pid && ps -fp $(cat runs/fig2_queue/queue.pid) && tail -30 runs/fig2_queue/pipeline.log'
```

## Notes

- The old waiting queue only had n4096 tasks.
- The desired post-expansion queue should include both n4096 and n32768.
- The current local repo has an implementation plan for `/data` routing and cache-aware training, but this must be verified and synced after reboot.
- Do not delete old Fig.3 results.
- Do not assume `/data` survived with the same UUID after expansion; always run `df -h /data` first.

# LNO327 CNN weak-peak / 4D fusion summary

本文档记录当前 LNO327 反向拟合主线。核心目标是用 Sunny.jl 生成
La3Ni2O7 / LNO327 的 spin-wave spectra，再训练模型从谱图反推参数：

```text
[SJc, SJ1a, SJ1b, SJ2, SA]
```

当前结论很明确：CNN 是主线。峰位置、峰形、肩峰、弱峰和强度分布都在传递参数信息，
不能只把谱图展平成向量后交给 ridge 或普通 MLP。

## 当前最好结果

4D 数据集：

```text
data/lno327_inverse_4d_n512_q160_e180_h81_k121_l5.h5
figure3/spectra: (512, 160, 180)
figure4d/volumes: (512, 10, 5, 81, 121)
split: train/val/test = 360/76/76
```

| 方法 | test mean range MAE | 备注 |
| --- | ---: | --- |
| ridge Fig.3 sqrt n512 | 11.19% | 历史向量 baseline |
| CNN Fig.3 weak-peak n512 | 4.07% | 历史 Fig.3-only，旧 split |
| CNN 2D gated ensemble n512 | 4.01% | 历史 Fig.2+Fig.3 gated ensemble |
| Fig.3-only on 4D split | 9.14% | 同一个 4D HDF5 split 的公平对照 |
| gated4d | 4.68% | 第一版 4D gated fusion |
| residual4d fixed Fig.3 | 4.63% | Fig.3 warm-start 后冻结，4D 学 residual |
| gated4d + residual fixed + two-stage per-param ensemble | 4.14% | head-only sweep 前的 4D 最好结果 |
| residual4d head-only aux0 cooldown single | 3.77% | `2e-3 -> 5e-4 -> 2e-4` |
| gated4d + cooldown residual global ensemble | 3.67% | 当前最好 global ensemble |
| gated4d + cooldown residual per-param ensemble | 3.64% | 当前最好整体结果 |
| wide Fig.3-only n512 | 5.45% | 更大参数范围，同一 wide split |
| wide gated4d n512 | 4.62% | wide 范围下 4D 明确优于 Fig.3-only |
| wide gated4d + residual head cooldown ensemble | 3.80% | wide 范围当前最好结果，第二 seed |

当前最佳 checkpoint 组合：

```text
runs/lno327_inverse_cnn_weakpeak_n512/gated4d/cnn.pt
runs/lno327_inverse_cnn_weakpeak_n512/residual4d_resume_heads_lr2e3aux000_lr5e4_then_lr2e4_e60_aux000_seed20260610/cnn.pt
```

最佳 ensemble 输出目录：

```text
runs/lno327_inverse_cnn_weakpeak_n512/gated4d_resumeheads_lr2e3aux000_lr5e4_then_lr2e4_ensemble_grid101
```

## 为什么这条路线有效

有效策略不是简单把 Fig.3 和 4D maps 拼起来，而是：

1. 让 Fig.3 作为主证据。
2. 用 4D 分支学习 Fig.3 没有解释掉的 residual correction。
3. 后期只训练 heads，不再扰动已经学好的 encoder。
4. 去掉 auxiliary heads 的 loss，让训练目标集中到 fused prediction。
5. 用较高学习率把 residual head 推到新 basin，再用低学习率 cooldown。

最终使用的 head-only cooldown：

```text
resume fixed residual4d checkpoint
trainable_scope = heads
aux_weight = 0.0
lr schedule by separate runs: 2e-3 -> 5e-4 -> 2e-4
```

第三段 `1e-4` cooldown 没有继续改善，说明 `2e-4` 附近已经接近当前甜点。

## 最佳模型分参数表现

当前最好 per-parameter ensemble 的 test 表现：

| 参数 | MAE | RMSE | R2 | MAE/range |
| --- | ---: | ---: | ---: | ---: |
| SJc | 1.0902 | 1.5767 | 0.9330 | 5.47% |
| SJ1a | 0.1450 | 0.1909 | 0.9776 | 3.24% |
| SJ1b | 0.1797 | 0.2469 | 0.9703 | 3.02% |
| SJ2 | 0.1434 | 0.1992 | 0.9863 | 2.40% |
| SA | 0.00591 | 0.01028 | 0.9325 | 4.08% |

对论文参考参数的预测：

| 参数 | reference | prediction | delta |
| --- | ---: | ---: | ---: |
| SJc | 39.86 | 41.21 | +1.35 |
| SJ1a | 2.36 | 2.383 | +0.023 |
| SJ1b | 3.71 | 3.689 | -0.021 |
| SJ2 | 4.63 | 4.509 | -0.121 |
| SA | -0.070 | -0.0611 | +0.0089 |

## 参数范围扩展

数据生成脚本已经支持更大的参数范围：

```bash
julia --project=. scripts/lno327_generate_inverse_dataset.jl \
  --out data/lno327_inverse_4d_wide_n512_q160_e180_h81_k121_l5.h5 \
  --n-samples 512 \
  --include-4d-volume \
  --nq 160 --ne 180 \
  --n-map-h 81 --n-map-k 121 --n-map-l 5 \
  --range-preset wide \
  --seed 20260525
```

也可以用 `--range-scale` 或每个参数的 `--sjc-min/max`、`--sj1a-min/max`、
`--sj1b-min/max`、`--sj2-min/max`、`--sa-min/max` 做更细控制。

当前 `wide` preset：

| 参数 | 范围 |
| --- | ---: |
| SJc | 25 到 55 |
| SJ1a | 0.2 到 6.0 |
| SJ1b | 0.5 到 8.0 |
| SJ2 | 1.0 到 9.0 |
| SA | -0.20 到 -0.001 |

本机已经完成过 full n=512 wide-range 4D 数据集生成：

```text
data/lno327_inverse_4d_wide_n512_q160_e180_h81_k121_l5.h5
figure3/spectra: (512, 160, 180)
figure4d/volumes: (512, 10, 5, 81, 121)
param min: SJc 25.04, SJ1a 0.201, SJ1b 0.701, SJ2 1.062, SA -0.1998
param max: SJc 54.97, SJ1a 5.985, SJ1b 7.996, SJ2 8.983, SA -0.00137
```

也完成过 wide-range smoke test：

```bash
$env:JULIA_DEPOT_PATH='D:\DEV\ins-cnn\.julia-depot'
.\.julia-runtime\julia-1.12.5\bin\julia.exe --project=. scripts\lno327_generate_inverse_dataset.jl \
  --out data\smoke_lno327_wide_ranges.h5 \
  --n-samples 4 --nq 8 --ne 10 --n-map-h 8 --n-map-k 9 \
  --range-preset wide --seed 20260525 --force
```

## 复现实验命令

从 fixed residual4d checkpoint 开始做 head-only aux0 高学习率：

```bash
.\.venv-gpu\Scripts\python.exe scripts\lno327_run_cnn_task.py \
  --data data\lno327_inverse_4d_n512_q160_e180_h81_k121_l5.h5 \
  --out runs\lno327_inverse_cnn_weakpeak_n512\residual4d_resume_heads_lr2e3_e60_aux000_seed20260605 \
  --mode residual4d \
  --epochs 60 --batch-size 32 --hidden 32 --embedding 64 \
  --lr 0.002 --weight-decay 0.001 \
  --percentile 99.5 --gamma 0.5 \
  --seed 20260605 --device cuda --log-every 10 \
  --aux-weight 0.0 \
  --resume-checkpoint runs\lno327_inverse_cnn_weakpeak_n512\residual4d_warm_fig3_fixed_e120_aux003_seed20260524\cnn.pt \
  --trainable-scope heads --patience 60
```

然后依次从前一段 checkpoint cooldown：

```text
lr = 5e-4, epochs = 60
lr = 2e-4, epochs = 60
```

最后做 ensemble：

```bash
.\.venv-gpu\Scripts\python.exe -m inverse_spectrum.ensemble_lno327_cnn \
  --data data\lno327_inverse_4d_n512_q160_e180_h81_k121_l5.h5 \
  --checkpoints \
    runs\lno327_inverse_cnn_weakpeak_n512\gated4d\cnn.pt \
    runs\lno327_inverse_cnn_weakpeak_n512\residual4d_resume_heads_lr2e3aux000_lr5e4_then_lr2e4_e60_aux000_seed20260610\cnn.pt \
  --labels gated4d resume_heads_lr2e3aux000_lr5e4_then_lr2e4 \
  --out runs\lno327_inverse_cnn_weakpeak_n512\gated4d_resumeheads_lr2e3aux000_lr5e4_then_lr2e4_ensemble_grid101 \
  --grid 101 --batch-size 32 --device cuda
```

## 下一步

1. 把当前 narrow-range 4D 最好结果固定为 baseline：per-param ensemble 3.64%。
2. wide-range n512 已经跑通，并在更大参数范围上达到 global ensemble 3.80%。
3. 第二个 wide residual head seed 已经复现趋势并略微刷新；三模型 ensemble 没有进一步改善。
4. 如果 wide-range 还要继续增强，优先改采样和 split stratification，再考虑扩大模型。
5. 解冻 4D encoder 在 GTX 1050 Ti 上成本太高，除非降低 map 分辨率或换更强 GPU，否则不作为首选。

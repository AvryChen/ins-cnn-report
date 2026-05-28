# 云端训练状态报告

更新时间：2026-05-26 23:46 CST

## 当前阶段

pure-wide n1024 数据集已经重建完成，当前进入 PyTorch/CUDA 训练阶段。训练代码已经加入可选 AMP：

- `scripts/lno327_run_cnn_task.py` 支持 `--amp --amp-dtype float16|bfloat16`
- `scripts/lno327_cloud_gpu_sweep.py` 支持 AMP sweep
- `scripts/lno327_tune_cnn_batch.py` 支持 AMP batch 探测
- `inverse_spectrum/lno327_cnn.py` 在训练、验证、预测和解释性输出中使用 autocast；float16 使用 GradScaler

本地语法检查通过；远端 CUDA 环境 import/help 检查通过；远端 smoke 数据集上 AMP 1 epoch 训练通过，日志显示 `AMP enabled dtype=float16 with GradScaler`。

## 数据集状态

目标数据集已经生成：

```text
data/lno327_inverse_4d_wide_extra_augtrain_n1024_q160_e180_h81_k121_l5.h5  2.0G
```

中间数据：

```text
data/lno327_inverse_4d_wide_n512_q160_e180_h81_k121_l5.h5        1016M
data/lno327_inverse_4d_wide_extra_n512_q160_e180_h81_k121_l5.h5  1016M
```

生成方式：base-wide 和 extra-wide 各 16 个 chunk，每个 chunk 32 个样本，最多 10 个 Julia worker 并行。这个阶段已经结束。

## AMP Sweep 结论

已在 n1024 真实数据集上运行 AMP float16 GPU sweep：

```text
runs/lno327_inverse_cnn_weakpeak_n512/gpu_sweep_amp_n1024_20260526/
```

主要结果：

| mode | hidden | embedding | batch | 状态 | 峰值显存 | samples/s |
|---|---:|---:|---:|---|---:|---:|
| gated4d | 32 | 64 | 128 | OK | 27391 MB | 29.2 |
| gated4d | 32 | 64 | 160/192 | OOM | - | - |
| gated4d | 48 | 96 | 96/128 | OOM | - | - |
| residual4d | 32 | 64 | 128 | OK | 27391 MB | 27.8 |
| residual4d | 32 | 64 | 160/192 | OOM | - | - |
| residual4d | 48 | 96 | 96/128 | OOM | - | - |

结论：AMP float16 可用，但没有把 batch 上限推到 160。当前 4080S 上最稳的正式配置仍然是 `hidden=32, embedding=64, batch=128`。更大模型和更大 batch 会爆显存。

## 已完成正式训练：gated4d AMP float16

gated4d AMP float16 正式训练已经完成：

```bash
~/anaconda3/envs/torch2.4_cuda12.1/bin/python scripts/lno327_run_cnn_task.py \
  --data data/lno327_inverse_4d_wide_extra_augtrain_n1024_q160_e180_h81_k121_l5.h5 \
  --out runs/lno327_inverse_cnn_weakpeak_n512/gated4d_wide_extra_augtrain_scratch_e280_aux015_seed20260705_ampfp16_b128 \
  --mode gated4d \
  --epochs 280 \
  --batch-size 128 \
  --hidden 32 \
  --embedding 64 \
  --lr 1e-3 \
  --weight-decay 1e-3 \
  --aux-weight 0.15 \
  --seed 20260705 \
  --device cuda \
  --log-every 10 \
  --patience 80 \
  --amp \
  --amp-dtype float16
```

GPU 监控：

```text
runs/lno327_inverse_cnn_weakpeak_n512/gated4d_wide_extra_augtrain_scratch_e280_aux015_seed20260705_ampfp16_b128/gpu_trace.csv
```

最终状态：

```text
AMP enabled dtype=float16 with GradScaler
best_epoch=274
best_val_loss=0.00290311
test mean range-normalized MAE=4.2938%
GPU: peak 29730 MiB / 32760 MiB, avg utilization 95.2%, peak power 293.41 W
```

per-parameter test error：

| 参数 | range MAE | MAE | RMSE | 说明 |
|---|---:|---:|---:|---|
| SJc | 5.7731% | 1.7269 | 2.4864 | 层间/双层方向交换 |
| SJ1a | 3.8126% | 0.2206 | 0.2905 | 面内 a 方向最近邻交换 |
| SJ1b | 3.6834% | 0.2723 | 0.3733 | 面内 b 方向最近邻交换 |
| SJ2 | 2.5685% | 0.2053 | 0.2854 | 面内次近邻交换 |
| SA | 5.6317% | 0.01115 | 0.01462 | 单离子各向异性 |

判断：AMP float16 比同 seed FP32 的 `4.5813%` 略好，但仍差于历史 pure-wide ensemble `2.3839%`。因此下一步不是宣称刷新结果，而是继续跑 residual4d AMP seed20260706，并在两个模型完成后做 validation-selected ensemble。

## 已完成正式训练：residual4d AMP float16

residual4d AMP float16 正式训练已经完成：

```bash
~/anaconda3/envs/torch2.4_cuda12.1/bin/python scripts/lno327_run_cnn_task.py \
  --data data/lno327_inverse_4d_wide_extra_augtrain_n1024_q160_e180_h81_k121_l5.h5 \
  --out runs/lno327_inverse_cnn_weakpeak_n512/residual4d_wide_extra_augtrain_scratch_e280_aux015_seed20260706_ampfp16_b128 \
  --mode residual4d \
  --epochs 280 \
  --batch-size 128 \
  --hidden 32 \
  --embedding 64 \
  --lr 1e-3 \
  --weight-decay 1e-3 \
  --aux-weight 0.15 \
  --seed 20260706 \
  --device cuda \
  --log-every 10 \
  --patience 80 \
  --amp \
  --amp-dtype float16
```

run 目录：

```text
runs/lno327_inverse_cnn_weakpeak_n512/residual4d_wide_extra_augtrain_scratch_e280_aux015_seed20260706_ampfp16_b128
```

结果：

```text
best_epoch=278
best_val_loss=0.00519702
test mean range-normalized MAE=4.5626%
GPU: peak 29730 MiB / 32760 MiB, avg utilization 95.6%, peak power 292.58 W
```

per-parameter test error：

| 参数 | range MAE | MAE | RMSE |
|---|---:|---:|---:|
| SJc | 5.8383% | 1.7464 | 2.4205 |
| SJ1a | 4.0512% | 0.2344 | 0.2890 |
| SJ1b | 4.0860% | 0.3020 | 0.3775 |
| SJ2 | 2.7721% | 0.2216 | 0.3040 |
| SA | 6.0657% | 0.01201 | 0.01664 |

## Ensemble 结果

已生成 gated/residual 二模型 validation-selected ensemble：

```text
runs/lno327_inverse_cnn_weakpeak_n512/gated_residual_ampfp16_b128_seed20260705_20260706_ensemble_grid201
```

| 方法 | test mean range MAE | 说明 |
|---|---:|---|
| gated4d AMP seed20260705 | 4.2938% | 两个 AMP 单模型中最好 |
| residual4d AMP seed20260706 | 4.5626% | 明显差于 gated4d |
| global ensemble | 4.1060% | 权重 gated 0.765 / residual 0.235 |
| per-parameter ensemble | 4.1586% | 本轮不如 global ensemble |

global ensemble per-parameter test error：

| 参数 | range MAE | MAE | RMSE |
|---|---:|---:|---:|
| SJc | 5.6377% | 1.6864 | 2.3718 |
| SJ1a | 3.6105% | 0.2089 | 0.2702 |
| SJ1b | 3.5676% | 0.2637 | 0.3521 |
| SJ2 | 2.4127% | 0.1928 | 0.2773 |
| SA | 5.3016% | 0.01050 | 0.01338 |

结论：global ensemble 比两个单模型都好，但仍比历史 pure-wide 最好 `2.3839%` 差 `+1.7221` 个百分点，不能计为刷新。

## 报告与脚本修复

已重跑四个报告脚本：

```text
scripts/lno327_write_cloud_fwhm10_report.py
scripts/lno327_write_4d_final_report.py
scripts/lno327_write_training_visual_report.py
scripts/lno327_write_data_parameter_report.py
```

前三个直接完成；`scripts/lno327_write_data_parameter_report.py` 有 f-string 嵌套引号语法错误，已最小修复并成功生成：

```text
runs/lno327_4d_task/lno327_data_parameter_report.html
```

公开报告发布状态：`scripts/cloud_netcheck.sh` 显示 `git@github.com: Permission denied (publickey)`，因此当前机器还不能通过 deploy key 发布到 `AvryChen/ins-cnn-report`。

本轮已执行：

```text
git commit -m "Update cloud training report"
git push origin main
scripts/lno327_publish_report_pages.sh docs/cloud_fwhm10_n512/model_comparison_seed20260527
```

主仓库推送成功；最新提交以 `git log -1 --oneline` 为准。公开 Pages 发布失败，错误仍为 `Permission denied (publickey)`。

## 已知对照

同 seed 的 FP32 gated4d run 已完成：

```text
runs/lno327_inverse_cnn_weakpeak_n512/gated4d_wide_extra_augtrain_scratch_e280_aux015_seed20260705_ckpt/
```

结果：

```text
best_epoch = 278
test mean range-normalized MAE = 4.5813%
```

这比历史 pure-wide n1024 单模型目标 `2.8056%` 和 ensemble 最好 `2.3839%` 差。当前 AMP run 的意义是确认同数据、同 seed、同 batch 下 AMP 是否保持精度并改善吞吐，而不是只追求显存占满。

## 下一步

1. 不建议继续增大 batch 或 hidden：batch=128 已接近 32GB 显存上限。
2. 优先检查历史 old seed2/seed3 checkpoint 是否可恢复；当前两模型 ensemble 仍停在 4.1060%。
3. 如果旧 checkpoint 不可用，再新增 pure-wide n1024 scratch seed，而不是转向 FWHM10 n512。
4. 若新增 seed 仍停在 4% 级别，应核对当前 n1024 数据 split/生成版本与历史 `2.3839%` artifacts 是否完全一致。

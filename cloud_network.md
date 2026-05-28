# 云端网络与公开报告发布说明

本文记录 `ins-cnn` 云端训练机的网络、远端路径和报告发布约定。不要把代理节点、
订阅链接、token、私钥写进仓库；这些只应存在于本机临时文件、远端系统配置或
GitHub Deploy key 页面。

## 当前远端约定

- SSH alias: `ins-cnn-cloud`
- 实际项目目录: `~/work/ins-cnn`
- 兼容入口: `~/ins-cnn` 可作为指向 `~/work/ins-cnn` 的符号链接
- Python 训练环境: `~/anaconda3/envs/torch2.4_cuda12.1/bin/python`
- Julia: `~/.local/bin/julia`
- GPU: RTX 4080 SUPER

远端自检：

```bash
bash scripts/cloud_netcheck.sh
bash scripts/cloud_enable_tun.sh status
```

## Xray 与 TUN

远端采用 `xray-core` 直接出网，不再依赖本机 `v2rayN` 反向隧道。

- `xray.service`: systemd 管理，期望状态为 `active`
- HTTP proxy: `127.0.0.1:18080`
- SOCKS proxy: `127.0.0.1:18081`
- proxy env: `~/.proxy_env`
- Codex wrapper: `/usr/local/bin/codex`，应读取 `~/.proxy_env`
- TUN interface: `xray0`
- TUN routes:
  - `0.0.0.0/1 dev xray0`
  - `128.0.0.0/1 dev xray0`

启用 TUN 时，`scripts/cloud_enable_tun.sh` 会做两类保护：

- 当前 SSH 客户端和 Xray 上游 IP 走物理网卡，避免连接断开或代理环路；
- `sshd` 源端口 `22` 的回包通过 `fwmark` 策略路由走物理网卡。

如果 TUN 出问题：

```bash
bash scripts/cloud_enable_tun.sh disable
```

重新启用：

```bash
bash scripts/cloud_enable_tun.sh enable
```

## 网络检查

```bash
bash scripts/cloud_netcheck.sh
```

检查内容包括：

- 远端 repo 路径和 git 状态；
- `~/.proxy_env` 与 `127.0.0.1:18080` 是否存在；
- `xray.service`、`xray0` 和两条 TUN 路由；
- 无代理环境下直接访问 OpenAI/GitHub；
- 通过 `proxyrun` 访问 OpenAI/GitHub；
- Codex CLI 版本和登录状态；
- Python/Julia/GPU 状态；
- `github.com-ins-cnn-report` deploy key 是否能认证。

## 配置或刷新 Xray

把直接节点分享链接保存到本机临时文件，不要提交：

```bash
umask 077
printf '%s\n' '你的分享链接' > /private/tmp/ins_cnn_xray_share.txt
```

生成并下发远端配置：

```bash
bash scripts/cloud_configure_xray.sh /private/tmp/ins_cnn_xray_share.txt
```

脚本支持直接节点链接：

- `vless://`
- `vmess://`
- `trojan://`
- `ss://`

如果是订阅地址，需要先解析订阅并选择一个节点。

## 本机代理桥

`scripts/cloud_proxy_bridge.sh` 是旧方案和应急方案：本机运行代理，本机通过 SSH
反向隧道把代理暴露给远端。当前主流程不依赖它。

```bash
bash scripts/cloud_proxy_bridge.sh start
bash scripts/cloud_proxy_bridge.sh status
bash scripts/cloud_proxy_bridge.sh stop
```

## 公开报告仓库

主代码仓库 `AvryChen/ins-cnn` 保持 private。公开网页只发布到：

```text
AvryChen/ins-cnn-report
https://avrychen.github.io/ins-cnn-report/
```

发布脚本：

```bash
scripts/lno327_publish_report_pages.sh \
  docs/cloud_fwhm10_n512/gated4d_seed20260526
```

云端自动发布需要 GitHub deploy key：

- SSH alias: `github.com-ins-cnn-report`
- Remote: `git@github.com-ins-cnn-report:AvryChen/ins-cnn-report.git`
- GitHub 页面: `ins-cnn-report -> Settings -> Deploy keys`
- 必须勾选 `Allow write access`

如果 deploy key 还没加好，训练和报告生成仍可继续；只是最终发布需要本机或手动
推送到 `ins-cnn-report`。

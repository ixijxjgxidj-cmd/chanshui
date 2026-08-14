# 第31轮：INSTANCE 完整公开波形下载（进行中）

## 远程专用目录

`/root/5.6+chanshui1/public_round31`

## 文件清单

- URL：`http://repo.pi.ingv.it/instance/Instance_events_gm.hdf5.bz2`
- HTTP Content-Length：`161809684189` bytes
- ETag：`"25ac9bd6dd-5c10a3e15c0c0"`
- Accept-Ranges：`bytes`
- 下载方式：`curl -L -C - --retry 20 --retry-all-errors`
- 当前：后台断点续传，已在 2026-08-15 07:25 启动；2026-08-15 07:31 核验为 2,991,890,432 bytes（约 1.85%），速率约 8 MB/s，预计还需约 5.5 小时
- 远程磁盘：`/root` 可用约 1.1 TB；主下载进程 PID `1112310` 存活
- 临时文件不会进入 GitHub；仅提交元数据、脚本和校验记录。

## 后续协议

下载完成后先核验文件大小与 ETag，再顺序解压完整 HDF5；读取 metadata 中 `source_magnitude`、`trace_P_arrival_sample`、`station_network_code`，过滤 earthquake、ML/Mw、P±5 秒可用记录，按事件分组固定 train/dev，并构建远程 `t2_cache_instance31`。在公开 INSTANCE 区域留出与 STEAD 台站留出均通过前，不读取比赛数据。

## 合规

本轮只下载公开 INSTANCE；不读取 R1/R2/08 数据，不在本地训练或下载。

## 2026-08-15 07:36 状态
- 远程压缩波形：5,914,763,264 bytes（约 3.66%），速率约 8.2 MB/s，预计剩余约 5 小时 9 分。
- 下载进程仍为 PID 1112310；未启动任何比赛数据读取或训练。
- 旧轮27重复探针曾误启动，已立即终止，不产生新结论。

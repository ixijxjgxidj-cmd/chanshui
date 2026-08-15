# T2 第37轮：现有公开数据池的任务标签谱系审计

## 目的

第36轮后，T2 的任何候选录取都必须使用真实跨区域公开数据。远程机器已有 ETHZ、CREW、GEOFON、Iquique、OBST 等池；本轮核验它们能否用于 T2 震级跨域验证，避免把 T1 拾取数据误当作 T2 定级数据。

## 合规

本轮只列举远程公开池的 HDF5 结构和属性；没有读取比赛包、没有训练、没有本地下载。

## 审计结果

| 数据池 | 样本数 | 波形/属性 | 震级标签 | 可用于 |
|---|---:|---|---|---|
| ETHZ | 14,881 train / 1,427 dev | 3×3001, `p_sample_100hz`, `s_sample_100hz`, `source_event`, `sp_s` | 无 | T1 跨域拾取 |
| CREW | 23,342 train / 1,980 dev | 同上 | 无 | T1 跨域拾取 |
| GEOFON | 1,499 train / 302 val | 3×3001, P/S 属性 | 无 | T1 跨域拾取 |
| Iquique | 501 | 3×3001, P/S 属性 | 无 | T1 盲域评估 |
| OBST | 487 | 3×3001, P/S 属性 | 无 | T1 盲域评估 |

所有 HDF5 的 `data/<id>` 属性只有 `p_sample_100hz`、`s_sample_100hz`、采样率、事件 ID/`sp_s` 等；**没有 `source_magnitude`、震级目录或可复原的震级元数据链接**。因此它们绝不能被用于 T2 的 λ、中心或任何震级模型的跨域标定。

## 判定

- 已有池可支撑 T1 域泛化/蒸馏路线，但不支撑当前 T2 任务。
- T2 的真实跨区域协议继续等待正在下载的 INSTANCE：其官方 metadata 明确含 `source_magnitude`、`source_magnitude_type`、`trace_P_arrival_sample`、网络/台站血缘。
- 不从事件 ID 互联网反查标签：这会造成记录链接失败、版本不一致和不可审计的数据泄漏风险。

## 远程证据

`/root/5.6+chanshui1/pool/*.hdf5`；属性审计由远程 `h5py` 读取完成。

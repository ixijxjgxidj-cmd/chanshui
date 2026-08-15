# INSTANCE HDF5 审计根因与修复（2026-08-15）

审计器原先把 metadata 的 trace_{E,N,Z}_min_counts（原始 ADC 整数）与 HDF5 反演后的物理量直接比较。HDF5 根结构显示 data_format/instrument_response=restituted、unit=mps/mps2，因此尺度不一致导致误报 component_order_ok=false。

修复后改用单位匹配的物理量字段：mps2 使用 trace_{E,N,Z}_pga_cmps2/100，mps 使用 trace_{E,N,Z}_pgv_cmps/100，与每个 HDF5 通道的绝对峰值逐通道比较，允许 0.5% 浮点误差。

120 条样本、6 种排列验证显示 (0,1,2)=ENZ 的中位 log10 误差为 1.5e-6（mps）和 2.6e-8（mps2），相关系数均为 1.0；其他排列明显更差。远程 400 条审计结果：missing_key=0、shape_mismatch=0、nonfinite=0、component_order_checked=12、component_order_matches_ENZ=12、ready_for_cache=true。

详见 experiments/t2_round39b_cross_dataset/instance_orchestrator.py。

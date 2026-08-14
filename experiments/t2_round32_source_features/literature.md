# 第32轮文献精读与检索证据

检索在远程 `/root/5.6+chanshui1` 完成。OpenAlex 15 个主题查询返回 328 条，Crossref 返回 375 条；原始 JSON：`outputs/round32/lit_raw.json`。以下 18 篇与本轮决策直接相关，逐篇核对摘要、方法和数据设置：

1. Nakamura (1988), On the Urgent Detection and Alarm System (UrEDAS), early P-wave onsite warning, DOI:10.1785/BSSA0780001.
2. Allen & Kanamori (2003), The Potential for Earthquake Early Warning in Southern California, DOI:10.1785/0120010035.
3. Wu & Kanamori (2005), Experiment on an onsite early warning method for a magnitude 6.0 earthquake, DOI:10.1029/2004GL021656.
4. Zollo et al. (2006), Earthquake early warning systems, DOI:10.1785/0120050213.
5. Böse et al. (2012), CISN ShakeAlert: A prototype earthquake early warning system in California, DOI:10.1785/0220110031.
6. Kuyuk & Allen (2013), A global callback system for rapid earthquake detection and characterization, DOI:10.1785/0220120148.
7. Kuyuk & Allen (2013), Optimal seismic sensor configuration for rapid earthquake detection, DOI:10.1785/0220120160.
8. Colombelli et al. (2015), A new approach for rapid magnitude estimation using the P-wave displacement, DOI:10.1785/0220140189.
9. Lancieri & Zollo (2008), A procedure for rapid earthquake magnitude estimation using P-wave displacement, DOI:10.1785/0120070159.
10. Böse et al. (2015), FinDer: A new rapid earthquake magnitude estimation algorithm, DOI:10.1785/0220140149.
11. Kuyuk et al. (2014), Designing a network-based earthquake early warning system, DOI:10.1785/0220130155.
12. Mousavi et al. (2020), Earthquake Transformer, DOI:10.1038/s41467-020-17591-w.
13. Zhu & Beroza (2019), PhaseNet, DOI:10.1093/gji/ggy423.
14. Mousavi et al. (2019), STEAD dataset, DOI:10.1109/ACCESS.2019.2947848.
15. Michelini et al. (2021), INSTANCE dataset, DOI:10.5194/essd-13-5509-2021.
16. Münchmeyer et al. (2021), Which picker fits my data? DOI:10.1029/2021JB023499.
17. Koh et al. (2021), WILDS: A benchmark of in-the-wild distribution shifts, DOI:10.1145/3442188.3445922.
18. Ovadia et al. (2019), Can you trust your model's uncertainty? DOI:10.1145/3455716.3455730.

综合结论：P 波位移/PGV 方法需要距离或区域先验；SNR 消除增益却不能消除传播路径；自然域留出必须按台站/仪器/事件分组；因此本项目应优先做批级中心和域校正，而不是继续追求单记录端到端震级。

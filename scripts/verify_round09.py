"""round-09 论文清单原文核验：逐条抓 arXiv abs / DOI landing page 并留证。

用法：python scripts/verify_round09.py --out memory/papers/_raw/round09_verified.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lit_search import Record, verify_record  # noqa: E402

# (标签, 标题, arxiv_id, doi)
ITEMS = [
    ("A1-phasenet", "PhaseNet: A Deep-Neural-Network-Based Seismic Arrival Time Picking Method", "1803.03211", "10.1093/gji/ggy423"),
    ("A2-eqt", "Earthquake transformer-an attentive deep-learning model for simultaneous earthquake detection and phase picking", "", "10.1038/s41467-020-17591-w"),
    ("A3-labelinacc", "Learning Earthquake Wave Arrival Time Picking from Labels with Inaccuracies", "2606.15377", ""),
    ("A4-labelimb", "Improving Deep Learning-Based Seismic Phase Picking by Addressing Label Imbalance", "", "10.21203/rs.3.rs-10439246/v1"),
    ("A5-bayesunc", "A Deep-Learning Phase Picker with Calibrated Bayesian-Derived Uncertainties for Earthquakes in the Yellowstone Volcanic Region", "", "10.1785/0120230068"),
    ("A6-softlabel-icc", "Capturing expert uncertainty: ICC-informed soft labelling for volcano-seismicity", "", "10.1007/s00445-025-01875-4"),
    ("A7-arru", "ARRU Phase Picker: Attention Recurrent-Residual U-Net for Picking Seismic P- and S-Phase Arrivals", "", "10.1785/0220200382"),
    ("A8-whichpicker", "Which picker fits my data? A quantitative evaluation of deep learning based seismic pickers", "", "10.1029/2021JB023499"),
    ("A9-seislm", "SeisLM: a Foundation Model for Seismic Waveforms", "2410.15765", ""),
    ("A10-nabro", "A Little Data Goes a Long Way: Automating Seismic Phase Arrival Picking at Nabro Volcano With Transfer Learning", "", "10.1029/2021jb021910"),
    ("A11-bridgescales", "Using a Deep Neural Network and Transfer Learning to Bridge Scales for Seismic Phase Picking", "", "10.1029/2020gl088651"),
    ("A12-labquakes", "From Labquakes to Megathrusts: Scaling Deep Learning Based Pickers Over 15 Orders of Magnitude", "", "10.1029/2024jh000220"),
    ("A13-customlocal", "Customization of a deep neural network using local data for seismic phase picking", "", "10.3389/feart.2023.1306488"),
    ("A14-pickblue", "PickBlue: Seismic Phase Picking for Ocean Bottom Seismometers With Deep Learning", "", "10.1029/2023ea003332"),
    ("A15-obstransformer", "OBSTransformer: a deep-learning seismic phase picker for OBS data using automated labelling", "2306.04753", "10.1093/gji/ggae049"),
    ("A16-das-semisup", "Seismic arrival-time picking on distributed acoustic sensing data using semi-supervised learning", "2302.08747", "10.1038/s41467-023-43355-3"),
    ("A17-csesnet", "CSESnet: A deep learning P-wave detection model based on UNet++ designed for China Seismic Experimental Site", "", "10.3389/feart.2022.1032839"),
    ("A18-dasformer", "DASFormer: self-supervised pretraining for earthquake monitoring", "", "10.1007/s44267-025-00085-y"),
    ("A19-multievent", "Toward Robust Seismic Phase Picking in Realistic Multi-Event Scenarios", "", "10.5194/egusphere-egu26-6122"),
    ("A20-phaselink", "PhaseLink: A Deep Learning Approach to Seismic Phase Association", "", "10.1029/2018JB016674"),
    ("B1-darkpose", "Distribution-Aware Coordinate Representation for Human Pose Estimation", "1910.06278", ""),
    ("B2-udp", "The Devil is in the Details: Delving into Unbiased Data Processing for Human Pose Estimation", "1911.07524", ""),
    ("B3-rethinkheatmap", "Rethinking the Heatmap Regression for Bottom-up Human Pose Estimation", "2012.15175", ""),
    ("B4-kd-bias-var", "Rethinking Soft Labels for Knowledge Distillation: A Bias-Variance Tradeoff Perspective", "2102.00650", ""),
    ("B5-lsr-kd", "Revisiting Knowledge Distillation via Label Smoothing Regularization", "1909.11723", ""),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for tag, title, aid, doi in ITEMS:
        rec = Record(channel="curated", title=title, arxiv_id=aid, doi=doi)
        v = verify_record(rec)
        rows.append({"tag": tag, "title": title, "arxiv_id": aid, "doi": doi, "verification": v})
        sys.stderr.write("%-18s %-5s ratio=%-6s %s\n" % (
            tag, "OK" if v.get("ok") else "FAIL", v.get("title_token_match_ratio"), v.get("final_url", v.get("reason", ""))[:78]))
    n_ok = sum(1 for r in rows if r["verification"].get("ok"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"n_total": len(rows), "n_ok": n_ok, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    sys.stderr.write("verified_ok %d / %d -> %s\n" % (n_ok, len(rows), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
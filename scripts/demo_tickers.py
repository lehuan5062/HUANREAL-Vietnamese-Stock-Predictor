"""Curated list of liquid Vietnamese tickers for quick demos.

Sourced from VN30, HNX30, and a few liquid UPCOM names as of 2025-2026.
Updates: refresh manually if a ticker leaves the index or stops trading.
"""
VN30 = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "LPB", "MBB", "MSN", "MWG", "PLX", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
]

HNX_LIQUID = [
    "SHS", "PVS", "CEO", "MBS", "IDC", "VCS", "DTD", "TNG", "L14", "NVB",
    "BAB", "VC3", "PVI", "TIG", "DDG", "API", "AMV", "HUT", "PVC", "VFS",
]

UPCOM_LIQUID = [
    "BSR", "ACV", "MCH", "VEA", "VGI", "VTP", "MML", "QNS", "FOX", "VGT",
    "OIL", "MSR", "DVN", "BVB", "ABB",
]

# A handful of HOSE mid-caps not in VN30 but commonly liquid
HOSE_MID = [
    "DGC", "PNJ", "DCM", "DPM", "GMD", "VHC", "HDG", "KDH", "NLG", "DXG",
    "REE", "PVD", "DBC",
]

ALL = sorted(set(VN30 + HNX_LIQUID + UPCOM_LIQUID + HOSE_MID))


if __name__ == "__main__":
    print(" ".join(ALL))
    print(f"\ntotal: {len(ALL)}")

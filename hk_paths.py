"""Shared paths for Hong Kong HLS phenology (cluster or local override)."""
from __future__ import annotations

import os
from pathlib import Path

HK_ROOT = Path(os.environ.get("HK_HLS_ROOT", "/intelnvme03/ziyun218/hls_49QHE_hk"))
PHENO = Path(os.environ.get("HK_PHENO_ROOT", HK_ROOT / "phenology"))
IMPUTATOR = Path(os.environ.get("HK_IMPUTATOR_ROOT", HK_ROOT / "imputator_hk"))

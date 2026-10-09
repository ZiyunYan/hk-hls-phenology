#!/usr/bin/env python3
"""Download Hong Kong Observatory daily temperature, rainfall and wind.

Station coordinates come from the Observatory station table. Files are the
official all-year CSVs.
"""
from __future__ import annotations

import re
import time
import urllib.request
from pathlib import Path

from hk_paths import PHENO

OUT = PHENO / "climate"
PAGE = "https://www.hko.gov.hk/en/cis/stn.htm"
BASE = "https://data.weather.gov.hk/weatherAPI/hko_data/csdi/dataset"


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, flush=True)


def dms(text: str) -> float:
    m = re.search(r"(\d+)°(\d+)'(\d+)", text)
    if not m:
        raise ValueError(text)
    deg, minute, sec = (int(x) for x in m.groups())
    return deg + minute / 60 + sec / 3600


def stations(html: str) -> list[dict]:
    start = html.find('id="automatic"')
    end = html.find("</table>", start)
    rows = re.findall(r"<tr[\s\S]*?</tr>", html[start:end])
    found = []
    for row in rows[2:]:
        cells = re.findall(r"<t[dh][^>]*>([\s\S]*?)</t[dh]>", row)
        cells = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", c)).strip() for c in cells]
        if len(cells) < 16:
            continue
        name = cells[0]
        code_m = re.search(r"\(([A-Z0-9]+)\)", name)
        if not code_m:
            continue
        marks = ["✔" in c or "10004" in c for c in cells[4:16]]
        # column order after elevation: wind, temp, wet, dew, rh, pressure, rain, cloud, sun, solar, vis, heat
        found.append({
            "code": code_m.group(1),
            "name": name,
            "lat": dms(cells[1]),
            "lon": dms(cells[2]),
            "elev_m": cells[3],
            "wind": marks[0],
            "temp": marks[1],
            "rain": marks[6],
        })
    return found


def fetch(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 200:
        return True
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        log(f"fail {url} {exc}")
        if dest.exists():
            dest.unlink()
        return False
    text = dest.read_text(errors="ignore")[:80]
    if "html" in text.lower() or dest.stat().st_size < 200:
        dest.unlink(missing_ok=True)
        return False
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    html = urllib.request.urlopen(PAGE, timeout=60).read().decode("utf-8", "ignore")
    rows = stations(html)
    log(f"stations {len(rows)}")
    lines = ["code,name,lat,lon,elev_m,temp,rain,wind"]
    for row in rows:
        lines.append(
            f"{row['code']},{row['name'].replace(',', ' ')},{row['lat']:.6f},{row['lon']:.6f},"
            f"{row['elev_m']},{int(row['temp'])},{int(row['rain'])},{int(row['wind'])}"
        )
        for kind, flag, suffix in (
            ("temp", row["temp"], "TEMP"),
            ("rain", row["rain"], "RF"),
            ("wind", row["wind"], "WSPD"),
        ):
            if not flag:
                continue
            dest = OUT / f"daily_{row['code']}_{suffix}_ALL.csv"
            url = f"{BASE}/daily_{row['code']}_{suffix}_ALL.csv"
            ok = fetch(url, dest)
            log(f"{'ok' if ok else 'missing'} {dest.name}")
    (OUT / "stations.csv").write_text("\n".join(lines) + "\n")
    log("station table written")


if __name__ == "__main__":
    main()

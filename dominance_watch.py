#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dominance Watch with Regime Bands
"""

import os, io, time, json, math, csv
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================== 設定 ==================
OUT_DIR = os.environ.get("OUT_DIR", "./dominance_out")
CSV_PATH = os.path.join(OUT_DIR, "dominance_daily.csv")
PNG_PATH = os.path.join(OUT_DIR, "dominance_chart.png")

# 帯の色（必要なら好みで変更可）
BAND_COLORS = {
    "イケイケ期（Alt循環）": "#6EE7B7",   # 緑寄り（薄）
    "アルト抜け（リバランス）": "#FDE68A", # 黄寄り（薄）
    "資金抜け（リスクオフ）": "#FCA5A5",  # 赤寄り（薄）
    "中立/混合: 明確な循環シグナルなし": None,  # 帯なし
}
BAND_ALPHA = 0.16

# Stable 集計：DefiLlama の symbol ベース
STABLE_SYMBOLS_WHITELIST = {
    "USDT", "USDC", "DAI", "FDUSD", "TUSD", "BUSD", "PYUSD", "USDP", "GUSD",
}

DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "").strip()
JST = timezone(timedelta(hours=9))

# ================== HTTP ==================
def http_get_json(url: str, params: Optional[dict] = None, headers: Optional[dict] = None, retries: int = 3, timeout: int = 20):
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            last_err = e
        time.sleep(1.0 + 0.5 * i)
    raise last_err

# ================== 取得 ==================
def fetch_coingecko_global():
    data = http_get_json("https://api.coingecko.com/api/v3/global")
    if not data or "data" not in data: raise RuntimeError("Invalid CoinGecko response")
    return data["data"]

def fetch_llama_stables_sum_usd() -> float:
    data = http_get_json("https://stablecoins.llama.fi/stablecoins", params={"includePrices":"true"})
    if not data or "peggedAssets" not in data: raise RuntimeError("Invalid DefiLlama response")

    total = 0.0
    for asset in data["peggedAssets"]:
        sym = (asset.get("symbol") or "").upper()
        if sym not in STABLE_SYMBOLS_WHITELIST:
            continue
        mcap_usd = asset.get("mcap") or asset.get("totalCirculatingUSD")
        if mcap_usd is None:
            cur = asset.get("current") or {}
            circ = cur.get("circulating") or cur.get("minted")
            price = cur.get("price")
            if circ is not None and price is not None:
                try: mcap_usd = float(circ) * float(price)
                except: mcap_usd = None
        if mcap_usd:
            total += float(mcap_usd)
    return float(total)

# ================== 計算 ==================
def compute_metrics() -> Dict[str, float]:
    cg = fetch_coingecko_global()
    total_usd = float(cg["total_market_cap"]["usd"])
    btc_d = float(cg["market_cap_percentage"].get("btc", 0.0))
    eth_d = float(cg["market_cap_percentage"].get("eth", 0.0))
    stable_usd = fetch_llama_stables_sum_usd()
    stable_d = (stable_usd / total_usd * 100.0) if total_usd > 0 else 0.0
    alt_d = max(0.0, 100.0 - (btc_d + eth_d + stable_d))  # 数値誤差ガード

    return {
        "total_usd": total_usd,
        "btc_d": btc_d,
        "eth_d": eth_d,
        "stable_usd": stable_usd,
        "stable_d": stable_d,
        "alt_d": alt_d,
        "btc_mcap": total_usd * (btc_d/100.0),
        "eth_mcap": total_usd * (eth_d/100.0),
    }

# ================== 判定 ==================
def scenario_label(cur: Dict[str, float], prev: Optional[Dict[str, float]]) -> str:
    if prev is None:
        return "初回記録（判定なし）"

    # ノイズ抑制のため ±0.1% しきい値
    def up(k):   return cur[k] > prev[k] * 1.001
    def down(k): return cur[k] < prev[k] * 0.999

    total_up, total_down = up("total_usd"), down("total_usd")
    btc_up, btc_down     = up("btc_d"),   down("btc_d")
    eth_up, eth_down     = up("eth_d"),   down("eth_d")
    st_up,  st_down      = up("stable_d"),down("stable_d")
    alt_up, alt_down     = up("alt_d"),   down("alt_d")

    # 期間ラベル（バンド用のメイン3種）
    if total_up and alt_up and st_down and (btc_down or eth_down):
        return "イケイケ期（Alt循環）"
    if (not total_up or total_down) and alt_down and (btc_up or st_up):
        return "アルト抜け（リバランス）"
    if total_down and st_up and alt_down:
        return "資金抜け（リスクオフ）"

    # 補助
    if total_up and btc_up and eth_up:
        return "通常上昇: BTC・ETH主導の健全上げ"

    return "中立/混合: 明確な循環シグナルなし"

# ================== CSV ==================
def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)

def read_last_row(path: str) -> Optional[Dict[str, float]]:
    if not os.path.exists(path): return None
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            if not rows: return None
            r = rows[-1]
            return {
                "total_usd": float(r["total_usd"]),
                "btc_d": float(r["btc_d"]),
                "eth_d": float(r["eth_d"]),
                "stable_usd": float(r["stable_usd"]),
                "stable_d": float(r["stable_d"]),
                "alt_d": float(r["alt_d"]),
                "btc_mcap": float(r["btc_mcap"]),
                "eth_mcap": float(r["eth_mcap"]),
            }
    except Exception:
        return None

def append_csv(path: str, rec: Dict[str, Any]) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ts_iso","total_usd","btc_mcap","eth_mcap","stable_usd",
            "btc_d","eth_d","stable_d","alt_d","scenario"
        ])
        if not exists:
            w.writeheader()
        w.writerow(rec)

# ================== 描画（バンド付き） ==================
def _classify_rows_for_bands(df_row_prev, df_row_cur) -> str:
    """CSVの2行を比較してシナリオ再判定（後からでも帯が引けるように）。"""
    cur = {
        "total_usd": float(df_row_cur["total_usd"]),
        "btc_d": float(df_row_cur["btc_d"]),
        "eth_d": float(df_row_cur["eth_d"]),
        "stable_d": float(df_row_cur["stable_d"]),
        "alt_d": float(df_row_cur["alt_d"]),
    }
    if df_row_prev is None:
        return "初回記録（判定なし）"
    prev = {
        "total_usd": float(df_row_prev["total_usd"]),
        "btc_d": float(df_row_prev["btc_d"]),
        "eth_d": float(df_row_prev["eth_d"]),
        "stable_d": float(df_row_prev["stable_d"]),
        "alt_d": float(df_row_prev["alt_d"]),
    }
    return scenario_label(cur, prev)

def plot_png(path: str) -> None:
    import pandas as pd
    if not os.path.exists(CSV_PATH): return
    df = pd.read_csv(CSV_PATH)
    if df.empty: return

    df["ts_iso"] = pd.to_datetime(df["ts_iso"])
    df = df.sort_values("ts_iso")

    # 直近平均で正規化した TOTAL（%）を破線で重ねる
    total_norm = df["total_usd"] / df["total_usd"].rolling(20, min_periods=1).mean() * 100.0

    fig = plt.figure(figsize=(13, 6.2))
    ax = plt.gca()

    # ===== 帯（期間）算出 =====
    # 1行ずつ比較して scenario を生成（CSVに既にある場合でも再計算で頑健に）
    scenarios = []
    prev_row = None
    for _, row in df.iterrows():
        sc = _classify_rows_for_bands(prev_row, row) if prev_row is not None else "初回記録（判定なし）"
        scenarios.append(sc)
        prev_row = row

    # 連続区間ごとにまとめて塗る
    cur_label = None
    start_idx = 0
    for i, sc in enumerate(scenarios + ["__END__"]):  # 番兵
        if cur_label is None:
            cur_label = sc
            start_idx = 0
        if i == 0:
            cur_label = sc
            start_idx = 0
        elif sc != cur_label:
            color = BAND_COLORS.get(cur_label)
            if color:
                x0 = df["ts_iso"].iloc[i-1] if start_idx == i else df["ts_iso"].iloc[start_idx]
                x1 = df["ts_iso"].iloc[i-1]
                ax.axvspan(x0, x1, alpha=BAND_ALPHA, color=color)
            cur_label = sc
            start_idx = i

    # ===== 線（4ドミ＋TOTAL） =====
    ax.plot(df["ts_iso"], df["btc_d"],    label="BTC.D (%)")
    ax.plot(df["ts_iso"], df["eth_d"],    label="ETH.D (%)")
    ax.plot(df["ts_iso"], df["stable_d"], label="STABLE.D (%)")
    ax.plot(df["ts_iso"], df["alt_d"],    label="ALT.D (%)")
    ax.plot(df["ts_iso"], total_norm, linestyle="--", label="TOTAL (norm %)")

    # タイトル・凡例
    ax.set_title("Dominance Watch: BTC / ETH / Stable / Alt vs TOTAL (with Regime Bands)")
    ax.set_xlabel("Date (JST)")
    ax.set_ylabel("Percent")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    # シナリオの色の凡例（帯用）
    # 既存凡例に追加ラベルをダミー線で入れる（色の意味を図中に残す）
    from matplotlib.lines import Line2D
    extra = []
    for name, color in BAND_COLORS.items():
        if color:
            extra.append(Line2D([0], [0], color=color, lw=8, alpha=BAND_ALPHA, label=f"Band: {name}"))
    leg = ax.legend(handles=ax.get_legend_handles_labels()[0] + extra, loc="upper left", fontsize=9)
    ax.add_artist(leg)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

# ================== Discord ==================
def discord_send_png(webhook: str, title: str, content: str, png_path: str):
    if not webhook: return
    try:
        with open(png_path, "rb") as f:
            files = {"file": (os.path.basename(png_path), f, "image/png")}
            data = {"content": f"**{title}**\n{content}"}
            r = requests.post(webhook, data=data, files=files, timeout=20)
            if r.status_code >= 300:
                print(f"[Discord] HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"[Discord] Error: {e}")

# ================== メイン ==================
def main():
    ensure_outdir()
    now_jst = datetime.now(JST)

    # 現在と直近
    cur = compute_metrics()
    prev = read_last_row(CSV_PATH)

    # ラベル
    label = scenario_label(cur, prev)

    rec = {
        "ts_iso": now_jst.isoformat(timespec="seconds"),
        "total_usd": f"{cur['total_usd']:.2f}",
        "btc_mcap": f"{cur['btc_mcap']:.2f}",
        "eth_mcap": f"{cur['eth_mcap']:.2f}",
        "stable_usd": f"{cur['stable_usd']:.2f}",
        "btc_d": f"{cur['btc_d']:.4f}",
        "eth_d": f"{cur['eth_d']:.4f}",
        "stable_d": f"{cur['stable_d']:.4f}",
        "alt_d": f"{cur['alt_d']:.4f}",
        "scenario": label,
    }
    append_csv(CSV_PATH, rec)
    plot_png(PNG_PATH)

    print(json.dumps(rec, ensure_ascii=False, indent=2))
    print(f"CSV: {CSV_PATH}")
    print(f"PNG: {PNG_PATH}")
    print(f"Scenario: {label}")

    if DISCORD_WEBHOOK:
        content = (
            f"{now_jst:%Y-%m-%d %H:%M JST}\n"
            f"TOTAL: ${float(rec['total_usd']):,.0f}\n"
            f"BTC.D: {float(rec['btc_d']):.2f}% | ETH.D: {float(rec['eth_d']):.2f}% | "
            f"STABLE.D: {float(rec['stable_d']):.2f}% | ALT.D: {float(rec['alt_d']):.2f}%\n"
            f"判定: {label}"
        )
        discord_send_png(DISCORD_WEBHOOK, "Dominance Watch", content, PNG_PATH)

if __name__ == "__main__":
    main()

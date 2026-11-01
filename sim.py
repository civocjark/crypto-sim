"""Fiktívny krypto test 24/7: 100 $ na účet, veľa obchodov, beží zadarmo na GitHub Actions.

Účty (každý začína so 100 $):
  A  momentum 1×        – všetko do najsilnejšej kryptomeny poslednej hodiny
  B  momentum 5×        – to isté s 5× pákou (veľmi agresívne)
  D  momentum 10×       – to isté s 10× pákou (lotéria – cesta k 1 000 $ alebo k nule)
  C  držať BTC          – porovnanie

Pred každým vstupom Simulátor 10 000× prehrá najbližšie 2 hodiny (z pohybov za posledných 24 h)
a obchod pustí, len ak po poplatkoch vychádza kladný očakávaný výsledok.
Žiadne skutočné peniaze, žiadna burza – ceny z verejného API Coinbase (záloha Kraken).
"""
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC", "BCH"]
KRAKEN = {"BTC": "XBTUSD", "DOGE": "XDGUSD"}
STATE = "state.json"
REPORT = "REPORT.md"
RUN_MINUTES = float(os.getenv("RUN_MINUTES", "55"))
TEST_DAYS = 7
START_USD = 100.0

ACCOUNTS = {
    "A": {"name": "A · momentum 1×", "lev": 1, "fee": 0.0010, "slip": 0.0005},
    "B": {"name": "B · momentum 5×", "lev": 5, "fee": 0.0005, "slip": 0.0003},
    "D": {"name": "D · momentum 10× (lotéria)", "lev": 10, "fee": 0.0005, "slip": 0.0003},
    "C": {"name": "C · držať BTC", "lev": 1, "fee": 0.0010, "slip": 0.0005, "hold": True},
}
TP, SL = 0.015, 0.0075            # cieľ +1,5 %, stop −0,75 % (pohyb ceny)
TRAIL_START, TRAIL = 0.0075, 0.005  # po +0,75 % posúvaný stop 0,5 % pod maximom
MAX_HOLD_MIN = 120
MIN_MOM = 0.004                   # min. +0,4 % za hodinu
SIMS = 10_000

S = requests.Session()
S.headers["User-Agent"] = "crypto-sim/1.0 (fiktivny test)"
TG_TOKEN = os.getenv("TG_TOKEN", "")
TG_CHAT = os.getenv("TG_CHAT_ID", "")


def now() -> datetime:
    return datetime.now(timezone.utc)


def sk_time(d: datetime) -> str:
    return (d + timedelta(hours=1)).strftime("%d.%m. %H:%M")


# ------------------------------------------------------------------ dáta

def candles(coin: str) -> np.ndarray | None:
    """5-minútové uzávierky, najstaršia → najnovšia (~300 sviečok = 25 h)."""
    try:
        r = S.get(f"https://api.exchange.coinbase.com/products/{coin}-USD/candles",
                  params={"granularity": 300}, timeout=15)
        r.raise_for_status()
        rows = sorted(r.json(), key=lambda x: x[0])
        c = np.array([float(x[4]) for x in rows])
        if len(c) > 50:
            return c
    except Exception:
        pass
    try:
        pair = KRAKEN.get(coin, f"{coin}USD")
        r = S.get("https://api.kraken.com/0/public/OHLC", params={"pair": pair, "interval": 5}, timeout=15)
        res = r.json()["result"]
        key = next(k for k in res if k != "last")
        c = np.array([float(x[4]) for x in res[key]])
        return c if len(c) > 50 else None
    except Exception:
        return None


def price(coin: str) -> float | None:
    try:
        r = S.get(f"https://api.exchange.coinbase.com/products/{coin}-USD/ticker", timeout=10)
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception:
        c = candles(coin)
        return float(c[-1]) if c is not None else None


def ema(x: np.ndarray, n: int) -> float:
    a = 2 / (n + 1)
    e = x[0]
    for v in x[1:]:
        e = a * v + (1 - a) * e
    return float(e)


# ------------------------------------------------------------------ simulátor

def simulate(closes: np.ndarray, steps: int = 24, n: int = SIMS, seed: int | None = None) -> dict:
    """10 000 ciest na 2 hodiny (24 × 5 min) z pohybov za posledných 24 h: čo príde skôr – cieľ, alebo stop?"""
    r = np.diff(np.log(closes[-289:]))
    rng = np.random.default_rng(seed)
    paths = np.cumsum(rng.choice(r, size=(n, steps)), axis=1)
    up, dn = math.log(1 + TP), math.log(1 - SL)
    big = steps + 1
    t_up = np.where((paths >= up).any(1), (paths >= up).argmax(1), big)
    t_dn = np.where((paths <= dn).any(1), (paths <= dn).argmax(1), big)
    p_tp = float((t_up < t_dn).mean())
    p_sl = float((t_dn < t_up).mean())
    none = (t_up == big) & (t_dn == big)
    rest = float(np.exp(paths[none, -1]).mean() - 1) if none.any() else 0.0
    ev = p_tp * TP - p_sl * SL + float(none.mean()) * rest
    return {"p_tp": round(p_tp, 3), "p_sl": round(p_sl, 3), "ev": ev}


# ------------------------------------------------------------------ účty

def new_state() -> dict:
    t = now().isoformat(timespec="seconds")
    return {"started": t, "finished": False, "runs": 0,
            "accounts": {k: {"cash": START_USD, "pos": None, "trades": 0, "wins": 0, "fees": 0.0,
                             "peak": START_USD, "max_dd": 0.0, "dead": False, "history": []}
                         for k in ACCOUNTS},
            "trades": []}


def equity(acc: dict, prices: dict) -> float:
    p = acc["pos"]
    if not p:
        return acc["cash"]
    px = prices.get(p["coin"]) or p["entry"]
    return max(0.0, p["margin"] + p["units"] * (px - p["entry"]))


def open_pos(k: str, acc: dict, coin: str, px: float, why: str, events: list) -> None:
    cfg = ACCOUNTS[k]
    fill = px * (1 + cfg["slip"])
    margin = acc["cash"]
    notional = margin * cfg["lev"]
    fee = notional * cfg["fee"]
    margin -= fee
    acc["fees"] += fee
    acc["cash"] = 0.0
    acc["pos"] = {"coin": coin, "entry": fill, "units": notional / fill, "margin": margin,
                  "time": now().isoformat(timespec="seconds"), "peak": fill}
    events.append(f"🟢 {k} KÚPA {coin} @ {fill:,.4g} ({cfg['lev']}×, {notional:,.0f} $) · {why}")


def close_pos(k: str, acc: dict, px: float, why: str, events: list, trades: list) -> None:
    cfg = ACCOUNTS[k]
    p = acc["pos"]
    fill = px * (1 - cfg["slip"])
    fee = p["units"] * fill * cfg["fee"]
    value = max(0.0, p["margin"] + p["units"] * (fill - p["entry"]) - fee)
    start_margin = p["margin"] + p["units"] * p["entry"] * cfg["fee"]
    pnl = value - start_margin
    acc["cash"] = value
    acc["fees"] += fee
    acc["pos"] = None
    acc["trades"] += 1
    acc["wins"] += pnl > 0
    if value < 1.0:
        acc["dead"] = True
    move = fill / p["entry"] - 1
    trades.append({"t": now().isoformat(timespec="seconds"), "acc": k, "coin": p["coin"],
                   "entry": p["entry"], "exit": fill, "move": round(move, 5), "pnl": round(pnl, 4),
                   "equity": round(value, 4), "why": why})
    trades[:] = trades[-400:]
    events.append(f"{'✅' if pnl > 0 else '🔴'} {k} PREDAJ {p['coin']} {move:+.2%} → {pnl:+.2f} $ · účet {value:,.2f} $ · {why}"
                  + (" · 💀 KONIEC ÚČTU" if acc["dead"] else ""))


def manage(k: str, acc: dict, prices: dict, events: list, trades: list) -> None:
    p = acc["pos"]
    if not p or ACCOUNTS[k].get("hold"):
        return
    px = prices.get(p["coin"])
    if not px:
        return
    lev = ACCOUNTS[k]["lev"]
    p["peak"] = max(p["peak"], px)
    move = px / p["entry"] - 1
    peak_move = p["peak"] / p["entry"] - 1
    held = (now() - datetime.fromisoformat(p["time"])).total_seconds() / 60
    if move <= -0.95 / lev:
        close_pos(k, acc, px, "💥 likvidácia", events, trades)
    elif move >= TP:
        close_pos(k, acc, px, "cieľ", events, trades)
    elif move <= -SL:
        close_pos(k, acc, px, "stop-loss", events, trades)
    elif peak_move >= TRAIL_START and px <= p["peak"] * (1 - TRAIL):
        close_pos(k, acc, px, "posúvaný stop", events, trades)
    elif held >= MAX_HOLD_MIN:
        close_pos(k, acc, px, "čas vypršal (2 h)", events, trades)


# ------------------------------------------------------------------ výstupy

def telegram(text: str) -> None:
    if not (TG_TOKEN and TG_CHAT):
        print("[telegram vypnutý]", text)
        return
    try:
        S.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
               data={"chat_id": TG_CHAT, "text": text[:4000]}, timeout=15)
    except Exception as e:
        print("telegram chyba:", e)


def summary(state: dict, prices: dict, title: str) -> str:
    lines = [title]
    for k, cfg in ACCOUNTS.items():
        a = state["accounts"][k]
        eq = equity(a, prices)
        wr = f"{a['wins'] / a['trades']:.0%}" if a["trades"] else "–"
        pos = f" · drží {a['pos']['coin']}" if a["pos"] and not cfg.get("hold") else ""
        lines.append(f"{cfg['name']}: {eq:,.2f} $ ({eq / START_USD - 1:+.1%}) · obchody {a['trades']}, "
                     f"úspešnosť {wr}, poplatky {a['fees']:.2f} $, max prepad {a['max_dd']:.0%}{pos}"
                     + (" 💀" if a["dead"] else ""))
    return "\n".join(lines)


def write_report(state: dict, prices: dict) -> None:
    started = datetime.fromisoformat(state["started"])
    rows = []
    for k, cfg in ACCOUNTS.items():
        a = state["accounts"][k]
        eq = equity(a, prices)
        wr = f"{a['wins'] / a['trades']:.0%}" if a["trades"] else "–"
        rows.append(f"| {cfg['name']} | {eq:,.2f} $ | {eq / START_USD - 1:+.1%} | {a['trades']} | {wr} | "
                    f"{a['fees']:.2f} $ | {a['max_dd']:.0%} |")
    last = "\n".join(f"| {sk_time(datetime.fromisoformat(t['t']))} | {t['acc']} | {t['coin']} | {t['move']:+.2%} | "
                     f"{t['pnl']:+.2f} $ | {t['equity']:,.2f} $ | {t['why']} |" for t in reversed(state["trades"][-30:]))
    txt = f"""# Fiktívny krypto test: 100 $ → ?

Štart: {sk_time(started)} · koniec: {sk_time(started + timedelta(days=TEST_DAYS))} (náš čas) · behy: {state['runs']}
Aktualizované: {sk_time(now())} · **žiadne skutočné peniaze**

| Účet | Hodnota | Zmena | Obchody | Úspešnosť | Poplatky | Max prepad |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

Pravidlá: vstup do kryptomeny s najsilnejším pohybom za 1 h (nad 20-sviečkovým priemerom, BTC nepadá),
len ak Simulátor (10 000 ciest na 2 h) dá kladný očakávaný výsledok po poplatkoch.
Výstup: cieľ +{TP:.1%}, stop −{SL:.2%}, posúvaný stop, max {MAX_HOLD_MIN} min.

## Posledné obchody
| Čas | Účet | Coin | Pohyb | Zisk | Účet po | Dôvod |
|---|---|---|---|---|---|---|
{last}
"""
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(txt)


# ------------------------------------------------------------------ hlavná slučka

def main() -> None:
    state = json.load(open(STATE, encoding="utf-8")) if os.path.exists(STATE) else None
    if state and state.get("final_sent"):
        print("Test už skončil.")
        return
    first = state is None
    if first:
        state = new_state()
    state["runs"] += 1
    started = datetime.fromisoformat(state["started"])
    events: list[str] = []
    prices: dict = {}

    if first:
        telegram("🚀 Fiktívny krypto test spustený: 4 účty po 100 $ (1×, 5×, 10×, držať BTC), 7 dní 24/7.\n"
                 "Žiadne skutočné peniaze. Priebežný stav: súbor REPORT.md v repe crypto-sim.")

    t0 = time.time()
    last_scan = 0.0
    while time.time() - t0 < RUN_MINUTES * 60:
        loop_t = time.time()
        if now() >= started + timedelta(days=TEST_DAYS):
            state["finished"] = True
        held = {a["pos"]["coin"] for a in state["accounts"].values() if a["pos"]} | {"BTC"}
        for c in held:
            px = price(c)
            if px:
                prices[c] = px
        # C: držať BTC
        c_acc = state["accounts"]["C"]
        if not c_acc["pos"] and prices.get("BTC"):
            open_pos("C", c_acc, "BTC", prices["BTC"], "kúpiť a držať", events)
        # výstupy
        for k, a in state["accounts"].items():
            manage(k, a, prices, events, state["trades"])
            if state["finished"] and a["pos"] and not ACCOUNTS[k].get("hold"):
                close_pos(k, a, prices.get(a["pos"]["coin"], a["pos"]["entry"]), "koniec testu", events, state["trades"])
        # vstupy – sken každých 5 minút
        flat = [k for k, a in state["accounts"].items()
                if not a["pos"] and not a["dead"] and not ACCOUNTS[k].get("hold")]
        if flat and not state["finished"] and time.time() - last_scan >= 290:
            last_scan = time.time()
            data = {c: candles(c) for c in COINS}
            btc = data.get("BTC")
            btc_ok = btc is not None and btc[-1] / btc[-13] - 1 > -0.01
            cands = []
            for c, cl in data.items():
                if cl is None or len(cl) < 120:
                    continue
                prices[c] = float(cl[-1])
                mom = cl[-1] / cl[-13] - 1
                if mom > MIN_MOM and cl[-1] > ema(cl[-60:], 20):
                    cands.append((mom, c, cl))
            cands.sort(reverse=True)
            if btc_ok:
                for mom, c, cl in cands[:3]:
                    sim = simulate(cl)
                    done = False
                    for k in flat:
                        cfg = ACCOUNTS[k]
                        cost = 2 * (cfg["fee"] + cfg["slip"])
                        if sim["ev"] - cost > 0 and sim["p_tp"] > sim["p_sl"]:
                            px = price(c) or float(cl[-1])
                            open_pos(k, state["accounts"][k], c, px,
                                     f"+{mom:.1%}/h · sim: cieľ {sim['p_tp']:.0%} vs stop {sim['p_sl']:.0%}", events)
                            done = True
                    if done:
                        break
        # prepad účtov
        for k, a in state["accounts"].items():
            eq = equity(a, prices)
            a["peak"] = max(a["peak"], eq)
            a["max_dd"] = min(a["max_dd"], eq / a["peak"] - 1)
        if state["finished"]:
            break
        time.sleep(max(5.0, 60 - (time.time() - loop_t)))

    # história a výstupy
    hour = now().strftime("%Y-%m-%dT%H")
    for k, a in state["accounts"].items():
        a["history"].append([hour, round(equity(a, prices), 2)])
        a["history"] = a["history"][-24 * 8:]
    write_report(state, prices)
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)

    if events:
        telegram(("\n".join(events[-25:]))[:3500])
    sk_hour = (now() + timedelta(hours=1)).hour
    if state["finished"]:
        telegram(summary(state, prices, "🏁 KONIEC TESTU po 7 dňoch:"))
        state["final_sent"] = True
        with open(STATE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=1)
    elif sk_hour in (8, 20) or first:
        telegram(summary(state, prices, f"📊 Stav {sk_time(now())}:"))


if __name__ == "__main__":
    main()

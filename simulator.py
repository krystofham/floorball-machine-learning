"""
Florbal simulátor s SQLite cache
Použití:
  python simulator.py           # načte z cache, stáhne jen co chybí
  python simulator.py --refresh # vymaže cache a stáhne vše znovu
  python simulator.py --refresh-players  # přestáhne jen hráče/formu
  python simulator.py --refresh-matches  # přestáhne jen zápasy

DB soubor: florbal.db (ve stejném adresáři)
"""

import argparse
import json
import re
import sqlite3
import time
import unicodedata
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import PoissonRegressor

warnings.filterwarnings("ignore")

# ── Konfigurace ────────────────────────────────────────────
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
N_SIM         = 100_000
DECAY         = 0.05
FORM_GAMES    = 5
TEAM_FORM_N   = 6
COMPETITION_FILTER = "liga dorostenců"
MAX_WORKERS   = 8
DB_FILE       = "florbal.db"
CACHE_TTL_H   = 12   # hodiny – po kolika hodinách je cache "stará" (jen informativní)

TEAMS = {
    "1. SC NATIOS Vítkovice B":    "42746",
    "Asper Šumperk":               "42337",
    "FBC Mohelnice":               "42659",
    "FbC KOVO KM Frýdek-Místek":  "41474",
    "FBC ZŠ Uničov":              "40950",
    "Torpedo Havířov B":           "40666",
    "FBC Vikings Kopřivnice":      "42424",
    "FBC Hranice":                 "42029",
    "FBC Přerov":                  "41426",
    "1.MVIL Ostrava":              "42144",
    "FBC ORCA KRNOV":              "41613",
}

MATCHES_TO_SIMULATE = [
    ("1.MVIL Ostrava",  "1. SC NATIOS Vítkovice B"),
    ("FBC Přerov",      "1. SC NATIOS Vítkovice B"),
]

# ── Argumenty ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--refresh",         action="store_true", help="Smaž cache a stáhni vše")
parser.add_argument("--refresh-matches", action="store_true", help="Přestáhni jen zápasy")
parser.add_argument("--refresh-players", action="store_true", help="Přestáhni jen hráče a formu")
args = parser.parse_args()

# ── SQLite cache ───────────────────────────────────────────

def db_connect():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS matches (
            home        TEXT,
            away        TEXT,
            round       INTEGER,
            home_goals  INTEGER,
            away_goals  INTEGER,
            fetched_at  REAL,
            PRIMARY KEY (home, away, round)
        );
        CREATE TABLE IF NOT EXISTS players (
            team_id     TEXT,
            player_id   TEXT,
            name        TEXT,
            games       INTEGER,
            goals       INTEGER,
            assists     INTEGER,
            form_goals  REAL,
            form_assists REAL,
            fetched_at  REAL,
            PRIMARY KEY (team_id, player_id)
        );
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    conn.commit()
    return conn

def db_clear_matches(conn):
    conn.execute("DELETE FROM matches")
    conn.commit()

def db_clear_players(conn):
    conn.execute("DELETE FROM players")
    conn.commit()

def db_save_matches(conn, matches):
    now = time.time()
    conn.executemany(
        "INSERT OR REPLACE INTO matches VALUES (?,?,?,?,?,?)",
        [(m["home"], m["away"], m["round"], m["home_goals"], m["away_goals"], now)
         for m in matches]
    )
    conn.commit()

def db_load_matches(conn):
    rows = conn.execute("SELECT * FROM matches").fetchall()
    return [dict(r) for r in rows]

def db_save_players(conn, team_id, players):
    now = time.time()
    conn.executemany(
        "INSERT OR REPLACE INTO players VALUES (?,?,?,?,?,?,?,?,?)",
        [(team_id, p["player_id"], p["name"], p["games"], p["goals"],
          p["assists"], p.get("form_goals", 0.0), p.get("form_assists", 0.0), now)
         for p in players]
    )
    conn.commit()

def db_load_players(conn, team_id):
    rows = conn.execute(
        "SELECT * FROM players WHERE team_id=?", (team_id,)
    ).fetchall()
    return [dict(r) for r in rows]

def db_cache_age_h(conn, table):
    """Vrátí stáří cache v hodinách (nejstarší záznam)."""
    row = conn.execute(f"SELECT MIN(fetched_at) FROM {table}").fetchone()
    if not row or not row[0]:
        return None
    return (time.time() - row[0]) / 3600

# ── Scraping ───────────────────────────────────────────────

def get_matches(team_id):
    url = f"https://www.ceskyflorbal.cz/team/detail/matches/{team_id}"
    r = requests.get(url, headers=HEADERS, timeout=15)
    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    for match_div in soup.select(".Match"):
        score_tag = match_div.select_one(".Match-score")
        if not score_tag:
            continue
        score_raw   = score_tag.get_text(strip=True)
        score_clean = re.sub(r"[a-z]+$", "", score_raw)
        if ":" not in score_clean:
            continue
        left  = match_div.select_one(".Match-leftContent  .Match-teamName")
        right = match_div.select_one(".Match-rightContent .Match-teamName")
        if not left or not right:
            continue
        round_tag = match_div.select_one(".Match-round")
        round_num = 0
        if round_tag:
            m = re.search(r"(\d+)", round_tag.text)
            if m:
                round_num = int(m.group(1))
        try:
            h, a = score_clean.split(":")
            results.append({
                "home": left.get_text(strip=True),
                "away": right.get_text(strip=True),
                "home_goals": int(h),
                "away_goals": int(a),
                "round": round_num,
            })
        except ValueError:
            continue
    return results


def normalize(s):
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().strip()


def get_team_players(team_id):
    # 1. Soupiska → player_id + jméno
    roster_url = f"https://www.ceskyflorbal.cz/team/detail/roster/{team_id}"
    r = requests.get(roster_url, headers=HEADERS, timeout=15)
    soup = BeautifulSoup(r.text, "html.parser")
    roster = {}
    for link in soup.select("a[href*='/person/detail/player/']"):
        pid = re.search(r"/person/detail/player/(\d+)", link["href"])
        h3 = link.select_one("h3")
        if pid and h3:
            name = h3.get_text(strip=True)
            roster[normalize(name)] = {"name": name, "player_id": pid.group(1)}

    # 2. Statistiky → góly, asistence, zápasy
    stats_url = f"https://www.ceskyflorbal.cz/team/detail/statistics/{team_id}"
    r = requests.get(stats_url, headers=HEADERS, timeout=15)
    soup = BeautifulSoup(r.text, "html.parser")
    stats = {}
    for t in soup.select("table"):
        hdrs = [th.get_text(strip=True) for th in t.select("th")]
        if "BVP" not in hdrs:
            continue
        try:
            z_idx   = hdrs.index("Z")
            bvp_idx = hdrs.index("BVP")
            a_idx   = hdrs.index("A")
        except ValueError:
            continue
        for row in t.select("tr"):
            cells = [td.get_text(strip=True) for td in row.select("td")]
            link  = row.select_one("a[href*='/person/detail/player/']")
            if not link or len(cells) < max(z_idx, bvp_idx, a_idx) + 1:
                continue
            raw_name = link.get_text(strip=True)
            raw_name_fixed = re.sub(
                r"([a-záčďéěíňóřšťúůýž])([A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ])", r"\1 \2", raw_name
            )
            try:
                stats[normalize(raw_name_fixed)] = {
                    "goals":   int(cells[bvp_idx]),
                    "assists": int(cells[a_idx]),
                    "games":   int(cells[z_idx]),
                }
            except ValueError:
                continue
        break

    # 3. Spoj – přímý match, pak fallback na příjmení
    def last_name(s):
        """Poslední slovo normalizovaného jména = příjmení."""
        parts = s.split()
        return parts[-1] if parts else s

    # Předpočítej příjmení ze stats pro fallback
    stats_by_lastname = {}
    for k, v in stats.items():
        ln = last_name(k)
        stats_by_lastname.setdefault(ln, []).append((k, v))

    players = []
    for nkey, rdata in roster.items():
        s = stats.get(nkey)
        if s is None:
            # Fallback: hledej podle příjmení
            ln = last_name(nkey)
            candidates = stats_by_lastname.get(ln, [])
            if len(candidates) == 1:
                s = candidates[0][1]
            else:
                s = {"goals": 0, "assists": 0, "games": 1}
        matched = s.get("goals", 0) > 0 or s.get("games", 1) > 1
        players.append({
            "name":      rdata["name"],
            "player_id": rdata["player_id"],
            "games":     max(s["games"], 1),
            "goals":     s["goals"],
            "assists":   s["assists"],
            "_matched":  matched,
        })

    # Debug: ukaž hráče bez stats matchingu
    no_match = [p for p in players if not p["_matched"]]
    if no_match:
        print(f"    [!] Hráči bez stats ({len(no_match)}): "
              + ", ".join(p["name"] for p in no_match))
        print(f"    Dostupné klíče ve stats: {sorted(stats.keys())}")

    return players


def get_player_form(player_id, competition_filter=COMPETITION_FILTER, n=FORM_GAMES):
    """
    Stáhne profil hráče a vrátí:
    - form_goals / form_assists: průměr posledních n zápasů ve filtrované soutěži
    - season_goals / season_games: sezónní součet ze všech soutěží (tabulka #0 na profilu)
      Toto slouží jako fallback pro hráče, kteří nejsou ve /statistics/ svého týmu
      (např. hosté evidovaní pod mateřským oddílem).
    """
    url = f"https://www.ceskyflorbal.cz/person/detail/player/{player_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
    except Exception:
        return {"form_goals": 0.0, "form_assists": 0.0, "season_goals": 0, "season_games": 1}
    soup = BeautifulSoup(r.text, "html.parser")
    tables = soup.select("table")

    # ── A) Sezónní statistiky z tabulky #0 (Soutěž / Z / B / A) ──
    # Bereme řádky aktuální sezóny filtrované přes competition_filter
    season_goals = 0
    season_games = 0
    for t in tables:
        hdrs = [th.get_text(strip=True) for th in t.select("th")]
        if "Soutěž" not in hdrs or "B" not in hdrs or "Z" not in hdrs:
            continue
        try:
            z_idx   = hdrs.index("Z")
            b_idx   = hdrs.index("B")
            comp_idx = hdrs.index("Soutěž")
        except ValueError:
            continue
        for row in t.select("tr"):
            cells = [td.get_text(strip=True) for td in row.select("td")]
            if len(cells) < max(z_idx, b_idx, comp_idx) + 1:
                continue
            comp = cells[comp_idx]
            if competition_filter.lower() not in comp.lower():
                continue
            try:
                season_goals += int(cells[b_idx])
                season_games += int(cells[z_idx])
            except ValueError:
                continue
        if season_games > 0:
            break  # první tabulka se soutěžním přehledem stačí

    # ── B) Forma: posledních n zápasů (tabulka #1, Datum/B/A) ──
    game_rows = []
    for t in tables:
        hdrs = [th.get_text(strip=True) for th in t.select("th")]
        if "Datum" not in hdrs or "B" not in hdrs or "A" not in hdrs:
            continue
        for row in t.select("tr"):
            tds = [td for td in row.select("td")
                   if "ProfilePerson--Table--mobileMatch" not in td.get("class", [])]
            if len(tds) < 6:
                continue
            comp_cell = tds[1].get_text(strip=True)
            b_cell    = tds[4].get_text(strip=True)
            a_cell    = tds[5].get_text(strip=True)
            if competition_filter.lower() not in comp_cell.lower():
                continue
            try:
                game_rows.append({"goals": int(b_cell), "assists": int(a_cell)})
            except ValueError:
                continue
        if game_rows:
            break

    recent = game_rows[:n]
    return {
        "form_goals":   float(np.mean([g["goals"]   for g in recent])) if recent else 0.0,
        "form_assists": float(np.mean([g["assists"] for g in recent])) if recent else 0.0,
        "season_goals": season_goals,
        "season_games": max(season_games, 1),
    }


# ── Forma týmu a H2H ───────────────────────────────────────

def team_form(team_name, all_matches_list, n=TEAM_FORM_N):
    team_matches = []
    for m in all_matches_list:
        if m["home"] == team_name:
            team_matches.append({"gf": m["home_goals"], "ga": m["away_goals"], "round": m["round"]})
        elif m["away"] == team_name:
            team_matches.append({"gf": m["away_goals"], "ga": m["home_goals"], "round": m["round"]})
    team_matches.sort(key=lambda x: x["round"], reverse=True)
    recent = team_matches[:n]
    if not recent:
        return {"form_gf": 0.0, "form_ga": 0.0, "form_pts": 0.0}
    gf  = float(np.mean([x["gf"] for x in recent]))
    ga  = float(np.mean([x["ga"] for x in recent]))
    pts = float(np.mean([3 if x["gf"] > x["ga"] else (1 if x["gf"] == x["ga"] else 0) for x in recent]))
    return {"form_gf": gf, "form_ga": ga, "form_pts": pts}


def h2h_stats(home, away, all_matches_list):
    games = [m for m in all_matches_list
             if (m["home"] == home and m["away"] == away)
             or (m["home"] == away and m["away"] == home)]
    if not games:
        return None, None
    hg, ag = [], []
    for m in games:
        if m["home"] == home:
            hg.append(m["home_goals"]); ag.append(m["away_goals"])
        else:
            hg.append(m["away_goals"]); ag.append(m["home_goals"])
    return float(np.mean(hg)), float(np.mean(ag))


# ── Načtení / stažení dat ──────────────────────────────────

conn = db_connect()

# Zápasy
refresh_m = args.refresh or args.refresh_matches
if refresh_m:
    print("Mažu cache zápasů...")
    db_clear_matches(conn)

cached_matches = db_load_matches(conn)
if cached_matches:
    age = db_cache_age_h(conn, "matches")
    print(f"Zápasy z cache ({len(cached_matches)} záznamů, stáří {age:.1f}h)")
    if age and age > CACHE_TTL_H:
        print(f"  [!] Cache je starší než {CACHE_TTL_H}h – zvažte --refresh-matches")
    matches_list = cached_matches
else:
    print("Stahuji zápasy (paralelně)...")
    all_matches: dict = {}

    def fetch_team_matches(args_):
        name, tid = args_
        return name, get_matches(tid)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_team_matches, (n, t)): n for n, t in TEAMS.items()}
        for fut in as_completed(futures):
            name, matches = fut.result()
            for m in matches:
                all_matches[(m["home"], m["away"], m["round"])] = m
            print(f"  {name}: OK")

    matches_list = list(all_matches.values())
    db_save_matches(conn, matches_list)
    print(f"  Uloženo do DB: {len(matches_list)} zápasů")

max_round = max(m["round"] for m in matches_list)
print(f"  Celkem: {len(matches_list)} zápasů (max kolo: {max_round})")

# Hráči + forma
refresh_p = args.refresh or args.refresh_players
if refresh_p:
    print("\nMažu cache hráčů...")
    db_clear_players(conn)

SIM_TEAMS = set(t for pair in MATCHES_TO_SIMULATE for t in pair)
team_player_data: dict = {}

teams_to_fetch = []
for name, tid in TEAMS.items():
    if name not in SIM_TEAMS:
        team_player_data[name] = []
        continue
    cached = db_load_players(conn, tid)
    if cached:
        team_player_data[name] = cached
    else:
        teams_to_fetch.append((name, tid))

if teams_to_fetch:
    print(f"\nStahuji hráče a formu pro {len(teams_to_fetch)} tým(y) (paralelně)...")

    def fetch_player_form(p):
        form = get_player_form(p["player_id"])
        result = {**p, **form}
        # Pokud profil hráče má více gólů než /statistics/ týmu,
        # použij data z profilu (hosté, hráči z více soutěží, atd.)
        if form["season_goals"] > result["goals"]:
            result["goals"] = form["season_goals"]
            result["games"] = max(result["games"], form["season_games"])
        return result

    def fetch_team_data(args_):
        name, tid = args_
        players = get_team_players(tid)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            enriched = list(ex.map(fetch_player_form, players))
        return name, tid, enriched

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(fetch_team_data, (n, t)): n for n, t in teams_to_fetch}
        for fut in as_completed(futures):
            name, tid, players = fut.result()
            team_player_data[name] = players
            db_save_players(conn, tid, players)
            ok = sum(1 for p in players if p.get("form_goals", 0) > 0)
            print(f"  {name}: {len(players)} hráčů, {ok} s formou > 0  [uloženo do DB]")
else:
    for name in SIM_TEAMS:
        if name in team_player_data:
            p = team_player_data[name]
            ok = sum(1 for x in p if x.get("form_goals", 0) > 0)
            age = db_cache_age_h(conn, "players")
            age_str = f"{age:.1f}h" if age else "?"
            print(f"  {name}: {len(p)} hráčů z cache (stáří {age_str})")

conn.close()

# ── Model ──────────────────────────────────────────────────
team_list = sorted(TEAMS.keys())
team_idx  = {t: i for i, t in enumerate(team_list)}
n_teams   = len(team_list)

rows, weights = [], []
for m in matches_list:
    h, a = m["home"], m["away"]
    if h not in team_idx or a not in team_idx:
        continue
    w  = np.exp(-DECAY * (max_round - m["round"]))
    hf = team_form(h, matches_list)
    af = team_form(a, matches_list)

    def make_x(atk, defn, is_home, atk_form, def_form):
        x = [0.0] * (n_teams * 2 + 5)
        x[team_idx[atk]]            = 1.0
        x[n_teams + team_idx[defn]] = 1.0
        x[-5] = float(is_home)
        x[-4] = atk_form["form_gf"]
        x[-3] = def_form["form_ga"]
        x[-2] = atk_form["form_pts"]
        x[-1] = def_form["form_pts"]
        return x

    rows.append((*make_x(h, a, True,  hf, af), m["home_goals"]))
    weights.append(w)
    rows.append((*make_x(a, h, False, af, hf), m["away_goals"]))
    weights.append(w)

feat_cols = ([f"atk_{t}" for t in team_list]
             + [f"def_{t}" for t in team_list]
             + ["home", "atk_form_gf", "def_form_ga", "atk_pts", "def_pts", "goals"])
df = pd.DataFrame(rows, columns=feat_cols)
X  = df.drop("goals", axis=1).values
y  = df["goals"].values

print(f"\nTrénuji model... (vzorků: {len(y)})")
model = PoissonRegressor(alpha=0.3, max_iter=3000)
model.fit(X, y, sample_weight=np.array(weights))
print("Hotovo.")


# ── Predikce střelců ───────────────────────────────────────

def predict_scorers(team_name, lam_team):
    players = team_player_data.get(team_name, [])
    if not players:
        return []

    # Skóre pro góly
    for p in players:
        season_avg   = p["goals"]   / max(p["games"], 1)
        form_avg     = p.get("form_goals", 0.0)
        p["score_g"] = (0.4 * season_avg + 0.6 * form_avg) if form_avg > 0 else season_avg

    # Skóre pro asistence (stejná logika)
    for p in players:
        season_avg_a = p["assists"] / max(p["games"], 1)
        form_avg_a   = p.get("form_assists", 0.0)
        p["score_a"] = (0.4 * season_avg_a + 0.6 * form_avg_a) if form_avg_a > 0 else season_avg_a

    total_g = sum(p["score_g"] for p in players) or 1.0
    total_a = sum(p["score_a"] for p in players) or 1.0

    result = []
    for p in players:
        expected_goals   = lam_team * (p["score_g"] / total_g)
        expected_assists = lam_team * 1.5 * (p["score_a"] / total_a)

        # Směrodatná odchylka – pro Poissonovo rozdělení σ = sqrt(λ)
        std_goals   = np.sqrt(expected_goals)
        std_assists = np.sqrt(expected_assists)

        prob_goal   = 1 - np.exp(-expected_goals)
        prob_assist = 1 - np.exp(-expected_assists)

        if prob_goal > 0.05:
            result.append({
                "name":             p["name"],
                "prob_goal":        prob_goal,
                "prob_assist":      prob_assist,
                "expected_goals":   expected_goals,
                "std_goals":        std_goals,
                "expected_assists": expected_assists,
                "std_assists":      std_assists,
            })
    return sorted(result, key=lambda x: x["prob_goal"], reverse=True)


# ── Simulace ───────────────────────────────────────────────

def make_x_pred(atk, defn, is_home):
    hf = team_form(atk,  matches_list)
    af = team_form(defn, matches_list)
    x  = [0.0] * (n_teams * 2 + 5)
    x[team_idx[atk]]            = 1.0
    x[n_teams + team_idx[defn]] = 1.0
    x[-5] = float(is_home)
    x[-4] = hf["form_gf"]
    x[-3] = af["form_ga"]
    x[-2] = hf["form_pts"]
    x[-1] = af["form_pts"]
    return x


def simulate(home, away):
    lam_h = model.predict([make_x_pred(home, away, True)])[0]
    lam_a = model.predict([make_x_pred(away, home, False)])[0]

    h2h_h, h2h_a = h2h_stats(home, away, matches_list)
    if h2h_h is not None:
        H2H_W = 0.2
        lam_h = (1 - H2H_W) * lam_h + H2H_W * h2h_h
        lam_a = (1 - H2H_W) * lam_a + H2H_W * h2h_a

    lam_h = max(lam_h, 0.3)
    lam_a = max(lam_a, 0.3)

    hg  = np.random.poisson(lam_h, N_SIM)
    ag  = np.random.poisson(lam_a, N_SIM)
    top = Counter(zip(hg, ag)).most_common(5)

    # Simulační směrodatné odchylky počtu gólů
    std_hg = float(np.std(hg))
    std_ag = float(np.std(ag))

    hf_home = team_form(home, matches_list)
    hf_away = team_form(away, matches_list)
    n_h2h   = len([m for m in matches_list
                   if (m["home"]==home and m["away"]==away)
                   or (m["home"]==away and m["away"]==home)])

    lines = []
    lines.append(f"{home} vs {away}")
    lines.append("")
    lines.append(f"Očekávané góly: {lam_h:.2f} +/- {np.sqrt(lam_h):.2f} : {lam_a:.2f} +/- {np.sqrt(lam_a):.2f}")
    if h2h_h is not None:
        lines.append(f"H2H průměr: {h2h_h:.1f} : {h2h_a:.1f} ({n_h2h} zápasů)")
    lines.append(f"Forma domácích: GF={hf_home['form_gf']:.1f} GA={hf_home['form_ga']:.1f} body/z={hf_home['form_pts']:.1f}")
    lines.append(f"Forma hostů:    GF={hf_away['form_gf']:.1f} GA={hf_away['form_ga']:.1f} body/z={hf_away['form_pts']:.1f}")
    lines.append("")
    lines.append(f"Výhra domácích: {np.mean(hg > ag)*100:.1f}%")
    lines.append(f"Výhra hostů:    {np.mean(ag > hg)*100:.1f}%")
    lines.append(f"Remíza (60 min): {np.mean(hg == ag)*100:.1f}%")
    lines.append(f"Průměr gólů: {np.mean(hg + ag):.1f} (sigma dom={std_hg:.2f}, host={std_ag:.2f})")
    lines.append("")
    lines.append("Nejpravděpodobnější výsledky:")
    for (h, a), cnt in top:
        lines.append(f"  {h}:{a}  {cnt / N_SIM * 100:.1f}%")

    for team, lam in [(home, lam_h), (away, lam_a)]:
        scorers = predict_scorers(team, lam)
        if not scorers:
            continue
        lines.append("")
        lines.append(f"Kandidáti - {team}:")
        for s in scorers:
            lines.append(
                f"  {s['name']}  "
                f"gól={s['prob_goal']*100:.1f}% ({s['expected_goals']:.2f}+/-{s['std_goals']:.2f})  "
                f"as={s['prob_assist']*100:.1f}% ({s['expected_assists']:.2f}+/-{s['std_assists']:.2f})"
            )

    print("\n" + "\n".join(lines))


for home, away in MATCHES_TO_SIMULATE:
    simulate(home, away)
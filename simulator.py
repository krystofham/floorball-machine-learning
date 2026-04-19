"""
Florbal simulátor s SQLite cache + HTML výstup
Hráče tahá ze zápisů zápasů (ne ze soupisky), takže zachytí i hosty a hráče z jiných soutěží.

Použití:
  python simulator.py           # načte z cache, stáhne jen co chybí
  python simulator.py --refresh # vymaže cache a stáhne vše znovu
  python simulator.py --refresh-players  # přestáhne jen hráče/formu
  python simulator.py --refresh-matches  # přestáhne jen zápasy

Výstup: report.html (ve stejném adresáři)
DB soubor: florbal.db (ve stejném adresáři)
"""

import argparse
import json
import re
import sqlite3
import time
import unicodedata
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

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
CACHE_TTL_H   = 12
HTML_OUTPUT   = "index.html"

TEAMS = {
    "1. SC NATIOS Vítkovice B":  "42747",
    "TJ Slovan Havířov B":       "43009",
    "1.FBK Sršni Rožnov p/R":   "40726",
    "1.FK Ostrava JIH":          "41224",
    "FbK Horní Suchá":           "40843",
    "1. FBK Eagles Orlová":      "41444",
    "Warriors Nový Jičín":       "42764",
    "FBC Spartak Bílovec":       "43204",
    "1. SC BOHUMÍN 98":          "42002",
    "FBC TIGERS PORUBA":         "41716",
    "S.K. P.E.M.A. OPAVA":      "43165",
    "Z.F.K. Petrovice":          "41706",
}

MATCHES_TO_SIMULATE = [
    ("1. SC NATIOS Vítkovice B", "FBC TIGERS PORUBA"),
    ("1. SC NATIOS Vítkovice B", "FbK Horní Suchá"),
]

# ── Argumenty ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--refresh",         action="store_true")
parser.add_argument("--refresh-matches", action="store_true")
parser.add_argument("--refresh-players", action="store_true")
args = parser.parse_args()

# ── SQLite cache ───────────────────────────────────────────

def db_connect():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS matches (
            match_id    TEXT,
            home        TEXT,
            away        TEXT,
            round       INTEGER,
            home_goals  INTEGER,
            away_goals  INTEGER,
            fetched_at  REAL,
            PRIMARY KEY (home, away, round)
        );
        CREATE TABLE IF NOT EXISTS players (
            team_name   TEXT,
            player_id   TEXT,
            name        TEXT,
            games       INTEGER,
            goals       INTEGER,
            assists     INTEGER,
            form_goals  REAL,
            form_assists REAL,
            fetched_at  REAL,
            PRIMARY KEY (team_name, player_id)
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
        "INSERT OR REPLACE INTO matches VALUES (?,?,?,?,?,?,?)",
        [(m.get("match_id",""), m["home"], m["away"], m["round"],
          m["home_goals"], m["away_goals"], now)
         for m in matches]
    )
    conn.commit()

def db_load_matches(conn):
    rows = conn.execute("SELECT * FROM matches").fetchall()
    return [dict(r) for r in rows]

def db_save_players(conn, team_name, players):
    now = time.time()
    conn.executemany(
        "INSERT OR REPLACE INTO players VALUES (?,?,?,?,?,?,?,?,?)",
        [(team_name, p["player_id"], p["name"], p["games"], p["goals"],
          p["assists"], p.get("form_goals", 0.0), p.get("form_assists", 0.0), now)
         for p in players]
    )
    conn.commit()

def db_load_players(conn, team_name):
    rows = conn.execute(
        "SELECT * FROM players WHERE team_name=?", (team_name,)
    ).fetchall()
    return [dict(r) for r in rows]

def db_cache_age_h(conn, table):
    row = conn.execute(f"SELECT MIN(fetched_at) FROM {table}").fetchone()
    if not row or not row[0]:
        return None
    return (time.time() - row[0]) / 3600

# ── Scraping zápasů ────────────────────────────────────────

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

        match_id = ""
        link = score_tag.select_one("a[href*='/match/detail/']")
        if link:
            m2 = re.search(r"/match/detail/default/(\d+)", link["href"])
            if m2:
                match_id = m2.group(1)

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
                "match_id":   match_id,
                "home":       left.get_text(strip=True),
                "away":       right.get_text(strip=True),
                "home_goals": int(h),
                "away_goals": int(a),
                "round":      round_num,
            })
        except ValueError:
            continue
    return results

# ── Scraping hráčů ze zápisu ──────────────────────────────

def _fix_name(s):
    s = re.sub(r"([a-záčďéěíňóřšťúůýž])([A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ])", r"\1 \2", s)
    s = re.sub(r"\s*(brankář|útočník|obránce|záložník)\s*$", "", s, flags=re.IGNORECASE)
    return s.strip()


def get_match_lineups(match_id):
    url = f"https://www.ceskyflorbal.cz/match/detail/default/{match_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
    except Exception as e:
        print(f"    [!] Chyba zápis {match_id}: {e}")
        return {}

    soup = BeautifulSoup(r.text, "html.parser")

    def clean_title(el):
        txt = el.get_text(strip=True)
        return re.sub(r"[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]{2,4}$", "", txt).strip()

    home_el  = soup.select_one(".MatchCenter-teamTitle-home")
    guest_el = soup.select_one(".MatchCenter-teamTitle-quest")
    if not home_el or not guest_el:
        return {}

    home_short  = clean_title(home_el)
    guest_short = clean_title(guest_el)

    def extract_pids(selector):
        pids = {}
        for el in soup.select(selector):
            a = el.select_one("a[href*='/person/detail/player/']")
            if not a:
                continue
            m = re.search(r"/person/detail/player/(\d+)", a["href"])
            if m:
                pids[m.group(1)] = _fix_name(a.get_text(strip=True))
        return pids

    home_pids  = extract_pids(".MatchCenter-player-home")
    guest_pids = extract_pids(".MatchCenter-player-quest")

    def parse_stat_table(table):
        out = {}
        for row in table.select("tr"):
            a = row.select_one("a[href*='/person/detail/player/']")
            if not a:
                continue
            m = re.search(r"/person/detail/player/(\d+)", a["href"])
            if not m:
                continue
            cells = [td.get_text(strip=True) for td in row.select("td")]
            try:
                count = int(cells[-1])
            except (ValueError, IndexError):
                count = 0
            pid = m.group(1)
            out[pid] = out.get(pid, 0) + count
        return out

    left_tables  = soup.select(".MatchCenter-statistics--left")
    right_tables = soup.select(".MatchCenter-statistics--right")

    home_goals    = parse_stat_table(left_tables[0])  if len(left_tables)  > 0 else {}
    guest_goals   = parse_stat_table(right_tables[0]) if len(right_tables) > 0 else {}
    home_assists  = parse_stat_table(left_tables[1])  if len(left_tables)  > 1 else {}
    guest_assists = parse_stat_table(right_tables[1]) if len(right_tables) > 1 else {}

    def build(pids, goals_map, assists_map):
        return [{"player_id": pid, "name": name,
                 "goals": goals_map.get(pid, 0), "assists": assists_map.get(pid, 0)}
                for pid, name in pids.items()]

    result = {}
    if home_pids:
        result[home_short]  = build(home_pids,  home_goals,  home_assists)
    if guest_pids:
        result[guest_short] = build(guest_pids, guest_goals, guest_assists)
    return result


# ── Agregace hráčů ze všech zápisů týmu ───────────────────

def normalize(s):
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().strip()


def build_team_players_from_matches(team_name, team_matches, all_match_lineups, debug=False):
    player_stats = defaultdict(lambda: {"name": "", "goals": 0, "assists": 0, "games": 0})
    matched_count = 0
    missed_count  = 0

    for m in team_matches:
        mid = m.get("match_id", "")
        if not mid or mid not in all_match_lineups:
            missed_count += 1
            continue
        lineups = all_match_lineups[mid]
        if not lineups:
            missed_count += 1
            continue

        team_players = None
        norm_tn = normalize(team_name)

        for lineup_team, players in lineups.items():
            norm_lt = normalize(lineup_team)
            if norm_lt == norm_tn or norm_lt in norm_tn or norm_tn in norm_lt:
                team_players = players
                break

        if team_players is None:
            if debug:
                print(f"    [dbg] Zápis {mid}: nenalezen '{team_name}' v {list(lineups.keys())}")
            missed_count += 1
            continue

        matched_count += 1
        for p in team_players:
            pid = p["player_id"]
            player_stats[pid]["name"]     = p["name"]
            player_stats[pid]["goals"]   += p["goals"]
            player_stats[pid]["assists"] += p["assists"]
            player_stats[pid]["games"]   += 1

    if debug or missed_count > 0:
        print(f"    [dbg] {team_name}: matched={matched_count} zápasů, missed={missed_count}")

    result = []
    for pid, s in player_stats.items():
        if s["games"] == 0:
            continue
        result.append({
            "player_id": pid,
            "name":      s["name"],
            "goals":     s["goals"],
            "assists":   s["assists"],
            "games":     max(s["games"], 1),
        })
    return result


# ── Forma hráče ────────────────────────────────────────────

def get_player_form(player_id, competition_filter=COMPETITION_FILTER, n=FORM_GAMES):
    url = f"https://www.ceskyflorbal.cz/person/detail/player/{player_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
    except Exception:
        return {"form_goals": 0.0, "form_assists": 0.0,
                "season_goals": 0, "season_assists": 0, "season_games": 0}
    soup = BeautifulSoup(r.text, "html.parser")
    tables = soup.select("table")

    season_goals = 0
    season_assists = 0
    season_games = 0
    for t in tables:
        hdrs = [th.get_text(strip=True) for th in t.select("th")]
        if "Soutěž" not in hdrs or "B" not in hdrs or "Z" not in hdrs:
            continue
        try:
            z_idx    = hdrs.index("Z")
            b_idx    = hdrs.index("B")
            a_idx    = hdrs.index("A") if "A" in hdrs else None
            comp_idx = hdrs.index("Soutěž")
        except ValueError:
            continue
        for row in t.select("tr"):
            cells = [td.get_text(strip=True) for td in row.select("td")]
            if len(cells) < max(z_idx, b_idx, comp_idx) + 1:
                continue
            try:
                season_goals   += int(cells[b_idx])
                season_games   += int(cells[z_idx])
                if a_idx and a_idx < len(cells):
                    season_assists += int(cells[a_idx])
            except ValueError:
                continue
        if season_games > 0:
            break

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
        "form_goals":    float(np.mean([g["goals"]   for g in recent])) if recent else 0.0,
        "form_assists":  float(np.mean([g["assists"] for g in recent])) if recent else 0.0,
        "season_goals":   season_goals,
        "season_assists": season_assists,
        "season_games":   season_games,
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

# ── Zápasy ────────────────────────────────────────────────
refresh_m = args.refresh or args.refresh_matches
if refresh_m:
    print("Mažu cache zápasů...")
    db_clear_matches(conn)

cached_matches = db_load_matches(conn)
if cached_matches:
    age = db_cache_age_h(conn, "matches")
    print(f"Zápasy z cache ({len(cached_matches)} záznamů, stáří {age:.1f}h)")
    if age and age > CACHE_TTL_H:
        print(f"  [!] Cache starší než {CACHE_TTL_H}h – zvažte --refresh-matches")
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
    print(f"  Uloženo: {len(matches_list)} zápasů")

max_round = max(m["round"] for m in matches_list)
print(f"  Celkem: {len(matches_list)} zápasů (max kolo: {max_round})")

# ── Hráči ze zápisů zápasů ────────────────────────────────
refresh_p = args.refresh or args.refresh_players
if refresh_p:
    print("\nMažu cache hráčů...")
    db_clear_players(conn)

SIM_TEAMS = set(t for pair in MATCHES_TO_SIMULATE for t in pair)
team_player_data: dict = {}

teams_to_fetch = []
for name in SIM_TEAMS:
    cached = db_load_players(conn, name)
    if cached:
        age = db_cache_age_h(conn, "players")
        print(f"  {name}: {len(cached)} hráčů z cache (stáří {age:.1f}h)" if age else
              f"  {name}: {len(cached)} hráčů z cache")
        team_player_data[name] = cached
    else:
        teams_to_fetch.append(name)

for name in TEAMS:
    if name not in team_player_data:
        team_player_data[name] = []

if teams_to_fetch:
    print(f"\nStahuji záznamy zápasů a hráče pro: {', '.join(teams_to_fetch)}")

    sim_match_ids = set()
    team_to_matches = defaultdict(list)
    for m in matches_list:
        for team in teams_to_fetch:
            if m["home"] == team or m["away"] == team:
                mid = m.get("match_id", "")
                if mid:
                    sim_match_ids.add(mid)
                    team_to_matches[team].append(m)

    print(f"  Stáhuji {len(sim_match_ids)} zápisů zápasů...")

    all_match_lineups = {}

    def fetch_lineup(mid):
        return mid, get_match_lineups(mid)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_lineup, mid): mid for mid in sim_match_ids}
        done = 0
        for fut in as_completed(futures):
            mid, lineups = fut.result()
            all_match_lineups[mid] = lineups
            done += 1
            if done % 5 == 0 or done == len(sim_match_ids):
                print(f"    Záznamy: {done}/{len(sim_match_ids)}")

    for team in teams_to_fetch:
        t_matches = team_to_matches[team]
        players = build_team_players_from_matches(team, t_matches, all_match_lineups, debug=True)

        if not players:
            print(f"  [!] {team}: žádní hráči nenalezeni v zápisech zápasů, zkouším soupisku...")
            tid = TEAMS.get(team, "")
            if tid:
                try:
                    roster_url = f"https://www.ceskyflorbal.cz/team/detail/roster/{tid}"
                    r = requests.get(roster_url, headers=HEADERS, timeout=15)
                    soup = BeautifulSoup(r.text, "html.parser")
                    players = []
                    for link in soup.select("a[href*='/person/detail/player/']"):
                        pid = re.search(r"/person/detail/player/(\d+)", link["href"])
                        h3 = link.select_one("h3")
                        if pid and h3:
                            players.append({
                                "player_id": pid.group(1),
                                "name":      h3.get_text(strip=True),
                                "goals":     0,
                                "assists":   0,
                                "games":     1,
                            })
                    print(f"    Soupiska: {len(players)} hráčů")
                except Exception as e:
                    print(f"    Fallback selhal: {e}")

        print(f"  {team}: {len(players)} hráčů z zápisů zápasů")

        def fetch_form(p):
            form = get_player_form(p["player_id"])
            result = {**p, **form}
            sg = form.get("season_goals",   0)
            sa = form.get("season_assists", 0)
            sz = form.get("season_games",   0)
            if sz > result["games"] or sg > result["goals"]:
                result["goals"]   = max(sg, result["goals"])
                result["assists"] = max(sa, result["assists"])
                result["games"]   = max(sz, result["games"])
            return result

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            players = list(ex.map(fetch_form, players))

        ok = sum(1 for p in players if p.get("form_goals", 0) > 0)
        print(f"    z toho {ok} s formou > 0")

        team_player_data[team] = players
        db_save_players(conn, team, players)

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

    PRIOR_STRENGTH = 4

    all_games = sum(p["games"] for p in players)
    all_goals = sum(p["goals"] for p in players)
    all_assists = sum(p["assists"] for p in players)

    league_avg_g = (all_goals   / all_games) if all_games > 0 else 0.1
    league_avg_a = (all_assists / all_games) if all_games > 0 else 0.1

    for p in players:
        g = p["games"]
        bayes_avg_g = (p["goals"]   + PRIOR_STRENGTH * league_avg_g) / (g + PRIOR_STRENGTH)
        bayes_avg_a = (p["assists"] + PRIOR_STRENGTH * league_avg_a) / (g + PRIOR_STRENGTH)

        form_avg_g = p.get("form_goals",   0.0)
        form_avg_a = p.get("form_assists", 0.0)

        if form_avg_g > 0:
            p["score_g"] = (0.35 * bayes_avg_g + 0.65 * form_avg_g)
        else:
            p["score_g"] = bayes_avg_g

        if form_avg_a > 0:
            p["score_a"] = (0.35 * bayes_avg_a + 0.65 * form_avg_a)
        else:
            p["score_a"] = bayes_avg_a

    total_g = sum(p["score_g"] for p in players) or 1.0
    total_a = sum(p["score_a"] for p in players) or 1.0

    result = []
    for p in players:
        expected_goals   = lam_team * (p["score_g"] / total_g)
        expected_assists = lam_team * 1.5 * (p["score_a"] / total_a)
        std_goals        = np.sqrt(expected_goals)
        std_assists      = np.sqrt(expected_assists)
        prob_goal        = 1 - np.exp(-expected_goals)
        prob_assist      = 1 - np.exp(-expected_assists)

        if prob_goal > 0.05:
            result.append({
                "name":             p["name"],
                "prob_goal":        prob_goal,
                "prob_assist":      prob_assist,
                "expected_goals":   expected_goals,
                "std_goals":        std_goals,
                "expected_assists": expected_assists,
                "std_assists":      std_assists,
                "games":            p["games"],
                "goals":            p["goals"],
                "assists":          p["assists"],
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
    """Vrátí slovník s výsledky simulace pro HTML generátor."""
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

    hf_home = team_form(home, matches_list)
    hf_away = team_form(away, matches_list)
    n_h2h   = len([m for m in matches_list
                   if (m["home"]==home and m["away"]==away)
                   or (m["home"]==away and m["away"]==home)])

    scorers_home = predict_scorers(home, lam_h)
    scorers_away = predict_scorers(away, lam_a)

    return {
        "home": home,
        "away": away,
        "lam_h": lam_h,
        "lam_a": lam_a,
        "lam_h_std": np.sqrt(lam_h),
        "lam_a_std": np.sqrt(lam_a),
        "prob_home_win": float(np.mean(hg > ag)),
        "prob_away_win": float(np.mean(ag > hg)),
        "prob_draw":     float(np.mean(hg == ag)),
        "avg_total_goals": float(np.mean(hg + ag)),
        "std_home": float(np.std(hg)),
        "std_away": float(np.std(ag)),
        "top_scores": [{"h": h, "a": a, "pct": cnt / N_SIM * 100} for (h, a), cnt in top],
        "h2h_h": h2h_h,
        "h2h_a": h2h_a,
        "n_h2h": n_h2h,
        "form_home": hf_home,
        "form_away": hf_away,
        "scorers_home": scorers_home,
        "scorers_away": scorers_away,
    }


# ── HTML generátor ─────────────────────────────────────────

def pct_bar(value, max_val=1.0, color="var(--accent)"):
    width = min(100, value / max_val * 100)
    return f'<div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%;background:{color}"></div></div>'


def scorers_table_html(scorers, team_name, lam_team):
    if not scorers:
        return f'<p class="no-data">Žádná data o hráčích pro {team_name}</p>'

    max_xg = max(s["expected_goals"] for s in scorers) or 1.0
    max_xa = max(s["expected_assists"] for s in scorers) or 1.0

    rows_html = ""
    for i, s in enumerate(scorers):
        medal = ""
        if i == 0: medal = '<span class="medal gold">★</span>'
        elif i == 1: medal = '<span class="medal silver">★</span>'
        elif i == 2: medal = '<span class="medal bronze">★</span>'

        xg_bar   = pct_bar(s["expected_goals"],   max_xg, "var(--accent-g)")
        xa_bar   = pct_bar(s["expected_assists"],  max_xa, "var(--accent-a)")
        prob_bar = pct_bar(s["prob_goal"], 1.0, "var(--accent)")

        rows_html += f"""
        <tr class="player-row">
          <td class="player-name">{medal}{s['name']}</td>
          <td class="stat-cell">{s['games']}</td>
          <td class="stat-cell">{s['goals']}</td>
          <td class="stat-cell">{s['assists']}</td>
          <td class="bar-cell">
            <div class="bar-label">{s['prob_goal']*100:.1f}%</div>
            {prob_bar}
          </td>
          <td class="bar-cell">
            <div class="bar-label">{s['expected_goals']:.2f} <span class="sigma">±{s['std_goals']:.2f}</span></div>
            {xg_bar}
          </td>
          <td class="bar-cell">
            <div class="bar-label">{s['prob_assist']*100:.1f}%</div>
            {xa_bar}
          </td>
          <td class="bar-cell">
            <div class="bar-label">{s['expected_assists']:.2f} <span class="sigma">±{s['std_assists']:.2f}</span></div>
            {pct_bar(s['expected_assists'], max_xa, "var(--accent-a)")}
          </td>
        </tr>"""
    return f"""
    <div class="scorers-section">
      <h3 class="scorers-title">Kandidáti na góly – {team_name}</h3>
      <div class="table-wrap">
        <table class="scorers-table">
          <thead>
            <tr>
              <th>Hráč</th>
              <th>Z</th><th>G</th><th>A</th>
              <th>Pravd. gólu</th>
              <th>xG ± σ</th>
              <th>Pravd. asistence</th>
              <th>xA ± σ</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
      </div>
    </div>"""


def match_card_html(res, idx):
    home = res["home"]
    away = res["away"]

    # Pravděpodobnostní trojúhelník
    pw = res["prob_home_win"] * 100
    pd_ = res["prob_draw"]    * 100
    pa = res["prob_away_win"] * 100

    h2h_html = ""
    if res["h2h_h"] is not None:
        h2h_html = f"""<div class="h2h-row">
          <span class="h2h-label">H2H průměr</span>
          <span class="h2h-score">{res['h2h_h']:.1f} : {res['h2h_a']:.1f}</span>
          <span class="h2h-count">({res['n_h2h']} {'zápas' if res['n_h2h'] == 1 else 'zápasů'})</span>
        </div>"""

    fh = res["form_home"]
    fa = res["form_away"]

    top_scores_html = ""
    for s in res["top_scores"]:
        top_scores_html += f"""<div class="score-pill">
          <span class="score-result">{s['h']}:{s['a']}</span>
          <span class="score-pct">{s['pct']:.1f}%</span>
        </div>"""

    scorers_html = (
        scorers_table_html(res["scorers_home"], home, res["lam_h"]) +
        scorers_table_html(res["scorers_away"], away, res["lam_a"])
    )

    return f"""
  <article class="match-card" id="match-{idx}">
    <header class="match-header">
      <div class="match-num">ZÁPAS {idx+1}</div>
      <div class="match-teams">
        <span class="team-home">{home}</span>
        <span class="vs">vs</span>
        <span class="team-away">{away}</span>
      </div>
    </header>

    <div class="match-body">
      <!-- Očekávané góly + forma -->
      <div class="xg-block">
        <div class="xg-side home-side">
          <div class="xg-value">{res['lam_h']:.2f}</div>
          <div class="xg-label">xG domácí</div>
          <div class="xg-std">σ = {res['lam_h_std']:.2f}</div>
        </div>
        <div class="xg-divider">
          <div class="xg-colon">:</div>
          <div class="xg-total-label">xG celkem<br><strong>{res['lam_h'] + res['lam_a']:.2f}</strong></div>
        </div>
        <div class="xg-side away-side">
          <div class="xg-value">{res['lam_a']:.2f}</div>
          <div class="xg-label">xG hosté</div>
          <div class="xg-std">σ = {res['lam_a_std']:.2f}</div>
        </div>
      </div>

      <!-- Win/draw/loss probs -->
      <div class="probs-block">
        <div class="prob-item">
          <div class="prob-val">{pw:.1f}%</div>
          <div class="prob-bar-wrap"><div class="prob-bar home-bar" style="height:{pw:.0f}%"></div></div>
          <div class="prob-label">Výhra dom.</div>
        </div>
        <div class="prob-item">
          <div class="prob-val">{pd_:.1f}%</div>
          <div class="prob-bar-wrap"><div class="prob-bar draw-bar" style="height:{pd_:.0f}%"></div></div>
          <div class="prob-label">Remíza</div>
        </div>
        <div class="prob-item">
          <div class="prob-val">{pa:.1f}%</div>
          <div class="prob-bar-wrap"><div class="prob-bar away-bar" style="height:{pa:.0f}%"></div></div>
          <div class="prob-label">Výhra hostů</div>
        </div>
      </div>

      <!-- Nejprav. výsledky -->
      <div class="scores-block">
        <div class="block-title">Nejpravděpodobnější výsledky</div>
        <div class="scores-grid">{top_scores_html}</div>
        <div class="avg-goals">Průměr gólů v zápase: <strong>{res['avg_total_goals']:.1f}</strong></div>
      </div>

      <!-- Forma + H2H -->
      <div class="form-block">
        <div class="block-title">Forma & H2H</div>
        <div class="form-grid">
          <div class="form-team">
            <div class="form-name">{home}</div>
            <div class="form-stats">
              <span>GF <strong>{fh['form_gf']:.1f}</strong></span>
              <span>GA <strong>{fh['form_ga']:.1f}</strong></span>
              <span>Body/z <strong>{fh['form_pts']:.1f}</strong></span>
            </div>
          </div>
          <div class="form-team">
            <div class="form-name">{away}</div>
            <div class="form-stats">
              <span>GF <strong>{fa['form_gf']:.1f}</strong></span>
              <span>GA <strong>{fa['form_ga']:.1f}</strong></span>
              <span>Body/z <strong>{fa['form_pts']:.1f}</strong></span>
            </div>
          </div>
        </div>
        {h2h_html}
      </div>
    </div>

    <!-- Střelci -->
    <div class="scorers-block">
      {scorers_html}
    </div>
  </article>"""


def generate_html(results):
    cards = "".join(match_card_html(r, i) for i, r in enumerate(results))
    now   = datetime.now().strftime("%d. %m. %Y %H:%M")

    return f"""<!DOCTYPE html>
<html lang="cs">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Florbal Simulátor – Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Barlow:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:        #0d0f14;
    --bg2:       #13161e;
    --bg3:       #1a1e28;
    --border:    #252a38;
    --text:      #e8ecf4;
    --muted:     #7a8399;
    --accent:    #00d4ff;
    --accent-g:  #00e87a;
    --accent-a:  #ff7c2a;
    --home-clr:  #4d9cff;
    --away-clr:  #ff4d7c;
    --draw-clr:  #f0c040;
    --gold:      #ffd700;
    --silver:    #c0c8d8;
    --bronze:    #cd7f32;
    --radius:    12px;
    --radius-sm: 6px;
    --font-head: 'Barlow Condensed', sans-serif;
    --font-body: 'Barlow', sans-serif;
  }}

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-body);
    font-weight: 300;
    line-height: 1.5;
    min-height: 100vh;
  }}

  /* ── Noise overlay ── */
  body::before {{
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
  }}

  /* ── Header ── */
  .site-header {{
    background: linear-gradient(135deg, #0d1220 0%, #111827 100%);
    border-bottom: 1px solid var(--border);
    padding: 40px 32px 32px;
    position: relative;
    overflow: hidden;
  }}
  .site-header::after {{
    content: '';
    position: absolute;
    bottom: 0; left: 0;
    width: 100%; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
  }}
  .header-inner {{
    max-width: 1200px;
    margin: 0 auto;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 16px;
  }}
  .logo {{
    font-family: var(--font-head);
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    line-height: 1;
    color: var(--text);
  }}
  .logo span {{
    color: var(--accent);
  }}
  .meta {{
    font-size: 0.78rem;
    color: var(--muted);
    text-align: right;
    line-height: 1.7;
  }}
  .meta strong {{ color: var(--text); }}

  /* ── Layout ── */
  .container {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 40px 24px 80px;
    position: relative;
    z-index: 1;
  }}

  /* ── Match card ── */
  .match-card {{
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 48px;
    overflow: hidden;
  }}

  .match-header {{
    background: linear-gradient(135deg, #141928, #1c2236);
    padding: 24px 32px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 20px;
  }}
  .match-num {{
    font-family: var(--font-head);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 3px;
    color: var(--accent);
    text-transform: uppercase;
    white-space: nowrap;
  }}
  .match-teams {{
    font-family: var(--font-head);
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }}
  .team-home {{ color: var(--home-clr); }}
  .team-away {{ color: var(--away-clr); }}
  .vs {{
    color: var(--muted);
    font-size: 1rem;
    font-weight: 400;
  }}

  .match-body {{
    display: grid;
    grid-template-columns: auto 1fr 1fr 1fr;
    gap: 0;
    border-bottom: 1px solid var(--border);
  }}

  /* ── xG block ── */
  .xg-block {{
    display: flex;
    align-items: center;
    padding: 28px 24px;
    gap: 16px;
    border-right: 1px solid var(--border);
    background: linear-gradient(180deg, #131726 0%, #0f1320 100%);
  }}
  .xg-side {{
    text-align: center;
    min-width: 72px;
  }}
  .xg-value {{
    font-family: var(--font-head);
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1;
  }}
  .home-side .xg-value {{ color: var(--home-clr); }}
  .away-side .xg-value {{ color: var(--away-clr); }}
  .xg-label {{
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    margin-top: 4px;
  }}
  .xg-std {{
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 2px;
  }}
  .xg-divider {{
    text-align: center;
  }}
  .xg-colon {{
    font-family: var(--font-head);
    font-size: 2rem;
    font-weight: 800;
    color: var(--muted);
  }}
  .xg-total-label {{
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    margin-top: 4px;
  }}
  .xg-total-label strong {{
    color: var(--text);
    font-size: 0.85rem;
    font-family: var(--font-head);
  }}

  /* ── Probs block ── */
  .probs-block {{
    display: flex;
    align-items: flex-end;
    justify-content: center;
    gap: 12px;
    padding: 28px 20px 20px;
    border-right: 1px solid var(--border);
  }}
  .prob-item {{
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    width: 56px;
  }}
  .prob-val {{
    font-family: var(--font-head);
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
  }}
  .prob-bar-wrap {{
    width: 28px;
    height: 60px;
    background: var(--bg3);
    border-radius: 4px;
    display: flex;
    align-items: flex-end;
    overflow: hidden;
  }}
  .prob-bar {{
    width: 100%;
    border-radius: 4px 4px 0 0;
    min-height: 2px;
    transition: height 0.5s ease;
  }}
  .home-bar {{ background: var(--home-clr); }}
  .draw-bar {{ background: var(--draw-clr); }}
  .away-bar {{ background: var(--away-clr); }}
  .prob-label {{
    font-size: 0.6rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    line-height: 1.2;
  }}

  /* ── Scores block ── */
  .scores-block {{
    padding: 24px 20px;
    border-right: 1px solid var(--border);
  }}
  .block-title {{
    font-family: var(--font-head);
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--muted);
    margin-bottom: 12px;
  }}
  .scores-grid {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 12px;
  }}
  .score-pill {{
    display: flex;
    align-items: center;
    gap: 6px;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4px 10px;
  }}
  .score-result {{
    font-family: var(--font-head);
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
  }}
  .score-pct {{
    font-size: 0.72rem;
    color: var(--accent);
    font-weight: 500;
  }}
  .avg-goals {{
    font-size: 0.78rem;
    color: var(--muted);
  }}
  .avg-goals strong {{ color: var(--text); }}

  /* ── Form block ── */
  .form-block {{
    padding: 24px 20px;
  }}
  .form-grid {{
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-bottom: 12px;
  }}
  .form-name {{
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--text);
    margin-bottom: 4px;
  }}
  .form-stats {{
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
  }}
  .form-stats span {{
    font-size: 0.72rem;
    color: var(--muted);
  }}
  .form-stats strong {{ color: var(--text); }}
  .h2h-row {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.75rem;
    color: var(--muted);
    padding-top: 10px;
    border-top: 1px solid var(--border);
    margin-top: 8px;
  }}
  .h2h-score {{
    font-family: var(--font-head);
    font-size: 1rem;
    font-weight: 700;
    color: var(--accent);
  }}

  /* ── Scorers ── */
  .scorers-block {{
    padding: 0 32px 32px;
  }}
  .scorers-section {{
    margin-top: 28px;
  }}
  .scorers-title {{
    font-family: var(--font-head);
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--text);
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }}
  .table-wrap {{
    overflow-x: auto;
    border-radius: var(--radius-sm);
  }}
  .scorers-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    min-width: 680px;
  }}
  .scorers-table th {{
    font-family: var(--font-head);
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--muted);
    text-align: left;
    padding: 8px 12px;
    background: var(--bg3);
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }}
  .scorers-table th:not(:first-child) {{ text-align: center; }}
  .player-row {{
    transition: background 0.15s;
    border-bottom: 1px solid var(--border);
  }}
  .player-row:last-child {{ border-bottom: none; }}
  .player-row:hover {{ background: var(--bg3); }}
  .player-name {{
    padding: 10px 12px;
    font-weight: 500;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
  }}
  .stat-cell {{
    padding: 10px 12px;
    text-align: center;
    color: var(--muted);
  }}
  .bar-cell {{
    padding: 8px 12px;
    min-width: 120px;
  }}
  .bar-label {{
    font-size: 0.78rem;
    color: var(--text);
    margin-bottom: 4px;
    text-align: right;
  }}
  .sigma {{
    font-size: 0.68rem;
    color: var(--muted);
  }}
  .bar-track {{
    height: 4px;
    background: var(--bg3);
    border-radius: 2px;
    overflow: hidden;
  }}
  .bar-fill {{
    height: 100%;
    border-radius: 2px;
    transition: width 0.4s ease;
  }}
  .medal {{ font-size: 0.85rem; }}
  .gold   {{ color: var(--gold);   }}
  .silver {{ color: var(--silver); }}
  .bronze {{ color: var(--bronze); }}

  .no-data {{
    color: var(--muted);
    font-style: italic;
    padding: 16px 0;
    font-size: 0.85rem;
  }}

  /* ── Footer ── */
  footer {{
    text-align: center;
    padding: 24px;
    color: var(--muted);
    font-size: 0.75rem;
    border-top: 1px solid var(--border);
    position: relative;
    z-index: 1;
  }}

  @media (max-width: 860px) {{
    .match-body {{
      grid-template-columns: 1fr 1fr;
    }}
    .form-block {{
      grid-column: 1 / -1;
      border-top: 1px solid var(--border);
    }}
  }}
  @media (max-width: 600px) {{
    .match-body {{ grid-template-columns: 1fr; }}
    .xg-block, .probs-block, .scores-block, .form-block {{
      border-right: none;
      border-bottom: 1px solid var(--border);
    }}
    .match-teams {{ font-size: 1.1rem; }}
    .scorers-block {{ padding: 0 16px 24px; }}
  }}
</style>
</head>
<body>

<header class="site-header">
  <div class="header-inner">
    <div>
      <div class="logo">Florbal<span>Sim</span></div>
    </div>
    <div class="meta">
      Generováno: <strong>{now}</strong><br>
      Simulace: <strong>{N_SIM:,}</strong> iterací &nbsp;|&nbsp;
      Model: <strong>Poissonova regrese</strong><br>
      Decay: <strong>{DECAY}</strong> &nbsp;|&nbsp;
      Forma: poslední <strong>{TEAM_FORM_N}</strong> kol
    </div>
  </div>
</header>

<main class="container">
  {cards}
</main>

<footer>
  Florbal Simulátor &copy; {datetime.now().year} &nbsp;|&nbsp;
  Data: ceskyflorbal.cz &nbsp;|&nbsp;
  Výsledky jsou pouze statistické odhady.
</footer>

</body>
</html>"""


# ── Spuštění ───────────────────────────────────────────────

print(f"\nSpouštím {len(MATCHES_TO_SIMULATE)} simulací...")
results = []
for home, away in MATCHES_TO_SIMULATE:
    print(f"  Simuluji: {home} vs {away}")
    results.append(simulate(home, away))

html = generate_html(results)
Path(HTML_OUTPUT).write_text(html, encoding="utf-8")
print(f"\nReport uložen do: {HTML_OUTPUT}")
print(f"Otevřete v prohlížeči: file://{Path(HTML_OUTPUT).resolve()}")
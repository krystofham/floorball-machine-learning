"""
Florbal simulátor s SQLite cache
Hráče tahá ze zápisů zápasů (ne ze soupisky), takže zachytí i hosty a hráče z jiných soutěží.

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
from collections import Counter, defaultdict
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
CACHE_TTL_H   = 12

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

        # Vytáhni match_id z odkazu
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
#
# Struktura detailu zápasu na ceskyflorbal.cz:
#   .MatchCenter-teamTitle-home  → zkrácený název domácích (např. "Horní SucháHSH")
#   .MatchCenter-teamTitle-quest → zkrácený název hostů
#   .MatchCenter-player-home     → každý domácí hráč (odkaz na profil)
#   .MatchCenter-player-quest    → každý hostující hráč
#   .MatchCenter-statistics--left  → tabulky statistik domácích (1.góly 2.asistence 3.TM)
#   .MatchCenter-statistics--right → tabulky statistik hostů
#   Řádek stat. tabulky: cells=[jméno, počet] + odkaz na hráče

def _fix_name(s):
    """Opraví slité jméno 'MartinBÚRAN' → 'Martin Búran' a odstraní pozici."""
    s = re.sub(r"([a-záčďéěíňóřšťúůýž])([A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ])", r"\1 \2", s)
    s = re.sub(r"\s*(brankář|útočník|obránce|záložník)\s*$", "", s, flags=re.IGNORECASE)
    return s.strip()


def get_match_lineups(match_id):
    """
    Ze zápisu stáhne soupisky obou týmů s góly a asistencemi.
    Vrátí dict: { zkrácený_název_týmu: [ {player_id, name, goals, assists} ] }
    """
    url = f"https://www.ceskyflorbal.cz/match/detail/default/{match_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
    except Exception as e:
        print(f"    [!] Chyba zápis {match_id}: {e}")
        return {}

    soup = BeautifulSoup(r.text, "html.parser")

    # 1. Zkrácené názvy týmů (ořež 2–4 písmennou zkratku na konci)
    def clean_title(el):
        txt = el.get_text(strip=True)
        return re.sub(r"[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]{2,4}$", "", txt).strip()

    home_el  = soup.select_one(".MatchCenter-teamTitle-home")
    guest_el = soup.select_one(".MatchCenter-teamTitle-quest")
    if not home_el or not guest_el:
        return {}

    home_short  = clean_title(home_el)
    guest_short = clean_title(guest_el)

    # 2. Sestavy – kdo nastoupil
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

    # 3. Stat tabulky: [0]=góly [1]=asistence
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

    # 4. Sestav výsledek
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
    """
    Agreguje statistiky hráčů z jednotlivých zápisů zápasů.
    Klíče v lineups jsou zkrácené názvy z webu (např. "Vítkovice", "Horní Suchá").
    """
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

        # Hledáme soupisku pro náš tým:
        # Zkrácený název ze zápisu musí být podřetězcem plného názvu týmu,
        # nebo naopak. Normalizujeme (bez diakritiky, lowercase).
        team_players = None
        norm_tn = normalize(team_name)

        for lineup_team, players in lineups.items():
            norm_lt = normalize(lineup_team)
            # Přesný match nebo jeden obsahuje druhý
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
    """
    Vrátí formu hráče (průměr posledních n zápasů v competition_filter)
    a také sezónní součty ze VŠECH soutěží (season_goals, season_assists, season_games).
    Sezónní součty použijeme jako fallback/korekci pokud záznamy zápasů
    zachytily jen část hráčovy aktivity (hraje ve více soutěžích).
    """
    url = f"https://www.ceskyflorbal.cz/person/detail/player/{player_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
    except Exception:
        return {"form_goals": 0.0, "form_assists": 0.0,
                "season_goals": 0, "season_assists": 0, "season_games": 0}
    soup = BeautifulSoup(r.text, "html.parser")
    tables = soup.select("table")

    # ── A) Sezónní součty ze soutěžní tabulky (Soutěž / Z / B / A) ──
    # Bereme VŠECHNY řádky aktuální sezóny (ne jen filtrovanou soutěž),
    # protože hráč může hrát za stejný tým ve více soutěžích.
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

    # ── B) Forma: posledních n zápasů filtrovaných dle competition_filter ──
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

# Pro týmy mimo simulaci – prázdná data (pro model stačí zápasy)
for name in TEAMS:
    if name not in team_player_data:
        team_player_data[name] = []

if teams_to_fetch:
    print(f"\nStahuji záznamy zápasů a hráče pro: {', '.join(teams_to_fetch)}")

    # Zjisti které match_id jsou relevantní pro simulované týmy
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

    # Stáhni záznamy zápasů paralelně
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

    # Agreguj hráče pro každý tým
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

        # Stáhni formu hráčů paralelně + korekce ze sezónních součtů z profilu
        def fetch_form(p):
            form = get_player_form(p["player_id"])
            result = {**p, **form}
            sg = form.get("season_goals",   0)
            sa = form.get("season_assists", 0)
            sz = form.get("season_games",   0)
            # Hráč může hrát ve více soutěžích – profil má kompletní data
            if sz > result["games"] or sg > result["goals"]:
                result["goals"]   = max(sg, result["goals"])
                result["assists"] = max(sa, result["assists"])
                result["games"]   = max(sz, result["games"])
            return result

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            players = list(ex.map(fetch_form, players))

        ok = sum(1 for p in players if p.get("form_goals", 0) > 0)
        print(f"    z toho {ok} s formou > 0")
        for p in players:
            sg = p.get("season_goals", 0)
            sz = p.get("season_games", 0)
            if sz > 0 and (sz != p["games"] or sg != p["goals"]):
                print(f"    [korekce] {p['name']}: {p['games']}Z/{p['goals']}G "
                      f"(profil: {sz}Z/{sg}G)")

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

    # ── Bayesovský prior ──────────────────────────────────────
    # Hráč s 1 gólem v 1 zápase dostane průměr blízký průměru týmu, ne 1.0.
    # prior_strength = kolik "fiktivních" zápasů přidáme s průměrem ligy.
    # Čím více reálných zápasů hráč má, tím méně prior ovlivní výsledek.
    # Typicky: hráč s < 3 zápasy je silně přitažen k průměru.
    PRIOR_STRENGTH = 4  # ekvivalent 4 průměrných zápasů jako prior

    all_games = sum(p["games"] for p in players)
    all_goals = sum(p["goals"] for p in players)
    all_assists = sum(p["assists"] for p in players)

    # Průměr na zápas přes celý tým (ligový prior)
    league_avg_g = (all_goals   / all_games) if all_games > 0 else 0.1
    league_avg_a = (all_assists / all_games) if all_games > 0 else 0.1

    for p in players:
        g = p["games"]
        # Bayesovský odhad: (skutečné góly + prior) / (skutečné zápasy + prior_strength)
        bayes_avg_g = (p["goals"]   + PRIOR_STRENGTH * league_avg_g) / (g + PRIOR_STRENGTH)
        bayes_avg_a = (p["assists"] + PRIOR_STRENGTH * league_avg_a) / (g + PRIOR_STRENGTH)

        form_avg_g = p.get("form_goals",   0.0)
        form_avg_a = p.get("form_assists", 0.0)

        # Forma (posledních 5 zápasů) má váhu jen pokud existuje
        # Forma také prochází Bayesovským vyhlazením – jen mírnějším (kratší okno)
        FORM_PRIOR = 2
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
    std_hg = float(np.std(hg))
    std_ag = float(np.std(ag))

    hf_home = team_form(home, matches_list)
    hf_away = team_form(away, matches_list)
    n_h2h   = len([m for m in matches_list
                   if (m["home"]==home and m["away"]==away)
                   or (m["home"]==away and m["away"]==home)])

    sep = "─" * 60
    lines = []
    lines.append(sep)
    lines.append(f"  {home}  vs  {away}")
    lines.append(sep)
    lines.append(f"Očekávané góly: {lam_h:.2f} ± {np.sqrt(lam_h):.2f}  :  {lam_a:.2f} ± {np.sqrt(lam_a):.2f}")
    if h2h_h is not None:
        lines.append(f"H2H průměr:     {h2h_h:.1f} : {h2h_a:.1f}  ({n_h2h} vzájemný{'ch zápasů' if n_h2h > 1 else ' zápas'})")
    lines.append(f"Forma domácích: GF={hf_home['form_gf']:.1f}  GA={hf_home['form_ga']:.1f}  body/z={hf_home['form_pts']:.1f}")
    lines.append(f"Forma hostů:    GF={hf_away['form_gf']:.1f}  GA={hf_away['form_ga']:.1f}  body/z={hf_away['form_pts']:.1f}")
    lines.append("")
    lines.append(f"Výhra domácích:   {np.mean(hg > ag)*100:.1f}%")
    lines.append(f"Výhra hostů:      {np.mean(ag > hg)*100:.1f}%")
    lines.append(f"Remíza (60 min):  {np.mean(hg == ag)*100:.1f}%")
    lines.append(f"Průměr gólů:      {np.mean(hg + ag):.1f}  (σ dom={std_hg:.2f}, host={std_ag:.2f})")
    lines.append("")
    lines.append("Nejpravděpodobnější výsledky:")
    for (h, a), cnt in top:
        lines.append(f"  {h}:{a}  →  {cnt / N_SIM * 100:.1f}%")

    for team, lam in [(home, lam_h), (away, lam_a)]:
        scorers = predict_scorers(team, lam)
        if not scorers:
            lines.append(f"\n  [!] {team}: žádná data o hráčích")
            continue
        lines.append(f"\nKandidáti na góly – {team}:")
        lines.append(f"  {'Hráč':<28} {'Z':>3} {'G':>3} {'A':>3}  {'gól%':>6}  {'očekáv.±σ':>10}  {'as%':>6}  {'očekáv.±σ':>10}")
        lines.append(f"  {'─'*28} {'─'*3} {'─'*3} {'─'*3}  {'─'*6}  {'─'*10}  {'─'*6}  {'─'*10}")
        for s in scorers:
            lines.append(
                f"  {s['name']:<28} "
                f"{s['games']:>3} {s['goals']:>3} {s['assists']:>3}  "
                f"{s['prob_goal']*100:>5.1f}%  "
                f"{s['expected_goals']:>4.2f}±{s['std_goals']:.2f}  "
                f"{s['prob_assist']*100:>5.1f}%  "
                f"{s['expected_assists']:>4.2f}±{s['std_assists']:.2f}"
            )

    lines.append(sep)
    print("\n" + "\n".join(lines))


for home, away in MATCHES_TO_SIMULATE:
    simulate(home, away)
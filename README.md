# 🏑 Florbal Machine Learning Simulator

Pokročilý nástroj pro predikci výsledků florbalových zápasů a individuálních výkonů hráčů. Skript využívá reálná data z Českého florbalu, která analyzuje pomocí statistických modelů.

## 🚀 Hlavní Funkce
- **Scraping dat:** Automatické stahování výsledků zápasů, soupisek a statistik hráčů pomocí `BeautifulSoup`.
- **SQLite Caching:** Efektivní ukládání dat do lokální databáze pro rychlé opětovné spuštění bez nutnosti nového stahování.
- **Poisson Regression:** Statistický model (z knihovny `scikit-learn`) odhadující útočnou a obrannou sílu týmů na základě historických výsledků s časovým útlumem (novější zápasy mají větší váhu).
- **Monte Carlo Simulace:** 100 000 simulací pro každý zápas k určení pravděpodobnosti výhry, remízy a nejpravděpodobnějšího skóre.
- **Predikce střelců:** Výpočet pravděpodobnosti vstřelení gólu nebo asistence u konkrétních hráčů na základě jejich sezónní a aktuální formy.

## 🛠️ Technologie
- **Jazyk:** Python 3
- **Analýza dat:** Pandas, NumPy, Scikit-learn
- **Sběr dat:** Requests, BeautifulSoup4
- **Databáze:** SQLite3
- **Ostatní:** Multithreading (ThreadPoolExecutor) pro rychlý scraping

## 📊 Jak to funguje
Model nebere v úvahu jen "kdo je lepší", ale počítá s:
1. **Home Advantage:** Faktor domácího prostředí.
2. **Time Decay:** Starší výsledky mají menší váhu než ty aktuální.
3. **Player Form:** Pokud klíčový hráč v posledních 5 zápasech boduje, model zvýší pravděpodobnost jeho úspěchu v simulaci.
4. **H2H (Head-to-Head):** Historické vzájemné zápasy týmů.

---
*Poznámka: Tento projekt byl vyvinut s využitím AI pro optimalizaci statistických výpočtů a struktury kódu.*

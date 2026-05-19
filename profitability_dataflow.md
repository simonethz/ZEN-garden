Mathematischer Datenfluss – Profitabilitätsberechnung
Datei: zen_garden/plugins/investment_decisions/investment_decisions.py
Zielgröße: calculate_profitability → diskontierter Nettogewinn pro (Technologie, Knoten)

Notation
Symbol	Bedeutung	Einheit
tech	Konversionstechnologie	–
oc	Output-Carrier	–
node	Netzwerkknoten	–
t	aggregierter Betriebszeitschritt	–
n	Jahresoffset ab Investitionszeitpunkt	Jahre
L	Lebensdauer der Technologie	Jahre
d	Investitionsverzögerung (aktuell = 0)	Jahre
r	Diskontierungsrate	–
ΔC	Kapazitätszubau (aktuell = 1 GW)	GW
Δτ[t]	Dauer des Zeitschritts t	h
year(t)	Optimierungsjahr des Zeitschritts t	–
1 · Schattenpreise (Duale Variablen)
extract_aggregated_dual
Ausgangspunkt: Rohduale aus der Knotenenergiebilanzbedingung des LP-Solvers.

λ_raw[carrier, node, t]   ← constraint_nodal_energy_balance.dual
Normierung auf ein Optimierungsjahr (Intervall zwischen optimierten Jahren):

λ_agg[carrier, node, t] = λ_raw[carrier, node, t] / annuity[year(t)]
annuity[year] = interval_between_years für alle Jahre außer dem letzten (= 1).

extract_normalized_dual
Umrechnung von aggregierten Zeitschritten auf Stundenwerte (Division durch Zeitschrittdauer):

λ[carrier, node, t] = λ_agg[carrier, node, t] / Δτ[t]
Einheit: Geldeinheit / MWh (= Marktpreis-Äquivalent pro Energieeinheit)

get_shadow_price
Abbildung Carrier → Technologie über die Output-Carrier-Zuordnung:

p[tech, oc, node, t] = λ[oc, node, t]    für alle oc ∈ output_carriers(tech)
2 · Spezifische Produktion
get_specific_production
q[tech, oc, node, t] = f_out[tech, oc, node, t] · Δτ[t]
                        ─────────────────────────────────
                              C[tech, node, year(t)]
Variable	Quelle	Einheit
f_out	flow_conversion_output (Lösung)	MW
Δτ[t]	time_steps_operation_duration	h
C	capacity (Lösung, summiert über capacity_types)	GW
Einheit des Ergebnisses: MWh / GW pro Zeitschritt
→ Summe über alle t eines Jahres = Volllaststunden [h] bzw. jährliche Produktion pro GW [MWh/GW]

3 · Revenue (Erlösberechnung)
calculate_revenue
Schritt 1 – Jährlicher Erlös pro (tech, oc, node)

R_annual[tech, oc, node] =  Σ_t  p[tech, oc, node, t] · q[tech, oc, node, t]

                         =  Σ_t  λ[oc, node, t]  ·  f_out[tech, oc, node, t] · Δτ[t]
                                                     ──────────────────────────────────
                                                           C[tech, node, year(t)]
Einheit: Geldeinheit / GW / Jahr

Schritt 2 – Diskontierter Lebenszeitlerlös

                        L + d - 1
                           Σ          R_annual[tech, oc, node]
R[tech, oc, node] = ΔC ·  Σ        ──────────────────────────
                          n = d              (1 + r)^n
Der einzige optimierte Betriebszeitraum wird als repräsentatives Jahr für alle n Lebensjahre verwendet.
d = Investitionsverzögerung (aktuell 0), L = Lebensdauer.
Einheit: Geldeinheit (diskontierter Erlös über gesamte Lebensdauer)

Schritt 3 – Aggregation über Output-Carrier (in calculate_profitability)

R_total[tech, node] = Σ_oc  R[tech, oc, node]
4 · CAPEX
get_capex
Fall A – Linear (set_capex_linear)

CAPEX[tech, node] = capex_specific[tech, node, year_max]  ·  ΔC[tech]
Variable	Quelle	Einheit
capex_specific	parameters.capex_specific_conversion (letztes Jahr)	Geldeinheit/GW
ΔC	_capacity_addition (aktuell 1 GW)	GW
Einheit: Geldeinheit

Fall B – PWA (set_capex_pwa)

CAPEX[tech, node] = interp_linear(ΔC[tech];  breakpoints_capacity,  breakpoints_cost)
Stückweise lineare Interpolation (node-unabhängig, gleicher Wert für alle Knoten).

5 · Fixed OPEX (diskontiert)
get_fixed_opex_discounted
Für jedes Lebensjahr n ∈ [d, L+d-1] wird das zugehörige Datenjahr bestimmt:

Liegt n in set_time_steps_yearly → direkte Verwendung
n > year_max → Extrapolation mit year_max
n < year_min → Extrapolation mit year_min
Dazwischen → lineare Interpolation zwischen den benachbarten Datenjahren
                            L + d - 1
                               Σ          opex_fixed[tech, node, year(n)]
OPEX_fixed[tech, node] = ΔC · Σ        ────────────────────────────────
                              n = d                (1 + r)^n
Variable	Quelle	Einheit
opex_fixed	parameters.opex_specific_fixed (summiert über capacity_types)	Geldeinheit / GW / Jahr
Einheit: Geldeinheit (diskontierter Fixkostenstrom)

6 · Variable OPEX (diskontiert)
get_variable_opex_discounted
Schritt 1 – Spezifischer Referenzcarrier-Fluss (get_flow_reference_carrier)

q_ref[tech, node, t] = f_ref[tech, node, t] · Δτ[t]
                        ─────────────────────────────
                           C[tech, node, year(t)]
Der Referenzcarrier kann Input- oder Output-Carrier sein (wird in set_reference_carriers definiert).

Einheit: MWh / GW pro Zeitschritt

Schritt 2 – Jährliche variable OPEX-Kosten pro GW

OPEX_var_annual[tech, node, year] = Σ_{t ∈ year}  c_var[tech, node, t]  ·  q_ref[tech, node, t]

                                  = Σ_{t ∈ year}  c_var[tech, node, t]  ·  f_ref · Δτ[t]
                                                                             ──────────────
                                                                              C[tech, node]
Variable	Quelle	Einheit
c_var	parameters.opex_specific_variable	Geldeinheit / MWh
q_ref	aus get_flow_reference_carrier	MWh / GW
Einheit: Geldeinheit / GW / Jahr

Schritt 3 – Diskontierung

Gleiche Jahres-Lookup-Logik wie bei Fixed OPEX (Extrapolation/Interpolation):

                             L + d - 1
                                Σ          OPEX_var_annual[tech, node, year(n)]
OPEX_var[tech, node] = ΔC  ·  Σ        ────────────────────────────────────────
                               n = d                  (1 + r)^n
Einheit: Geldeinheit (diskontierter variabler Kostenstrom)

7 · Profitabilität
calculate_profitability
Profit[tech, node] = R_total[tech, node]
                   − CAPEX[tech, node]
                   − OPEX_fixed[tech, node]
                   − OPEX_var[tech, node]
Index: (set_conversion_technologies, set_nodes)
Einheit: Geldeinheit
Interpretation: Diskontierter Nettowert eines hypothetischen Kapazitätszubaus von ΔC = 1 GW über die gesamte Technologielebensdauer.

Übersicht Datenfluss
Solver-Lösung (LP-Duale)
    └─► extract_aggregated_dual   ──÷ annuity──►
        extract_normalized_dual   ──÷ Δτ[t]──►
            get_shadow_price      ──(Carrier→Tech-Mapping)──►  p[tech, oc, node, t]
                                                                       │
Solver-Lösung (Flows + Capacity)                                       │
    └─► get_specific_production ──►  q[tech, oc, node, t]             │
                                            │                          │
                                            └─────── × ───────────────┘
                                                       │
                                               R_annual[tech, oc, node]    [GE/GW/a]
                                                       │
                                          Σ_n 1/(1+r)^n  ×  ΔC
                                                       │
                                         R[tech, oc, node]  →  Σ_oc  →  R_total[tech, node]
                                                                                   │
parameters.capex_specific_conversion                                               │
    └─► get_capex  ────────────────────────────────────────────────────►  CAPEX[tech, node]
                                                                                   │
parameters.opex_specific_fixed                                                     │
    └─► get_fixed_opex_discounted  ─────────────────────────────────►  OPEX_fixed[tech, node]
                                                                                   │
parameters.opex_specific_variable × get_flow_reference_carrier                     │
    └─► get_variable_opex_discounted  ──────────────────────────────►  OPEX_var[tech, node]
                                                                                   │
                                          ┌────────────────────────────────────────┘
                                          │
                          Profit = R_total − CAPEX − OPEX_fixed − OPEX_var
Anmerkungen & bekannte Vereinfachungen
Thema	Aktueller Stand
Repräsentatives Jahr	Ein einziges optimiertes Betriebsjahr wird auf alle Lebensjahre hochgerechnet
Fixed OPEX Jahresentwicklung	Für Jahre außerhalb der Datenjahre wird der nächste verfügbare Wert verwendet (kein zeitliches OPEX-Wachstum innerhalb der Lebensdauer)
Kapazitätszubau ΔC	Aktuell fest = 1 GW für alle Technologien (_capacity_addition)
Investitionsverzögerung d	Aktuell = 0 für alle Technologien (_get_investment_delay)
Diskontierungsrate r	Systemweit einheitlich (kein tech-/knotenspezifischer Wert)
CAPEX-Diskontierung	CAPEX wird nicht diskontiert (nur einmaliger Betrag zum Investitionszeitpunkt)

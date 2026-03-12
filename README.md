# Mon IA Bourse - Système autonome quantitatif

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Scheduler

```bash
python scripts/scheduler.py
```

## Sortie par actif

- prob_up / prob_down
- confidence
- market_regime
- cycle_score
- explosion_score
- smart_money_score
- strategy_recommended
- risk_position_pct

## Robustesse runtime

- Si `torch` n'est pas installé, le pipeline bascule automatiquement sur un modèle sklearn (fallback explicite).
- Le statut RL/GNN est signalé dans la sortie (`rl_status`, `gnn_status`).

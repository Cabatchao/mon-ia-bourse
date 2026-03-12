# Mon IA Bourse - Plateforme Quant Deep Learning

## Lancement rapide

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

## Structure du projet

- `data/` collecte + stockage data lake
- `features/` feature engineering technique et statistique
- `models/` modèles deep learning et ensemble
- `training/` pipeline d'entraînement/validation temporelle
- `backtesting/` métriques et moteur d'évaluation stratégie
- `predictions/` formatage sorties probabilistes
- `utils/` modules distribués, risque, apprentissage continu
- `config/` configuration centralisée
- `docs/` documentation technique

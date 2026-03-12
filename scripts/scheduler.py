"""Automatisation: analyse quotidienne, ré-entraînement périodique, rapport hebdomadaire."""
from __future__ import annotations

import schedule
import time

from app import run_daily_analysis, run_retraining, run_weekly_report


schedule.every().day.at("07:00").do(run_daily_analysis)
schedule.every().monday.at("08:00").do(run_weekly_report)
schedule.every().day.at("23:00").do(run_retraining)


if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(30)

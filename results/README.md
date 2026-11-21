# Companion dashboard

The interactive evidence locker is generated locally (not committed) so GitHub language stats stay focused on Python source.

```bash
python db/seed_db.py          # if db/deal_desk.db is missing
python -m src.build_report    # writes results/index.html
open results/index.html       # or your browser of choice
```

Benchmark CSV/JSON inputs live in `outputs/` and are copied into `results/data/` by the report builder.

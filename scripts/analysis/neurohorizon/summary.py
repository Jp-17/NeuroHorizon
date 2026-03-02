import csv, json, os

base = "/root/autodl-tmp/NeuroHorizon/results/logs"
experiments = [
    ("250ms AR", "phase1_small_250ms", 12),
    ("500ms AR", "phase1_small_500ms", 25),
    ("1000ms AR", "phase1_small_1000ms", 50),
    ("1000ms noAR", "phase1_small_1000ms_noar", 50),
]

header = "{:<16} {:>4} {:<12} {:>8} {:>7} {:>8} {:>8}".format(
    "Experiment", "Bins", "Status", "BestR2", "BestEp", "FinalR2", "FinalVL")
print("=" * 80)
print("Phase 1.3 Prediction Window Experiments")
print("=" * 80)
print(header)
print("-" * 80)

results = []
for name, folder, nbins in experiments:
    path = os.path.join(base, folder, "lightning_logs/version_0/metrics.csv")
    if not os.path.exists(path):
        print("{:<16} {:>4} {:<12}".format(name, "--", "not_started"))
        results.append(dict(name=name, status="not_started"))
        continue
    with open(path) as f:
        reader = csv.DictReader(f)
        vals = []
        for row in reader:
            vl = row.get("val_loss", "")
            if vl and vl.strip():
                vals.append(row)
    if not vals:
        print("{:<16} {:>4} {:<12}".format(name, str(nbins), "no_val_data"))
        results.append(dict(name=name, status="no_validation"))
        continue
    best = max(vals, key=lambda r: float(r.get("val/r2", "0")))
    last = vals[-1]
    last_ep = int(last.get("epoch", 0))
    status = "completed" if last_ep >= 290 else "in_progress"
    br2 = float(best.get("val/r2", "0"))
    bep = int(best.get("epoch", 0))
    fr2 = float(last.get("val/r2", "0"))
    fvl = float(last["val_loss"])
    print("{:<16} {:>4} {:<12} {:>8.4f} {:>7d} {:>8.4f} {:>8.4f}".format(
        name, nbins, status, br2, bep, fr2, fvl))
    results.append(dict(name=name, status=status, bins=nbins,
        best_r2=br2, best_epoch=bep, final_r2=fr2, final_val_loss=fvl,
        last_epoch=last_ep, num_checkpoints=len(vals)))

print()
out = os.path.join(base, "phase1_summary.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print("Saved:", out)

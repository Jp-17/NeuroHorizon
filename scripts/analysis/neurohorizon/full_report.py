import csv, json, os

base = "/root/autodl-tmp/NeuroHorizon/results/logs"
experiments = [
    ("250ms AR", "phase1_small_250ms", 12),
    ("500ms AR", "phase1_small_500ms", 25),
    ("1000ms AR", "phase1_small_1000ms", 50),
    ("1000ms non-AR", "phase1_small_1000ms_noar", 50),
]

all_results = {}

for name, folder, nbins in experiments:
    path = os.path.join(base, folder, "lightning_logs/version_0/metrics.csv")
    with open(path) as f:
        reader = csv.DictReader(f)
        vals = []
        for row in reader:
            vl = row.get("val_loss", "")
            if vl and vl.strip():
                vals.append({
                    "epoch": int(row.get("epoch", 0)),
                    "val_loss": float(row["val_loss"]),
                    "r2": float(row.get("val/r2", 0)),
                })
    all_results[name] = {"bins": nbins, "vals": vals}

# Print full comparison
print("=" * 90)
print("Phase 1.3 - Full Prediction Window Experiment Report")
print("=" * 90)
print()

# Summary table
print("### Summary")
print()
hdr = "{:<16} {:>5} {:>8} {:>7} {:>8} {:>7} {:>8} {:>8}".format(
    "Experiment", "Bins", "BestR2", "BestEp", "FnlR2", "FnlEp", "FnlVL", "Status")
print(hdr)
print("-" * 90)

for name, data in all_results.items():
    vals = data["vals"]
    nbins = data["bins"]
    best = max(vals, key=lambda v: v["r2"])
    last = vals[-1]
    status = "OK" if last["epoch"] >= 290 else "INCOMPLETE"
    line = "{:<16} {:>5} {:>8.4f} {:>7d} {:>8.4f} {:>7d} {:>8.4f} {:>8}".format(
        name, nbins, best["r2"], best["epoch"], last["r2"], last["epoch"],
        last["val_loss"], status)
    print(line)

print()

# AR vs non-AR comparison
print("### AR vs non-AR Comparison (1000ms)")
print()
ar_vals = all_results["1000ms AR"]["vals"]
noar_vals = all_results["1000ms non-AR"]["vals"]
print("{:>7} {:>10} {:>10} {:>10}".format("Epoch", "AR_R2", "nonAR_R2", "Diff"))
print("-" * 40)
for i in range(min(len(ar_vals), len(noar_vals))):
    ar = ar_vals[i]
    noar = noar_vals[i]
    diff = noar["r2"] - ar["r2"]
    sign = "+" if diff >= 0 else ""
    print("{:>7d} {:>10.4f} {:>10.4f} {:>9}{:.4f}".format(
        ar["epoch"], ar["r2"], noar["r2"], sign, abs(diff)))

ar_best = max(ar_vals, key=lambda v: v["r2"])
noar_best = max(noar_vals, key=lambda v: v["r2"])
print()
print("AR best:    R2={:.4f} (epoch {})".format(ar_best["r2"], ar_best["epoch"]))
print("non-AR best: R2={:.4f} (epoch {})".format(noar_best["r2"], noar_best["epoch"]))
diff = noar_best["r2"] - ar_best["r2"]
sign = "+" if diff >= 0 else ""
print("Difference: {}{}".format(sign, "{:.4f}".format(diff)))

print()

# R2 degradation with window length
print("### R2 vs Prediction Window Length")
print()
for name in ["250ms AR", "500ms AR", "1000ms AR"]:
    data = all_results[name]
    best = max(data["vals"], key=lambda v: v["r2"])
    window_ms = int(name.split("ms")[0])
    print("  {}ms: bins={}, best_R2={:.4f}, per_bin_avg_R2={:.4f}".format(
        window_ms, data["bins"], best["r2"], best["r2"]))

# Detailed per-experiment curves
print()
print("### Detailed Training Curves")
for name, data in all_results.items():
    print()
    print("--- {} ({} bins) ---".format(name, data["bins"]))
    for v in data["vals"]:
        print("  epoch={:>3d}  val_loss={:.4f}  r2={:.4f}".format(
            v["epoch"], v["val_loss"], v["r2"]))
    best = max(data["vals"], key=lambda v: v["r2"])
    print("  >> Best: epoch={}, R2={:.4f}".format(best["epoch"], best["r2"]))

# Save comprehensive results
out_data = {}
for name, data in all_results.items():
    best = max(data["vals"], key=lambda v: v["r2"])
    last = data["vals"][-1]
    out_data[name] = {
        "bins": data["bins"],
        "best_r2": best["r2"],
        "best_epoch": best["epoch"],
        "final_r2": last["r2"],
        "final_val_loss": last["val_loss"],
        "final_epoch": last["epoch"],
        "all_vals": data["vals"],
    }

out_path = os.path.join(base, "phase1_full_report.json")
with open(out_path, "w") as f:
    json.dump(out_data, f, indent=2)
print()
print("Full report saved to:", out_path)

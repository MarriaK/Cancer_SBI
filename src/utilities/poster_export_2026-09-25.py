"""Per-arm posterior quantiles for example held-out tumours, three encoders, for poster panel e."""
import json, numpy as np, os
R = os.path.expanduser("~/cancer/results/2026-09-24")
runs = {"armtoken": ("AT0ens3", "posteriors_armtoken_AT0ens3.npz"),
        "cloneatt": ("R26", "posteriors_cloneatt_R26.npz"),
        "dominantclone": ("D0", "posteriors_dominantclone_D0.npz")}
qs = np.array([0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975])
out = {}
sim_ref = None
for model, (run, fn) in runs.items():
    d = np.load(f"{R}/{run}/posteriors/{fn}", allow_pickle=True)
    sims = d["sim_ids"].astype(str); theta = d["theta_true"]; S = d["samples"]
    if sim_ref is None:
        # pick 12 example tumours spread over ArmToken's per-tumour R2 (best, median, worst), deterministic
        pm = d["post_mean"]; r2_case = 1 - ((theta - pm) ** 2).sum(1) / ((theta - theta.mean(0)) ** 2).sum(1)
        order = np.argsort(r2_case); idx = np.unique(np.concatenate([order[-4:], order[len(order)//2-2:len(order)//2+2], order[:4]]))
        sim_ref = sims[idx]
    sel = np.array([np.where(sims == s)[0][0] for s in sim_ref])
    q = np.quantile(S[sel].astype(np.float32), qs, axis=1)          # (7, n, 44)
    out[model] = dict(run=run, quantiles=np.moveaxis(q, 0, -1), post_mean=d["post_mean"][sel], post_std=d["post_std"][sel])
    theta_ref = theta[sel]
np.savez_compressed(os.path.expanduser("~/cancer/results/best_honest/poster_export_2026-09-25.npz"),
    sim_ids=sim_ref, theta_true=theta_ref, quantile_levels=qs,
    **{f"{m}_quantiles": v["quantiles"] for m, v in out.items()},
    **{f"{m}_post_mean": v["post_mean"] for m, v in out.items()},
    **{f"{m}_post_std": v["post_std"] for m, v in out.items()},
    meta_json=json.dumps({m: v["run"] for m, v in out.items()} | {"quantile_axes": "(n_sims, 44, n_levels)", "sim_selection": "4 best, 4 median, 4 worst per-tumour true R2 of ArmToken AT0ens3"}))
print("saved", sim_ref.tolist())

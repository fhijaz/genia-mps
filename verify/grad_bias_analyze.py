import json

R = json.load(open("/Users/fhijaz/alt3d/speed/grad_bias_results.json"))

for tag in ("shift0", "shift4"):
    E = R[tag]
    print("===== %s  windows=%d M=%d maxlen=%d" %
          (tag, E["n_windows"], E["M"], E["max_len"]))
    for arm in ("mtl_fp16", "mtl_fp32", "baseline", "sdpa_fp32"):
        for seed in ("A", "B"):
            g = E["arms"][arm][seed]["global"]
            parts = []
            for t in ("out", "dq", "dk", "dv"):
                parts.append("%s: ma %.3e max %.3e sm %+.2e slope %+.1e" %
                             (t, g[t]["mean_abs"], g[t]["max_abs"],
                              g[t]["signed_mean"], g[t]["slope_err_on_truth"]))
            print(" %s %s  %s" % (arm, seed, "  ".join(parts)))
    for t in ("out", "dq", "dk", "dv"):
        a = E["arms"]["mtl_fp16"]["A"]["global"][t]
        b = E["arms"]["baseline"]["A"]["global"][t]
        c = E["arms"]["mtl_fp32"]["A"]["global"][t]
        print("  RATIO seedA %s: mtl_fp16/base meanabs x%.2f max x%.2f | "
              "mtl_fp32/base meanabs x%.2f | mean|truth| %.3e" %
              (t, a["mean_abs"] / b["mean_abs"], a["max_abs"] / b["max_abs"],
               c["mean_abs"] / b["mean_abs"], a["mean_abs_truth"]))
    print("  cross-seed pw dq corr:", E["cross_seed_pw_dq_corr"])
    if "determinism" in E:
        print("  determinism:", E["determinism"])
        print("  leakage:", json.dumps(E["leakage"]))
    # group breakdowns for mtl_fp16 vs baseline, seed A, dq
    for arm in ("mtl_fp16", "baseline"):
        gr = E["arms"][arm]["A"]["groups"]["dq"]
        print("  %s dq by_length: %s" % (arm, {
            k: ("sm %+.2e ma %.3e" % (v["signed_mean"], v["mean_abs"]))
            for k, v in gr["by_length"].items()}))
        print("  %s dq by_position: %s" % (arm, {
            k: ("sm %+.2e ma %.3e" % (v["signed_mean"], v["mean_abs"]))
            for k, v in gr["by_position"].items()}))
        print("  %s dq corr_meanabs_vs_len %+.3f corr_signed_vs_len %+.3f" %
              (arm, gr["corr_meanabs_vs_len"], gr["corr_signed_vs_len"]))
        bd = E["arms"][arm]["A"]["boundary"]["dq"]
        print("  %s dq boundary: %s" % (arm, {
            k: ("sm %+.2e ma %.3e n %d" % (v["signed_mean"], v["mean_abs"], v["count"]))
            for k, v in bd.items()}))
        bo = E["arms"][arm]["A"]["boundary"]["out"]
        print("  %s out boundary: %s" % (arm, {
            k: ("sm %+.2e ma %.3e" % (v["signed_mean"], v["mean_abs"]))
            for k, v in bo.items()}))

for tag in ("control_uniform128_randn", "control_uniform128_scaled",
            "control_uniform256_randn", "control_uniform256_scaled"):
    if tag not in R:
        continue
    E = R[tag]
    print("===== %s windows=%d" % (tag, E["n_windows"]))
    for arm in ("mtl_fp16", "mtl_fp32", "baseline"):
        a = E["arms"][arm]["A"]["global"]
        b = E["arms"]["baseline"]["A"]["global"]
        print("  %s dq ma %.3e (x%.2f vs base) sm %+.2e slope %+.1e | out ma %.3e" %
              (arm, a["dq"]["mean_abs"], a["dq"]["mean_abs"] / b["dq"]["mean_abs"],
               a["dq"]["signed_mean"], a["dq"]["slope_err_on_truth"],
               a["out"]["mean_abs"]))

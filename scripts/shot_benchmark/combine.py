
# --- V-JEPA2 (contexto de probabilidades) + verificador VLM ---
matches = load_matches(with_vlm=True)
n_gt = sum(len(m["gt"]) for m in matches)
print(f"{sum(len(m['y']) for m in matches)} candidatos, {sum(m['y'].sum() for m in matches)} tiros reales "
      f"(techo de recall {100 * sum(m['y'].sum() for m in matches) / n_gt:.1f}% de {n_gt})")
check_against_official(matches)
methods = {
    "V-JEPA2 (pico)": [m["curve"][m["idx"]] for m in matches],
    "VLM p_shot": [m["p_vlm"] for m in matches],
    "V-JEPA2 x VLM": [m["curve"][m["idx"]] * m["p_vlm"] for m in matches],
    "LR ctx K=3": lomo(matches, lambda m: ctx_feats(m, 3), LR()),
    "LR ctx K=3 + VLM": lomo(matches, lambda m: np.column_stack([ctx_feats(m, 3), vlm_feats(m)]), LR()),
    "LR ctx K=1 + VLM": lomo(matches, lambda m: np.column_stack([ctx_feats(m, 1), vlm_feats(m)]), LR()),
}
report(matches, methods)

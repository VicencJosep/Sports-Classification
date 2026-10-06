
# --- ¿El recorte central de V-JEPA2 es el cuello de botella? Misma cabeza LOMO, tres vistas ---
matches = load_matches(with_emb=True)  # "emb" = recorte central 256x256 (el de siempre)
for m, c in zip(matches, CACHES):
    for v in ("squash", "wide"):
        z = np.load(Path(f"output/shot_benchmark/candidate_embeddings_{v}") / f"{Path(c).parent.name}.npz")
        assert np.array_equal(z["idx"], m["idx"]), f"candidatos distintos en {m['name']} ({v})"
        m[f"emb_{v}"] = z["emb"].astype(np.float32)
    m["emb_crop"] = m["emb"]
check_against_official(matches)
methods = {
    "V-JEPA2 (pico)": [m["curve"][m["idx"]] for m in matches],
    "LR ctx K=3": lomo(matches, lambda m: ctx_feats(m, 3), LR()),
}
for v in ("crop", "squash", "wide"):
    methods[f"{v} emb C=.001"] = lomo(matches, lambda m, v=v: m[f"emb_{v}"], LR(0.001))
    methods[f"{v} emb C=.01"] = lomo(matches, lambda m, v=v: m[f"emb_{v}"], LR(0.01))
    methods[f"{v} emb+ctx"] = lomo(matches, lambda m, v=v: np.column_stack([m[f"emb_{v}"], ctx_feats(m, 3)]), LR(0.001))
report(matches, methods)

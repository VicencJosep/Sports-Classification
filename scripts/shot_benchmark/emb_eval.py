
# --- Cabeza entrenada sobre el embedding V-JEPA2 con los negativos difíciles reales ---
from sklearn.decomposition import PCA

matches = load_matches(with_emb=True)
check_against_official(matches)
PCA_LR = lambda: make_pipeline(StandardScaler(), PCA(64, random_state=0), LogisticRegression(C=0.1, max_iter=5000))
methods = {
    "V-JEPA2 (pico)": [m["curve"][m["idx"]] for m in matches],
    "LR ctx K=3": lomo(matches, lambda m: ctx_feats(m, 3), LR()),
    "LR emb C=0.001": lomo(matches, lambda m: m["emb"], LR(0.001)),
    "LR emb C=0.01": lomo(matches, lambda m: m["emb"], LR(0.01)),
    "PCA64+LR emb": lomo(matches, lambda m: m["emb"], PCA_LR),
    "LR emb+ctx": lomo(matches, lambda m: np.column_stack([m["emb"], ctx_feats(m, 3)]), LR(0.001)),
}
report(matches, methods)

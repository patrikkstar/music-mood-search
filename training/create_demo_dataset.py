from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 360

# Искусственные признаки размерности EffNet-Discogs.
x = rng.normal(0, 0.3, (n, 1280)).astype(np.float32)
labels = rng.integers(0, 2, size=(n, 3))

# Связь между признаками и метками для проверки обучения.
x[:, :3] += np.where(labels == 1, 1.0, -1.0)

df = pd.DataFrame(x, columns=[f"e{i}" for i in range(1280)])
df["track_id"] = [f"demo_{i:05d}" for i in range(n)]
df["artist_id"] = [f"artist_{i:05d}" for i in range(n)]
df["album_id"] = [f"album_{i:05d}" for i in range(n)]
df["duration"] = 120.0
df["split"] = ["train"] * 280 + ["validation"] * 40 + ["test"] * 40

for j, (category, tags) in enumerate(
    {
        "genre": ("pop", "rock"),
        "instrument": ("piano", "guitar"),
        "mood": ("calm", "energetic"),
    }.items()
):
    df[category] = [tags[v] for v in labels[:, j]]

out = Path("training/data/out/demo_embeddings.parquet")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out, index=False)

print(f"Created: {out}")
print(f"Tracks: {len(df)}")
print(df["split"].value_counts().to_string())

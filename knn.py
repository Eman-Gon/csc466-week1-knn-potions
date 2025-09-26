import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

df_views = pd.read_parquet("data/content_views.parquet")
df_subs  = pd.read_parquet("data/subscriptions.parquet")

pub_counts = df_subs.groupby("publisher_id")["adventurer_id"].nunique()
publisher_id = pub_counts.idxmax()
print(f"Publisher chosen: {publisher_id}")

subs_pub = df_subs[df_subs["publisher_id"] == publisher_id]
sub_ids = set(subs_pub["adventurer_id"].unique())

views_pub = df_views[
    (df_views["publisher_id"] == publisher_id) &
    (df_views["adventurer_id"].isin(sub_ids))
].copy()

views_pub["value"] = 1
user_item = (
    views_pub.groupby(["adventurer_id", "content_id"])["value"]
    .max().unstack(fill_value=0).astype(np.float32)
)

print(f"User-item matrix: {user_item.shape}")

item_user = user_item.T
knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=min(20, len(item_user)))
knn.fit(item_user.values)

items_index = item_user.index.to_list()
item_matrix = item_user.values

def recommend2(uid: str) -> list[str]:
    """Recommend 2 unseen items for a user"""
    if uid not in user_item.index:
        return []
    seen_idx = np.where(user_item.loc[uid].values > 0)[0]
    if len(seen_idx) == 0:
        return []
    dists, idxs = knn.kneighbors(item_matrix[seen_idx], return_distance=True)
    scores = np.zeros(user_item.shape[1], dtype=np.float32)
    for drow, irow in zip(dists, idxs):
        sims = 1.0 - drow
        np.add.at(scores, irow, sims)
    scores[seen_idx] = -np.inf
    top = np.argsort(-scores)[:2]
    return [user_item.columns[i] for i in top if np.isfinite(scores[i])]

user_activity = user_item.sum(axis=1).sort_values(ascending=False).index.tolist()
rows, picked_users = [], []
for uid in user_activity:
    recs = recommend2(uid)
    if len(recs) == 2:
        rows.append((uid, recs[0], recs[1]))
        picked_users.append(uid)
    if len(rows) >= 3:
        break

eval_df = pd.DataFrame(rows, columns=["adventurer_id", "rec1", "rec2"])
eval_df.to_csv("eval.csv", index=False)
print("\nTop 3 users with recommendations (saved to eval.csv):")
print(eval_df)

rng = np.random.default_rng(42)
def recall_at_2(recs, test):
    return len(set(recs[:2]) & test) / len(test) if test else 0.0

eval_rows = []
for uid in picked_users:
    seen_items = list(views_pub.loc[views_pub["adventurer_id"] == uid, "content_id"].unique())
    if len(seen_items) < 3:
        continue
    test_items = set(rng.choice(seen_items, size=2, replace=False))
    train_items = set(seen_items) - test_items
    # temporarily mask test items
    for t in test_items: user_item.at[uid, t] = 0.0
    recs = recommend2(uid)
    r = recall_at_2(recs, test_items)
    tp = len(set(recs) & test_items)
    fp = len(set(recs) - test_items)
    fn = len(test_items - set(recs))
    eval_rows.append((uid, list(train_items), list(test_items), recs, r, tp, fp, fn))
    # restore
    for t in test_items: user_item.at[uid, t] = 1.0

if eval_rows:
    eval_out = pd.DataFrame(eval_rows, columns=["adventurer_id","train_items","test_items","recs","recall@2","TP","FP","FN"])
    print("\nEvaluation results (holdout recall@2):")
    print(eval_out.to_string(index=False))
    print(f"\nAverage recall@2: {eval_out['recall@2'].mean():.3f}")
else:
    print("\nNot enough items for evaluation.")
import pandas as pd
import random

# 读取用户-POI 交互
checkins = pd.read_csv("Gowalla_checkins.txt", sep="\t", header=None, names=["user", "item", "time"])
user_positive = checkins.groupby("user")["item"].apply(set).to_dict()
all_items = set(checkins["item"].unique())

# 构造 ratings 数据
ratings = []

for user, pos_items in user_positive.items():
    # 正样本
    for item in pos_items:
        ratings.append((user, item, 1))
    # 负样本：从未交互的item中随机采样与正样本等数量
    neg_candidates = list(all_items - pos_items)
    sampled_neg = random.sample(neg_candidates, min(len(pos_items), len(neg_candidates)))
    for item in sampled_neg:
        ratings.append((user, item, 0))

# 输出为 ratings_final.txt
ratings_df = pd.DataFrame(ratings, columns=["user", "item", "rating"])
ratings_df.to_csv("ratings_final.txt", sep="\t", index=False, header=False)
print(f"写入完成，共包含记录数：{len(ratings_df)}")

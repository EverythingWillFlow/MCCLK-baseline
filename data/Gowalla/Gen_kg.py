import pandas as pd
import math

# ------------------ 地理距离函数 ------------------
def geo_distance(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# ------------------ 读取 user-item 交互 ------------------
checkins = pd.read_csv("Gowalla_checkins.txt", sep="\t", header=None, names=["user", "item", "time"])
interact_triples = checkins[["user", "item"]].drop_duplicates()
interact_triples["rel"] = 0

# ------------------ 读取 user-user 社交关系 ------------------
social = pd.read_csv("Gowalla_social_relations.txt", sep="\t", header=None, names=["user1", "user2"])
social_triples = pd.concat([
    social,
    social.rename(columns={"user1": "user2", "user2": "user1"})  # 加双向关系
]).drop_duplicates()
social_triples["rel"] = 1
social_triples.columns = ["head", "tail", "rel"]

# ------------------ 构建 item-item 地理近邻关系 ------------------
poi = pd.read_csv("Gowalla_poi_coos.txt", sep="\t", header=None, names=["item", "lat", "lon"])
poi = poi.set_index("item")
item_items = []

item_ids = list(poi.index)
for i in range(len(item_ids)):
    for j in range(i + 1, len(item_ids)):
        id1, id2 = item_ids[i], item_ids[j]
        lat1, lon1 = poi.loc[id1]
        lat2, lon2 = poi.loc[id2]
        if geo_distance(lat1, lon1, lat2, lon2) < 1.0:
            item_items.append((id1, id2, 2))
            item_items.append((id2, id1, 2))

item_df = pd.DataFrame(item_items, columns=["head", "tail", "rel"])

# ------------------ 合并所有关系 ------------------
interact_df = interact_triples.rename(columns={"user": "head", "item": "tail", "rel": "rel"})
kg_all = pd.concat([interact_df, social_triples, item_df])
kg_all = kg_all.drop_duplicates()

# ------------------ 输出 ------------------
kg_all.to_csv("kg_final.txt", sep="\t", index=False, header=False)
print(f"知识图谱构建完成，共包含三元组 {len(kg_all)} 条。")


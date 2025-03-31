# 读取原始文件并创建映射
relation_mapping = {}
current_id = 0

with open('./kg.txt', 'r') as f_in, open('./kg_final.txt', 'w') as f_out:
    for line in f_in:
        parts = line.strip().split(' ')
        if len(parts) != 3:
            continue

        head, relation, tail = parts

        # 如果关系类型未映射，分配新ID
        if relation not in relation_mapping:
            relation_mapping[relation] = current_id
            current_id += 1

        # 写入新格式
        f_out.write(f"{head}\t{relation_mapping[relation]}\t{tail}\n")
    print("over")


    # 处理kg_final-s.txt中新增的行(如果有)
    # 这部分需要根据实际情况调整
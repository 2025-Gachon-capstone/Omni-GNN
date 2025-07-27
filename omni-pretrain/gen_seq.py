import pandas as pd
import pickle as pkl
import numpy as np
import argparse
import os
from collections import defaultdict

parser = argparse.ArgumentParser(description="Instacart BERT-style Preprocessing")
parser.add_argument("--data_dir", type=str, default="../data", help="data directory.")
parser.add_argument("--bizdate", type=str, default="0000", help="output suffix.")
parser.add_argument("--dup", type=str, default="True", help="whether to do transaction duplication")
args = parser.parse_args()

print("Duplication:", args.dup)

def seq_duplicate(df):
    """
    72시간 내 동일 product_id 반복 구매 시 하나로 통합하며 count 누적
    """
    user2seq = {}
    df = df.sort_values(by=["user_id", "order_number", "add_to_cart_order"])

    for user_id, user_df in df.groupby("user_id"):
        current_time = 0
        time_list = []

        for row in user_df.itertuples():
            delta_days = row.days_since_prior_order
            hour = row.order_hour_of_day
            seconds = (delta_days if not pd.isna(delta_days) else 0) * 86400 + hour * 3600
            current_time += seconds
            time_list.append((row, current_time))

        user_seq = []
        i = 0
        while i < len(time_list):
            row, t = time_list[i]
            pid = row.product_id
            count = 1
            j = i + 1

            while j < len(time_list):
                row_next, t_next = time_list[j]
                if row_next.product_id == pid and (t_next - t) <= 86400 * 3:
                    count += 1
                    j += 1
                else:
                    break

            user_seq.append({
                "product_id": row.product_id,
                "reordered": row.reordered,
                "hour": row.order_hour_of_day,
                "aisle": row.aisle_id,
                "department": row.department_id,
                "count": count
            })
            i = j

        if len(user_seq) > 2:
            user2seq[user_id] = user_seq

    return user2seq

def seq_generation(df):
    """
    통합 없이 순서대로 구매 시퀀스 생성
    """
    user2seq = defaultdict(list)
    df = df.sort_values(by=["user_id", "order_number", "add_to_cart_order"])
    for row in df.itertuples():
        user2seq[row.user_id].append({
            "product_id": row.product_id,
            "reordered": row.reordered,
            "hour": row.order_hour_of_day,
            "aisle": row.aisle_id,
            "department": row.department_id,
            "count": 1
        })
    return {uid: seq for uid, seq in user2seq.items() if len(seq) > 2}

def feature_bucketization(user2seq):
    for uid in user2seq:
        for token in user2seq[uid]:
            cnt = token["count"]
            if cnt <= 1:
                token["count_bucket"] = 1
            elif cnt <= 3:
                token["count_bucket"] = 2
            elif cnt <= 5:
                token["count_bucket"] = 3
            elif cnt <= 10:
                token["count_bucket"] = 4
            else:
                token["count_bucket"] = 5
    return user2seq

def main():
    input_path = os.path.join(args.data_dir, "master_dataset_with_roles.csv")
    df = pd.read_csv(input_path)

    # 필터링: 학습에 사용할 prior만
    df = df[df["role_train"] == "prior"]

    # 선택 컬럼만 유지
    df = df[[
        "user_id", "product_id", "order_number", "add_to_cart_order",
        "reordered", "order_hour_of_day", "days_since_prior_order",
        "aisle_id", "department_id"
    ]].copy()

    # 시퀀스 생성
    if args.dup == "True":
        user2seq = seq_duplicate(df)
    else:
        user2seq = seq_generation(df)

    user2seq = feature_bucketization(user2seq)

    # 통계 출력
    lens = [len(seq) for seq in user2seq.values()]
    print("Total users:", len(user2seq))
    print("Mean length:", np.mean(lens))
    print("Median length:", np.median(lens))

    # 저장
    os.makedirs("../outputs", exist_ok=True)
    out_path = f"../outputs/user2seq_instacart_{args.bizdate}.pkl"
    with open(out_path, "wb") as f:
        pkl.dump(user2seq, f)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()

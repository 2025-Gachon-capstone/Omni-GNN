import pandas as pd
import pickle as pkl
import numpy as np
import argparse
import os
from collections import defaultdict

parser = argparse.ArgumentParser(description="Instacart BERT-style Preprocessing")
parser.add_argument("--data_dir", type=str, default="../data", help="data directory.")
parser.add_argument("--bizdate", type=str, default="0000", help="output suffix.")
args = parser.parse_args()

def seq_from_df(df):
    user2seq = defaultdict(list)
    df = df.sort_values(by=["user_id", "order_number", "add_to_cart_order"]).reset_index(drop=True)
    for row in df.itertuples():
        user2seq[row.user_id].append({
            "product_id": row.product_id,
            "reordered": row.reordered,
            "hour": row.order_hour_of_day,
            "aisle": row.aisle_id,
            "department": row.department_id,
            "count": 1,
            "days_since_prior_order": row.days_since_prior_order
        })
    return dict(user2seq)

def seq_duplicate_dict(user2seq_raw):
    user2seq = {}
    for uid, seq in user2seq_raw.items():
        merged_seq = []
        current_time = 0
        time_list = []

        for token in seq:
            delta_days = token.get("days_since_prior_order", 0)
            hour = token["hour"]
            seconds = (delta_days if not pd.isna(delta_days) else 0) * 86400 + hour * 3600
            current_time += seconds
            time_list.append((token, current_time))

        i = 0
        while i < len(time_list):
            token, t = time_list[i]
            pid = token["product_id"]
            count = 1
            j = i + 1
            while j < len(time_list):
                nxt_token, t_next = time_list[j]
                if nxt_token["product_id"] == pid and (t_next - t) <= 86400 * 3:
                    count += 1
                    j += 1
                else:
                    break
            merged_seq.append({
                "product_id": pid,
                "reordered": token["reordered"],
                "hour": token["hour"],
                "aisle": token["aisle"],
                "department": token["department"],
                "count": count
            })
            i = j

        user2seq[uid] = merged_seq
    return user2seq

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

def make_split2seq(df):
    df = df.sort_values(by=["user_id", "order_number", "add_to_cart_order"]).reset_index(drop=True)
    split2seq = {"train": {}, "val": {}, "test": {}}

    for uid, user_df in df.groupby("user_id"):
        # split 기준
        eval_set = user_df["eval_set"].iloc[-1]

        # prior 주문만 따로 추출
        prior_df = user_df[user_df["eval_set"] == "prior"]
        prior_seq_raw = seq_from_df(prior_df)

        if uid not in prior_seq_raw:
            continue

        # prior 통합
        prior_seq_merged = seq_duplicate_dict({uid: prior_seq_raw[uid]})
        if uid not in prior_seq_merged or len(prior_seq_merged[uid]) <= 2:
            continue

        # eval_set별 분기
        if eval_set == "prior":
            split2seq["train"][uid] = prior_seq_merged[uid]

        elif eval_set == "train":
            val_df = user_df[user_df["eval_set"] == "train"]
            val_seq = seq_from_df(val_df).get(uid, [])
            split2seq["train"][uid] = prior_seq_merged[uid]
            split2seq["val"][uid] = prior_seq_merged[uid] + val_seq

        elif eval_set == "test":
            test_df = user_df[user_df["eval_set"] == "test"]
            test_seq = seq_from_df(test_df).get(uid, [])
            split2seq["train"][uid] = prior_seq_merged[uid]
            split2seq["test"][uid] = prior_seq_merged[uid] + test_seq

    return split2seq

def print_dup_stats(raw_dict, dup_dict):
    for split in ["train", "val", "test"]:
        raw_seqs = raw_dict.get(split, {})
        dup_seqs = dup_dict.get(split, {})

        raw_users = set(raw_seqs.keys())
        dup_users = set(dup_seqs.keys())

        print(f"\n📊 Split: {split}")
        print(f"  - 사용자 수         : {len(raw_users):,} → {len(dup_users):,} (제거 {len(raw_users - dup_users):,}명)")

        raw_lens = [len(seq) for uid, seq in raw_seqs.items() if uid in dup_users]
        dup_lens = [len(seq) for seq in dup_seqs.values()]

        if raw_lens and dup_lens:
            print(f"  - 통합 전 토큰 수   : {sum(raw_lens):,}")
            print(f"  - 통합 후 토큰 수   : {sum(dup_lens):,}")
            print(f"  - 평균 시퀀스 길이  : {np.mean(raw_lens):.2f} → {np.mean(dup_lens):.2f}")
            print(f"  - 길이 감소율       : {(1 - sum(dup_lens)/sum(raw_lens))*100:.2f}%")
        else:
            print("  - 시퀀스 없음")

def save_split2seq(split2seq, tag, bizdate):
    out_dir = f"../outputs/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    for split in split2seq:
        path = os.path.join(out_dir, f"{split}_seq_{bizdate}.pkl")
        with open(path, "wb") as f:
            pkl.dump(split2seq[split], f)
        print(f"✅ Saved: {path}")

def main():
    input_path = os.path.join(args.data_dir, "master_dataset.csv")
    df = pd.read_csv(input_path)

    df = df[[
        "user_id", "product_id", "order_number", "add_to_cart_order",
        "reordered", "order_hour_of_day", "days_since_prior_order",
        "aisle_id", "department_id", "eval_set"
    ]].copy()

    split2seq_raw = make_split2seq(df)
    split2seq = {
        split: feature_bucketization(seq_duplicate_dict(split2seq_raw[split]))
        for split in split2seq_raw
    }

    print_dup_stats(split2seq_raw, split2seq)

    save_split2seq(split2seq_raw, tag="raw", bizdate=args.bizdate)
    save_split2seq(split2seq, tag="dup", bizdate=args.bizdate)

if __name__ == "__main__":
    main()
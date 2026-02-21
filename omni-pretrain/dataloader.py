import torch
import math
import random
import torch.utils.data as data_utils
import copy
import numpy as np

class InstacartDataloader:

    def __init__(self, args, vocab, user2seq):
        self.args = args
        self.vocab = vocab
        self.rng = random.Random(args.dataloader_random_seed)
        self.seq_list = self.preprocess(user2seq)

    def preprocess(self, user2seq):
        max_len = self.args.max_seq_length - 1  # for [USR]
        seqs = []

        for user_id, seq in user2seq.items():
            if len(seq) > max_len:
                # sliding window if needed
                for i in range(0, len(seq) - max_len + 1, max_len):
                    seqs.append((user_id, seq[i:i+max_len]))
            else:
                seqs.append((user_id, seq))

        self.rng.shuffle(seqs)
        print("Total sequences:", len(seqs))
        return seqs

    def get_train_loader(self):
        dataset = InstacartTrainDataset(self.args, self.vocab, self.seq_list)
        return data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True, pin_memory=True)

    def get_eval_loader(self):
        dataset = InstacartEvalDataset(self.args, self.vocab, self.seq_list)
        return data_utils.DataLoader(dataset, batch_size=self.args.eval_batch_size, shuffle=False, pin_memory=True)


class InstacartTrainDataset(data_utils.Dataset):

    def __init__(self, args, vocab, seq_list):
        self.args = args
        self.vocab = vocab
        self.seq_list = seq_list
        self.rng = random.Random(args.dataloader_random_seed)
        self.max_predictions_per_seq = math.ceil(args.max_seq_length * args.masked_lm_prob)

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, index):
        user_id, tranxs = copy.deepcopy(self.seq_list[index])
        user_token = f"[USR_{user_id}]"
        user_id_token = self.vocab.convert_tokens_to_ids([user_token])[0]

        # mask product_ids
        cand_indexes = list(range(1, len(tranxs) + 1))  # exclude user_id pos (0)
        self.rng.shuffle(cand_indexes)
        labels = [-1] * (len(tranxs) + 1)

        # prepend dummy feature for user
        tranxs = [{"product_id": user_id_token, "reordered": 0, "hour": 0, "aisle": 0, "department": 0, "count_bucket": 0, "positions": 0}] + tranxs

        num_masked = 0
        num_to_predict = min(self.max_predictions_per_seq, max(1, int(len(tranxs) * self.args.masked_lm_prob)))
        for idx in cand_indexes:
            if num_masked >= num_to_predict:
                break
            labels[idx] = tranxs[idx]["product_id"]
            tranxs[idx]["product_id"] = self.vocab.mask_token_id
            num_masked += 1

        # extract features
        input_ids = [t["product_id"] for t in tranxs]
        reordered = [t["reordered"]+1 for t in tranxs]
        hour = [t["hour"]+1 for t in tranxs]
        aisle = [t["aisle"] for t in tranxs]
        dept = [t["department"] for t in tranxs]
        count_bucket = [t["count_bucket"] for t in tranxs]
        positions = list(range(len(input_ids)))
        input_mask = [1] * len(input_ids)

        reordered_tensor = torch.LongTensor(reordered)
        if reordered_tensor.min() < 0 or reordered_tensor.max() > 2:
            print(f"[ERROR] reordered index out of range: min={reordered_tensor.min()}, max={reordered_tensor.max()}")

        # padding
        pad_len = self.args.max_seq_length - len(input_ids)
        for feat in [input_ids, reordered, hour, aisle, dept, count_bucket, input_mask, labels, positions]:
            if feat is labels:
                feat += [-1] * pad_len
            else:
                feat += [0] * pad_len

        return {
            "user_id": torch.LongTensor([user_id]),
            "input_ids": torch.LongTensor(input_ids),
            "reordered": torch.LongTensor(reordered),
            "hour": torch.LongTensor(hour),
            "aisle": torch.LongTensor(aisle),
            "dept": torch.LongTensor(dept),
            "count_bucket": torch.LongTensor(count_bucket),
            "positions": torch.LongTensor(positions),
            "input_mask": torch.LongTensor(input_mask),
            "labels": torch.LongTensor(labels)
        }


class InstacartEvalDataset(data_utils.Dataset):

    def __init__(self, args, vocab, seq_list):
        self.args = args
        self.vocab = vocab
        self.seq_list = seq_list

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, index):
        user_id, tranxs = self.seq_list[index]
        user_token = f"[USR_{user_id}]"
        user_id_token = self.vocab.convert_tokens_to_ids([user_token])[0]

        # prepend user token
        tranxs = [{"product_id": user_id_token, "reordered": 0, "hour": 0, "aisle": 0, "department": 0, "count_bucket": 0}] + tranxs

        input_ids = [t["product_id"] for t in tranxs]
        reordered = [t["reordered"]+1 for t in tranxs]
        hour = [t["hour"]+1 for t in tranxs]
        aisle = [t["aisle"] for t in tranxs]
        dept = [t["department"] for t in tranxs]
        count_bucket = [t["count_bucket"] for t in tranxs]
        positions = list(range(len(input_ids)))
        input_mask = [1] * len(input_ids)

        pad_len = self.args.max_seq_length - len(input_ids)
        for feat in [input_ids, reordered, hour, aisle, dept, count_bucket, input_mask, positions]:
            feat += [0] * pad_len

        return {
                "user_id": torch.LongTensor([user_id]),
                "input_ids": torch.LongTensor(input_ids),
                "reordered": torch.LongTensor(reordered),
                "hour": torch.LongTensor(hour),
                "aisle": torch.LongTensor(aisle),
                "dept": torch.LongTensor(dept),
                "count_bucket": torch.LongTensor(count_bucket),
                "positions": torch.LongTensor(positions),
                "input_mask": torch.LongTensor(input_mask),
            }

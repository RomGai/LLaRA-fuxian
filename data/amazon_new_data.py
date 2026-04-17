import os
import random
import hashlib
import torch.utils.data as data


class AmazonNewData(data.Dataset):
    def __init__(self, data_dir="new_data",
                 stage=None,
                 cans_num=10,
                 dataset_prefix="Baby_Products",
                 seq_max_len=10,
                 val_user_ratio=0.05,
                 seed=1234,
                 padding_item_id=None,
                 no_augment=True,
                 **kwargs):
        self.__dict__.update(locals())
        self.rng = random.Random(seed)
        self.item_id2name = self.get_item_id2name()
        self.item_ids = sorted(list(self.item_id2name.keys()))
        if self.padding_item_id is None:
            self.padding_item_id = max(self.item_ids) + 1
        self.check_data_leakage()
        self.session_data = self.load_rows_for_stage()

    def __len__(self):
        return len(self.session_data)

    def __getitem__(self, i):
        temp = self.session_data[i]
        target = temp["next"]
        if self.stage == "test":
            candidates = self.negative_sampling_for_eval(temp["seq_unpad"], target)
        else:
            candidates = self.negative_sampling_for_train(temp["seq_unpad"], target, temp["neg"])
        cans_name = [self.item_id2name[can] for can in candidates]
        sample = {
            "user_id": temp["user_id"],
            "seq": temp["seq"],
            "seq_name": [self.item_id2name[x] for x in temp["seq_unpad"]],
            "len_seq": temp["len_seq"],
            "cans": candidates,
            "cans_name": cans_name,
            "len_cans": len(candidates),
            "item_id": target,
            "item_name": self.item_id2name[target],
            "correct_answer": self.item_id2name[target]
        }
        return sample

    def check_data_leakage(self):
        train_users = set()
        test_users = set()
        for uid, _, _ in self.iter_triplets(self.stage_file("train")):
            train_users.add(uid)
        for uid, _, _ in self.iter_triplets(self.stage_file("test")):
            test_users.add(uid)
        overlap = train_users & test_users
        if overlap:
            raise ValueError(f"Data leakage detected: {len(overlap)} overlapped users between train/test.")
        print(f"[DataLeakageCheck] train_users={len(train_users)} test_users={len(test_users)} overlap={len(overlap)}")

    def stage_file(self, stage):
        return os.path.join(self.data_dir, f"{self.dataset_prefix}_user_items_negs_{stage}.csv")

    def get_item_id2name(self):
        path = os.path.join(self.data_dir, f"{self.dataset_prefix}_i_map.tsv")
        item_id2name = {}
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                original, item_id = line.split("\t")
                item_id = int(item_id)
                item_id2name[item_id] = original.strip()
        return item_id2name

    def iter_triplets(self, path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 3:
                    continue
                user_id = int(parts[0])
                pos = [int(x) for x in parts[1].split(",") if x]
                neg = [int(x) for x in parts[2].split(",") if x]
                if len(pos) < 2:
                    continue
                yield user_id, pos, neg

    def user_in_val_split(self, user_id):
        digest = hashlib.md5(f"{user_id}_{self.seed}".encode("utf-8")).hexdigest()
        score = int(digest[:8], 16) / 0xFFFFFFFF
        return score < self.val_user_ratio

    def build_row(self, user_id, pos, neg):
        target = pos[-1]
        seq_unpad = pos[:-1]
        seq_kept = seq_unpad[-self.seq_max_len:]
        len_seq = len(seq_kept)
        seq = seq_kept + [self.padding_item_id] * (self.seq_max_len - len_seq)
        return {
            "user_id": user_id,
            "seq_unpad": seq_unpad,
            "seq": seq,
            "len_seq": len_seq,
            "next": target,
            "neg": neg,
        }

    def load_rows_for_stage(self):
        rows = []
        if self.stage in {"train", "val"}:
            for user_id, pos, neg in self.iter_triplets(self.stage_file("train")):
                in_val = self.user_in_val_split(user_id)
                if self.stage == "train" and in_val:
                    continue
                if self.stage == "val" and not in_val:
                    continue
                rows.append(self.build_row(user_id, pos, neg))
        elif self.stage == "test":
            for user_id, pos, neg in self.iter_triplets(self.stage_file("test")):
                rows.append(self.build_row(user_id, pos, neg))
        else:
            raise ValueError(f"Unsupported stage: {self.stage}")
        print(f"[Dataset] stage={self.stage} size={len(rows)} prefix={self.dataset_prefix}")
        return rows

    def negative_sampling_for_train(self, seq_unpad, next_item, fixed_neg):
        seen = set(seq_unpad)
        seen.add(next_item)
        candidates = []
        for x in fixed_neg:
            if x in seen:
                continue
            candidates.append(x)
            if len(candidates) >= self.cans_num - 1:
                break
        while len(candidates) < self.cans_num - 1:
            x = self.rng.choice(self.item_ids)
            if x not in seen and x not in candidates:
                candidates.append(x)
        candidates.append(next_item)
        self.rng.shuffle(candidates)
        return candidates

    def negative_sampling_for_eval(self, seq_unpad, next_item):
        seen = set(seq_unpad)
        seen.add(next_item)
        all_candidates = [x for x in self.item_ids if x not in seen]
        neg_num = max(1, self.cans_num - 1)
        if len(all_candidates) < neg_num:
            sampled = all_candidates
        else:
            sampled = self.rng.sample(all_candidates, neg_num)
        candidates = sampled + [next_item]
        self.rng.shuffle(candidates)
        return candidates

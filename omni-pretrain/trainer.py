import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW


class InstacartTrainer(nn.Module):
    def __init__(self, args, vocab, model, data_loader):
        super().__init__()
        self.args = args
        self.device = args.device
        self.vocab = vocab
        self.model = model.to(self.device)
        self.data_loader = data_loader
        self.num_epochs = args.num_epochs
        self.optimizer, self.lr_scheduler = self._create_optimizer()

        self.dense = nn.Linear(args.hidden_size, args.hidden_size).to(self.device)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12).to(self.device)

        word_num = len(vocab.vocab_words) - self.args.user_size - 3
        if args.neg_strategy == "uniform":
            self.weights = torch.ones(word_num)
        elif args.neg_strategy == "zip": # Log-uniform (Zipfian) negative sampling
            self.weights = 1 / torch.arange(1., word_num + 1)
        elif args.neg_strategy == "freq": # Frequency-based negative sampling
            self.weights = torch.tensor(list(map(lambda x: pow(x, 1 / 1), vocab.frequency[self.args.user_size + 3:])), dtype=torch.float)
        else:
            raise ValueError("Please select correct negative sampling strategy: uniform, zip, freq.")

    def calculate_loss(self, batch):
        _, input_ids, reordered, hour, aisle, dept, count_bucket, input_mask, labels, positions = batch
        input_ids, reordered, hour, aisle, dept, count_bucket, input_mask, labels, positions = [x.to(self.device) for x in (input_ids, reordered, hour, aisle, dept, count_bucket, input_mask, labels, positions)]

        h = self.model(input_ids, reordered, hour, aisle, dept, count_bucket, positions)
        input_tensor = self.dense(h)
        input_tensor = F.gelu(input_tensor)
        input_tensor = self.LayerNorm(input_tensor)

        label_mask = labels != -1
        labels = torch.where(label_mask, labels, torch.zeros_like(labels))

        pos_embed = self.model.embedding.token_embed(labels)

        # 체크포인트 1: weight 길이
        assert len(self.weights) == self.args.product_size, f"Mismatch! weights: {len(self.weights)}, expected: {self.args.product_size}"

        # 체크포인트 2: negative index 유효성
        neg_ids = torch.multinomial(self.weights, self.args.neg_sample_num, replacement=False).to(self.device)
        assert (neg_ids < self.args.product_size).all()

        # 최종 ID
        neg_embed_ids = neg_ids + self.args.user_size + 3

        # 체크포인트 3: vocab 범위 초과 확인
        assert neg_embed_ids.max().item() < self.args.vocab_size

        # 임베딩
        neg_embed = self.model.embedding.token_embed(neg_embed_ids)



        pos_logits = torch.sum(input_tensor * pos_embed, dim=-1).unsqueeze(-1)
        neg_logits = torch.matmul(input_tensor, neg_embed.t())
        logits = torch.cat([pos_logits, neg_logits], dim=2)
        log_probs = torch.log_softmax(logits, -1)
        per_example_loss = -log_probs[:, :, 0]

        loss = torch.sum(label_mask * per_example_loss) / (torch.sum(label_mask) + 1e-5)
        return loss

    def train(self):
        accum_step = 0
        self.best_loss = float('inf')  # 가장 낮은 loss를 저장

        for epoch in range(self.num_epochs):
            epoch_loss, accum_step = self.train_one_epoch(epoch, accum_step)
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.save_model(epoch + 1, self.args.ckpt_dir, epoch_loss)

    def train_one_epoch(self, epoch, accum_step):
        self.model.train()
        total_loss = 0.0
        total_steps = 0

        for batch_idx, batch in enumerate(tqdm(self.data_loader)):
            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            if self.args.enable_lr_schedule:
                self.lr_scheduler.step()

            loss_value = loss.item()
            total_loss += loss_value
            total_steps += 1
            accum_step += 1

            tqdm.write(f"Epoch {epoch + 1}, Step {accum_step}, Loss {loss_value:.4f}")

        epoch_loss = total_loss / total_steps
        return epoch_loss, accum_step   


    def save_model(self, epoch, ckpt_dir, loss_value):
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pth")
        torch.save(self.model.state_dict(), ckpt_path)

        # 텍스트 로그로 loss 저장
        log_path = os.path.join(ckpt_dir, "loss_log.txt")
        with open(log_path, "a") as f:
            f.write(f"Epoch {epoch}, Loss {loss_value:.4f}\n")

        print(f"Model saved to {ckpt_path} with Loss {loss_value:.4f}")

    def _create_optimizer(self):
        optimizer = AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min((step + 1) / self.args.num_warmup_steps, 1.0)
            if step < self.args.num_warmup_steps else
            (self.args.num_train_steps - step) / max(1, self.args.num_train_steps - self.args.num_warmup_steps)
        )
        return optimizer, scheduler

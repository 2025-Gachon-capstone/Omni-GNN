from config import args
from dataloader import InstacartDataloader
from modeling import InstacartBERT  # 기존 BERT4ETH -> InstacartBERT로 이름 맞추기
from trainer import InstacartTrainer
import pickle as pkl
import torch
import os


def infer_embed():
    print("=========== Load Sequence ===========")
    with open(os.path.join(args.data_dir, f"user2seq_{args.bizdate}.pkl"), "rb") as f:
        user2seq = pkl.load(f)

    with open(os.path.join(args.data_dir, f"{args.vocab_filename}.{args.bizdate}"), "rb") as f:
        vocab = pkl.load(f)

    dataloader = InstacartDataloader(args, vocab, user2seq)
    eval_loader = dataloader.get_eval_loader()

    model = InstacartBERT(args)
    trainer = InstacartTrainer(args, vocab, model, eval_loader)

    ckpt_path = os.path.join(args.ckpt_dir, "epoch_10.pth")
    trainer.load(ckpt_path)

    # 특정 임베딩 벡터 출력
    token_weights = model.embedding.token_embed.weight  # nn.Embedding
    print("Token 200000:", token_weights[200000, :])
    print("Token 800000:", token_weights[800000, :])


if __name__ == "__main__":
    infer_embed()

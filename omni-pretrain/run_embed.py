from config import args
from dataloader import InstacartDataloader
from modeling import InstacartBERT
from trainer import InstacartTrainer
import pickle as pkl
import numpy as np
import os

def infer_embed():

    # prepare dataset
    print("===========Load Sequence===========")
    with open(os.path.join(args.data_dir, f"user2seq_instacart_{args.bizdate}.pkl"), "rb") as f:
        user2seq = pkl.load(f)

    # load vocab
    with open(os.path.join(args.data_dir, f"{args.vocab_filename}.{args.bizdate}"), "rb") as f:
        vocab = pkl.load(f)

    # dataloader
    dataloader = InstacartDataloader(args, vocab, user2seq)
    eval_loader = dataloader.get_eval_loader()

    # model
    model = InstacartBERT(args)

    # trainer
    trainer = InstacartTrainer(args, vocab, model, eval_loader)
    trainer.load(args.init_checkpoint)
    address_array, seq_embedding_array = trainer.infer_embedding()

    # save embedding
    checkpoint_name = args.init_checkpoint.split("/")[0]
    model_index = str(args.init_checkpoint.split("/")[-1].split(".pth")[0])

    output_dir = os.path.join(args.data_dir, f"{checkpoint_name}_{model_index}")
    os.makedirs(output_dir, exist_ok=True)

    embed_output_dir = os.path.join(output_dir, "embedding.npy")
    address_output_dir = os.path.join(output_dir, "address.npy")

    np.save(embed_output_dir, seq_embedding_array)
    np.save(address_output_dir, address_array)

    print("Finish..")

if __name__ == '__main__':
    infer_embed()

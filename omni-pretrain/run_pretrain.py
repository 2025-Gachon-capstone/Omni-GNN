from config import args
from dataloader import InstacartDataloader
from modeling import InstacartBERT
from trainer import InstacartTrainer
from vocab import FreqVocab
import pickle as pkl
import os

def train():

    # prepare dataset
    vocab = FreqVocab()
    print("===========Load Sequence===========")
    with open(os.path.join(args.data_dir, f"dup/train_seq_{args.bizdate}.pkl"), "rb") as f:
        user2seq = pkl.load(f)

    print("number of target users:", len(user2seq))
    vocab.update(user2seq)                # product_id용 counter 업데이트
    vocab.generate_vocab()               # BERT-style vocab 생성


    # save vocab
    vocab_file_name = os.path.join(args.data_dir, f"{args.vocab_filename}.{args.bizdate}")
    print("token_size:", len(vocab.vocab_words))
    print("vocab pickle file:", vocab_file_name)
    with open(vocab_file_name, 'wb') as output_file:
        pkl.dump(vocab, output_file, protocol=2)

    # dataloader
    dataloader = InstacartDataloader(args, vocab, user2seq)
    train_loader = dataloader.get_train_loader()

    # model
    model = InstacartBERT(args)

    # trainer
    trainer = InstacartTrainer(args, vocab, model, train_loader)
    trainer.train()

if __name__ == '__main__':
    train()
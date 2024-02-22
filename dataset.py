import json

import numpy as np
import sentencepiece as spm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


def load_tokenizer(tokenizer_path):
    sp_chn = spm.SentencePieceProcessor()
    sp_chn.Load(f'{tokenizer_path}/chn.model')
    sp_eng = spm.SentencePieceProcessor()
    sp_eng.Load(f'{tokenizer_path}/eng.model')
    return sp_chn, sp_eng

class MTDataset(Dataset):
    def __init__(self, data_path, tokenizer_path="data"):
        self.out_en_sent, self.out_ch_sent = self.get_dataset(data_path, sort=True)
        self.sp_chn, self.sp_eng = load_tokenizer(tokenizer_path)
        self.PAD = self.sp_eng.pad_id()  # 0
        self.BOS = self.sp_eng.bos_id()  # 2
        self.EOS = self.sp_eng.eos_id()  # 3

    @staticmethod
    def len_argsort(seq):
        """传入一系列句子数据(分好词的列表形式)，按照句子长度排序后，返回排序后原来各句子在数据中的索引下标"""
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    def get_dataset(self, data_path, sort=False):
        """把中文和英文按照同样的顺序排序, 以英文句子长度排序的(句子下标)顺序为基准"""
        # 加载数据集
        dataset = json.load(open(data_path, 'r'))
        # 将中文和英文分别加载为两个列表
        out_en_sent = []
        out_ch_sent = []
        for idx, _ in enumerate(dataset):
            out_en_sent.append(dataset[idx][0])
            out_ch_sent.append(dataset[idx][1])
        # 如果要按长度排序
        if sort:
            sorted_index = self.len_argsort(out_en_sent)
            out_en_sent = [out_en_sent[i] for i in sorted_index]
            out_ch_sent = [out_ch_sent[i] for i in sorted_index]
        return out_en_sent, out_ch_sent

    def __getitem__(self, idx):
        # get 方法，返回一个句对
        eng_text = self.out_en_sent[idx]
        chn_text = self.out_ch_sent[idx]
        return [eng_text, chn_text]

    def __len__(self):
        return len(self.out_en_sent)

    def collate_fn(self, batch):
        # 变长序列的 collate_fn 方法，需要进行 padding
        # 形成列表
        src_text = [x[0] for x in batch]
        tgt_text = [x[1] for x in batch]
        # 进行 tokenizer，然后加上 BOS 和 EOS
        src_tokens = [[self.BOS] + self.sp_eng.EncodeAsIds(sent) + [self.EOS] for sent in src_text]
        tgt_tokens = [[self.BOS] + self.sp_chn.EncodeAsIds(sent) + [self.EOS] for sent in tgt_text]
        # 进行 padding
        batch_input = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_tokens],
                                   batch_first=True, padding_value=self.PAD)
        batch_target = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens],
                                    batch_first=True, padding_value=self.PAD)
        src_mask = (batch_input != self.PAD).unsqueeze(-2)
        label_mask = (batch_target != self.PAD).unsqueeze(-2)
        return {"input_ids": batch_input, "label_ids": batch_target, "attention_mask": src_mask, "label_mask": label_mask}
        # return {"input_ids": batch_input, "attention_mask": src_mask}
        
if __name__ == "__main__":
    dataset = MTDataset("data/wmt/dev.json")
    dataloader = DataLoader(dataset, shuffle=True, batch_size=4, collate_fn=dataset.collate_fn)
    for item in dataloader:
        print(item.keys())
        break
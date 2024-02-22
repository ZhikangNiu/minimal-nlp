import sentencepiece as spm

def train(input_file, vocab_size, model_name, model_type, character_coverage): # 使用 Sentence Piece 基于训练数据来训练一个分词器 # args: # input_file: 训练使用的数据 # vocab_size: 设定的词表大小 # model_name: 模型命名 # model_type: 模型类型，一般选择 bpe # character_coverage: 覆盖的字符范围，中文一类的表意文字一般0.995，英文一类的字母文字一般1 # 采用命令行的形式实现 
    input_argument = '--input=%s --model_prefix=%s --vocab_size=%s --model_type=%s --character_coverage=%s ''--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 ' 
    cmd = input_argument % (input_file, model_name, vocab_size, model_type, character_coverage) 
    spm.SentencePieceTrainer.Train(cmd)
    
en_input = 'wmt/corpus.en' 
en_vocab_size = 32000 
en_model_name = 'eng' 
en_model_type = 'bpe' 
en_character_coverage = 1 
train(en_input, en_vocab_size, en_model_name, en_model_type, en_character_coverage)

ch_input = 'wmt/corpus.ch' 
ch_vocab_size = 32000 
ch_model_name = 'chn' 
ch_model_type = 'bpe' 
ch_character_coverage = 0.9995 
train(ch_input, ch_vocab_size, ch_model_name, ch_model_type, ch_character_coverage)
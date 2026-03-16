from datasets import load_dataset
import sentencepiece as spe 

ds = load_dataset("roneneldan/TinyStories")

NUM_TRAIN_ROWS=2119719
NUM_VALID_ROWS=21990
'''
# Gets the text for each row in ds['train']
def sentence_iterator():
    for row in ds['train']:
        yield row['text']

spe.SentencePieceTrainer.train(
    sentence_iterator=sentence_iterator(),
    model_prefix='tinystories_tokenizer',
    vocab_size=5000,
    model_type='bpe', # BPE stands for byte pair encoder which splits the sentence into pairs
    character_coverage=1.0,
    max_sentence_length=10000,
    input_sentence_size=NUM_TRAIN_ROWS,
    shuffle_input_sentence=True
)'''

# Used to load the model
sp=spe.SentencePieceProcessor()
sp.load("tinystories_tokenizer.model")

with open("tinystories_tokenizer.vocab") as f:
    for i in range(100):
        print(f.readline())
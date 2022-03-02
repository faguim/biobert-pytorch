import torch

# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from transformers import BertTokenizer, BertConfig

# from keras.preprocessing.sequence import pad_sequences
# from sklearn.utils import shuffle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(str(n_gpu)+ ' '+ str(device) + ' Device available: '+ torch.cuda.get_device_name(0))

num_epochs = 4
train_batch_size = 1

torch.cuda.empty_cache()

from sentence_transformers import SentenceTransformer, models

# MODEL = '../models/NER/NCBI-disease'

MODEL = 'bert-base-cased'

OUTPUT_MODEL = 'output/NLI'
# model_save_path = './models/clinical_bert/finetuned/output2'

DATASET = '../datasets/NLI/snli_1.0/'


word_embedding_model = models.Transformer(MODEL)

tokens = ["dyspnea", "fagner", "jessica"]
word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
word_embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
# word_embedding_model.save('/content/drive/MyDrive/Colab Notebooks/models/clinical_bert/output')
word_embedding_model.save(OUTPUT_MODEL)




from sentence_transformers import InputExample
from torch.utils.data import DataLoader

# model = SentenceTransformer('distilbert-base-nli-mean-tokens')
train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
   InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# print('s')
# # Load clinicalBert model pre-downloaded
# import pdb
# import torch

# md = torch.load('../models/clinical_bert/finetuned/output/0_Transformer/pytorch_model.bin',map_location='cpu')
# # md = torch.load('../models/clinical_bert/pytorch_model.bin',map_location='cpu')

# for k in md:
# #     print(k)
# # modelos finetunados tão vindo sem o prefixo bert.
#     if (k == 'bert.embeddings.word_embeddings.weight' or k == 'embeddings.word_embeddings.weight'):
#         embeds = md[k]

# # vectors = []
# print(md)
# # for l in range(len(embeds)):
# #     vector = embeds[l]
# #     tsv_row = ''
# #     for m in range(len(vector)):
# #         tsv_row += str(vector[m].tolist()) + '\t'

# #     vectors.append(tsv_row)

# # print(len(vectors))

import csv, logging

# Convert the dataset to a DataLoader ready for training
logging.info("Read STSbenchmark train dataset")

sts_dataset_path = '../datasets/NLI/snli_1.0/'

print('Read NLIbenchmark train dataset')

train_samples = []
dev_samples = []
test_samples = []

with open(DATASET + 'snli_1.0_train.txt', "r") as f:
#     labels = f.readlines()
#     print(f.readlines())
    reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
#         print(row)
#         score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1

        gold_label = row['gold_label']  # Normalize score to range 0 ... 1
        float_gold_label = 0

        if gold_label == 'contradiction':
            float_gold_label = 0 / 3.0
        if gold_label == 'entailment':
            float_gold_label = 1 / 3.0
        if gold_label == 'neutral':
            float_gold_label = 2 / 3.0


#         print(gold_label)
        inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=float_gold_label)

#         if row['split'] == 'dev':
        train_samples.append(inp_example)
#         elif row['split'] == 'test':
#             test_samples.append(inp_example)
#         else:
#             train_samples.append(inp_example)


print(train_samples[0])
# with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
# reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)

# print(word_embedding_model.fc1(x).size())


import math

from sentence_transformers import losses

from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator



# model_save_path = 'output/training_stsbenchmark_continue_training-'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# model_save_path = './models/clinical_bert/finetuned/output2'

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=word_embedding_model)


# Development set: Measure correlation between cosine score and gold labels
logging.info("Read SNLIbenchmark dev dataset")
# evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')


# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
word_embedding_model.fit(train_objectives=[(train_dataloader, train_loss)],
#           evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=OUTPUT_MODEL)


##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

# model = SentenceTransformer(model_save_path)
# test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
# test_evaluator(model, output_path=model_save_path)

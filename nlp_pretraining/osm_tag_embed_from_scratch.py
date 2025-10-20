from collections import defaultdict
import gc
import pickle
import logging
from transformers import BertModel, BertConfig
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.models import Transformer, Pooling, Normalize
from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformerTrainer
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss, CachedMultipleNegativesRankingLoss, CachedMultipleNegativesSymmetricRankingLoss
from sentence_transformers.training_args import BatchSamplers
import os
from datasets import Dataset
from datasets import load_dataset, concatenate_datasets
import random
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='bert-144-osm-tags-embed-from_scratch.log', level=logging.INFO)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

mini_batch_size = 1024
batch_size = mini_batch_size * 5
epochs = 4
save_steps = 250


class SampleSimilarityEvaluator(SentenceEvaluator):
    def __init__(self, tags=None):
        if tags is None:
            self.tags = [
                """    
    cycleway:right=no, highway=tertiary, lanes=2, name=Washington Boulevard, oneway=yes, surface=asphalt
    """,
                """    
    cycleway:right=yes, highway=tertiary, lanes=2, name=Washington Boulevard, oneway=yes, surface=asphalt
    """,
                """    
    cycleway:right=no, highway=tertiary, lanes=3, name=Washington Boulevard, oneway=yes, surface=asphalt
    """,
                """    
    cycleway:right=no, highway=tertiary, lanes=2, name=Washington Boulevard, oneway=no, surface=asphalt
    """,
                """    
    cycleway:right=no, highway=tertiary, lanes=2, name=San Francisco Street, network:wikipedia: ro:Societatea de Transport București, operator:short: STB, oneway=yes, surface=asphalt
    """,
                """    
    cycleway:right=yes, highway=tertiary, lanes=4, name=Washington Boulevard, oneway=yes, surface=asphalt
    """,
                """
    addr:city=Detroit, addr:housenumber=1449, addr:postcode=48226, addr:state=MI, addr:street=Woodward Avenue, architect=Albert Kahn, building:levels=8, building=commercial, name=Woodward Bldg, source=Bing, year_of_construction=1915
    """,
                """
    cycleway:both=no, highway=tertiary, lanes:backward=3, lanes:forward=2, lanes=5, name=Bagley Avenue, surface=asphalt, tiger:cfcc=A41, tiger:county=Wayne, MI, tiger:name_base=Bagley, tiger:name_type=St, tiger:reviewed=no, tiger:zip_left=48226, tiger:zip_right=48226
    """,
                """
    restriction=no_left_turn, type=restriction
    """
            ]

    def __call__(self, model):
        embeddings = model.encode(self.tags)
        similarities = model.similarity(embeddings, embeddings)
        logger.info(similarities)
        return float(similarities[0, 4])


def main():

    model = SentenceTransformer(
        'models/bert-144-l6-reset-custom-tokenizer/')

    sample_evaluator = SampleSimilarityEvaluator()
    print("Sample similarities for untrained model: ")
    sample_evaluator(model)

    dataset_folder_path = "res/relevant_tags_pairs_dataset_parquet_sharded_20_rep"

    shard_files = list(os.scandir(dataset_folder_path))
    shard_files = [file for file in shard_files if
                   file.is_file() and os.fsdecode(file).endswith(".parquet")]

    dataset = load_dataset("parquet", data_files=[
                           file.path for file in shard_files])

    loss = CachedMultipleNegativesSymmetricRankingLoss(
        model, mini_batch_size=mini_batch_size)

    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"models/bert-144-osm-tags-embed-from_scratch",
        # Optional training parameters:
        num_train_epochs=epochs,
        eval_strategy="steps",
        eval_steps=save_steps,
        save_strategy="steps",
        save_steps=save_steps,
        logging_steps=20,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=1e-3,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        # losses that use "in-batch negatives" benefit from no duplicates
        batch_sampler=BatchSamplers.NO_DUPLICATES,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=loss,
        evaluator=sample_evaluator
    )

    trainer.train()


if __name__ == '__main__':
    main()

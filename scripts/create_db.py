"""
python scripts/create_db.py --path_to_data ./data/train/ --embed_model 'sentence-transformers/all-MiniLM-L6-v2'

Note: Takes ~15 mins to run on training dataset (200 notes) on Macbook with MPS for BGE-large
"""
from typing import List
from dotenv import load_dotenv
import torch
from utils.helpers import embed, load_model
load_dotenv()
from utils.data_loader import XMLDataLoader
import argparse
import chromadb
from tqdm import tqdm

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create a Chroma db for the given data')
    parser.add_argument('--path_to_data', type=str, default='./data/train/', help='Path to the data')
    parser.add_argument('--embed_model', type=str, default='BAAI/bge-large-en-v1.5', help='Name of the model to use')
    parser.add_argument('--is_delete_collection', action="store_true", default=False, help='If TRUE, clear out collection (if already exists).')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    embed_model: str = args.embed_model
    path_to_data: str = args.path_to_data
    is_delete_collection: bool = args.is_delete_collection
    
    # Setup Chroma db
    collection_name: str = embed_model.split("/")[-1]
    client = chromadb.PersistentClient(path="./data/chroma")
    if is_delete_collection and client.get_collection(collection_name) is not None:
        print(f"Deleting existing collection: `{collection_name}`...")
        client.delete_collection(collection_name)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Load data
    dataloader = XMLDataLoader(path_to_data)
    dataset = dataloader.load_data()
    print("# of patients:", len(dataset))

    # Load model
    model, tokenizer = load_model(embed_model)
    model_max_len: int = model.config.max_position_embeddings
    print("Device:", model.device)
    print("Chunk size:", model.config.max_position_embeddings)

    # Get embeddings
    for patient in tqdm(dataset):
        # For each note...
        for note_idx, note in enumerate(patient['ehr']):
            tokens: torch.Tensor = tokenizer(note, return_tensors="pt", add_special_tokens=False)['input_ids'].to(model.device)
            # For each chunk of size `model_max_len` tokens in `note`...
            for chunk_idx, chunk_start in enumerate(range(0, len(tokens[0]), model_max_len)):
                # do mean pooling to get the embedding of this chunk
                embedding: torch.Tensor = model(input_ids=tokens[:,chunk_start:chunk_start+model_max_len])[0][0].mean(dim=0).detach().cpu()
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
                collection.upsert(
                    embeddings=embedding.tolist(),
                    metadatas={
                        'patient_id' : patient['patient_id'], 
                        'note_idx' : note_idx,
                        'chunk_idx' : chunk_idx, # measured in terms of tokens
                        'chunk_start' : chunk_start, # measured in terms of tokens
                        'chunk_end' : chunk_start+model_max_len, # measured in terms of tokens
                        **patient['labels'],
                    },
                    documents=tokenizer.decode(tokens[0,chunk_start:chunk_start+model_max_len].tolist()),
                    ids=f"{patient['patient_id']}_{note_idx}_{chunk_idx}",
                )
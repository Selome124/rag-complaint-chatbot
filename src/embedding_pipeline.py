import pandas as pd
from pathlib import Path
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# Paths
DATA_PATH = Path("data/processed/filtered_complaints.csv")
VECTOR_STORE_PATH = Path("vector_store/chroma_complaints")

VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print("Loaded dataset:", df.shape)

SAMPLE_SIZE = 12000

# Calculate samples per product (proportional)
product_counts = df["Product"].value_counts(normalize=True)
samples_per_product = (product_counts * SAMPLE_SIZE).astype(int)

sampled_dfs = []

for product, n_samples in samples_per_product.items():
    product_df = df[df["Product"] == product]
    sampled_dfs.append(
        product_df.sample(
            n=min(n_samples, len(product_df)),
            random_state=42
        )
    )

df_sampled = pd.concat(sampled_dfs).reset_index(drop=True)

print("Stratified sample shape:", df_sampled.shape)
print(df_sampled["Product"].value_counts())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

texts = []
metadatas = []

for _, row in tqdm(df_sampled.iterrows(), total=len(df_sampled)):
    chunks = text_splitter.split_text(row["clean_narrative"])
    
    for i, chunk in enumerate(chunks):
        texts.append(chunk)
        metadatas.append({
            "complaint_id": row.get("Complaint ID", None),
            "product": row["Product"],
            "issue": row.get("Issue", None),
            "company": row.get("Company", None),
            "chunk_index": i,
            "total_chunks": len(chunks)
        })

print(f"Total chunks created: {len(texts)}")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embedding_model,
    metadatas=metadatas,
    persist_directory=str(VECTOR_STORE_PATH)
)

vectorstore.persist()
print("Vector store successfully saved.")

if __name__ == "__main__":
    print("Embedding pipeline completed successfully.")



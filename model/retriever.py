from datasets import load_dataset
from config import CONFIG
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util


class Retriever:
    def __init__(self):
        self.corpus = None
        self.bm25 = None
        self.model = None
        self.chunk_embeddings = None

    def load_and_prepare_dataset(self):
        dataset = load_dataset(CONFIG['DATASET'])
        dataset = dataset['train'].select(range(CONFIG['MAX_NUM_OF_RECORDS']))
        dataset = dataset.map(lambda x: {'chunks': self.chunk_text(x['abstract'])})
        self.corpus = [chunk for chunks in dataset["chunks"] for chunk in chunks]

    def prepare_bm25(self):
        tokenized_corpus = [doc.split(" ") for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def compute_embeddings(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunk_embeddings = self.model.encode(self.corpus, convert_to_tensor=True)

    def chunk_text(self, text, chunk_size=CONFIG['CHUNK_SIZE']):
        words = text.split()
        return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def retrieve_documents_bm25(self, query):
        tokenized_query = query.split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        top_docs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:CONFIG['TOP_DOCS']]
        return [self.corpus[i] for i in top_docs]

    def retrieve_documents_semantic(self, query):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embedding, self.chunk_embeddings)[0]
        top_chunks = scores.topk(CONFIG['TOP_DOCS']).indices
        return [self.corpus[i] for i in top_chunks]
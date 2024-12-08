import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    "DATASET": "aalksii/ml-arxiv-papers",
    "MAX_NUM_OF_RECORDS": 1000,
    "TEXTAREA_HEIGHT": 200,
    "CHUNK_SIZE": 200,
    "OPENAI_ENGINE": "gpt-4o-mini",
    "MAX_TOKENS": 500,
    "TOP_DOCS": 3
}
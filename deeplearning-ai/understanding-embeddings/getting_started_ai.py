from utils import authenticate
from vertexai.language_models import TextEmbeddingModel
import vertexai
from sklearn.metrics.pairwise import cosine_similarity

REGION = 'us-central1'
credentials, PROJECT_ID = authenticate() # Get credentials and project ID


vertexai.init(project = PROJECT_ID, 
              location = REGION, 
              credentials = credentials)

embedding_model = TextEmbeddingModel.from_pretrained(
    "textembedding-gecko@001")

embedding = embedding_model.get_embeddings(
    ["What is the meaning of life?"])



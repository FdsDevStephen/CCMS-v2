import torch
from transformers import AutoTokenizer, AutoModel


class Embedder:

    def __init__(self, model_name="BAAI/bge-m3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[Embedder] Loading {model_name}")
        print(f"[Embedder] Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()

        return torch.sum(
            token_embeddings * input_mask_expanded,
            dim=1
        ) / torch.clamp(
            input_mask_expanded.sum(dim=1),
            min=1e-9
        )

    def embed_documents(self, texts):
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            output = self.model(**encoded)

        embeddings = self._mean_pooling(
            output,
            encoded["attention_mask"]
        )

        embeddings = torch.nn.functional.normalize(
            embeddings,
            p=2,
            dim=1
        )

        return embeddings.cpu().numpy()

    def embed_query(self, text):
        return self.embed_documents([text])[0]
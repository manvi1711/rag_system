import json
import numpy as np
import boto3

class TitanEmbedder:
    def __init__(self, model_id, region="us-east-1"):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id

    def embed_one(self, text: str):
        body = json.dumps({"inputText": text})
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json"
        )
        output = json.loads(response["body"].read())
        return output["embedding"]

    def embed_many(self, texts):
        return [self.embed_one(t) for t in texts]

    # 🔥 REQUIRED FOR LANGCHAIN COMPATIBILITY
    def embed_documents(self, texts):
        return self.embed_many(texts)

    def embed_query(self, text):
        return self.embed_one(text)

class Endee:
    def __init__(self, host="localhost", port=8082, api_key=None):
        self.host = host
        self.port = port
        self.api_key = api_key

    def create_index(self, name, dimension, precision):
        # Index creation logic
        pass

    def upsert(self, index_name, vectors):
        # Vector upload logic
        pass

    def search(self, index_name, query_vector, top_k=5):
        # Search logic
        return []
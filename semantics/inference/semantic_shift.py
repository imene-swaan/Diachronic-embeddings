import numpy as np
from typing import List, Optional




class SemanticShift:
    def __init__(
            self,
            embeddings: List[np.ndarray],
            ) -> None:
        
        self.embeddings = embeddings


    def get_shift(self) -> List[np.ndarray]:
        shift = []
        for i in range(1, len(self.embeddings)):
            shift.append(self.embeddings[i] - self.embeddings[i-1])
        
        return shift
    


    def get_pair_similarities(
            self, 
            sort_desc: bool = True, 
            labels: Optional[List[str]] = None
            ) -> List[np.ndarray]:
        
        pair_similarities = []

        for i in range(len(self.embeddings)):
            for j in range(i+1, len(self.embeddings)):
                pair_similarity = self._cosine_similarity(self.embeddings[i], self.embeddings[j])
                pair = (labels[i], labels[j]) if labels else (i, j)
                pair_similarities.append((pair, pair_similarity))

        sorted_pairs = sorted(pair_similarities, key=lambda x: x[1], reverse= sort_desc)
        return sorted_pairs


    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))






















if __name__ == "__main__":
    pass

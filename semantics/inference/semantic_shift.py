import numpy as np
from typing import List, Optional, Literal
import ruptures as rpt



class SemanticShift:
    def __init__(
            self,
            embeddings: List[np.ndarray],
            ) -> None:
        
        self.embeddings = embeddings

        
    
    def ChangePointDetection(
            self, 
            cost_func: Optional[Literal["l1", "l2", "rbf", "linear", "normal"]]= "rbf",
            min_size: Optional[int] = 2,
            jump: Optional[int] = 5,
            penalty: Optional[float] = 0.5,
            ref: Optional[int] = -1,
            labels: Optional[List[str]] = None
            ) -> List[np.ndarray]:
        
        ts = [x[1] for x in self.get_ref_shift(ref=ref, labels=labels)]
        y = np.array(ts)

        algo = rpt.Pelt(model=cost_func, jump=jump, min_size=min_size)
        algo.fit(y)
        breaks = algo.predict(pen=penalty)
        return [labels[i-1] if labels else i for i in breaks]
       
    
    
    def get_ref_shift(self, ref: int = -1, labels: Optional[List[str]] = None, to_score: bool= True) -> List[np.ndarray]:
        shift = []
        for i in range(len(self.embeddings)):
            pair_similarity = self._cosine_similarity(self.embeddings[ref], self.embeddings[i])
            if to_score:
                pair_similarity = self._convert_similarity_to_score(pair_similarity)
            pair = (labels[ref], labels[i]) if labels else (ref, i)
            shift.append((pair, pair_similarity))
        return shift

    
    def get_sequence_shift(self, labels: Optional[List[str]] = None, to_score: bool= True) -> List[np.ndarray]:
        shift = []

        for i in range(len(self.embeddings) - 1):
            pair_similarity = self._cosine_similarity(self.embeddings[i], self.embeddings[i+1])
            if to_score:
                pair_similarity = self._convert_similarity_to_score(pair_similarity)
            pair = (labels[i], labels[i+1]) if labels else (i, i+1)
            shift.append((pair, pair_similarity))
        return shift


    def get_pair_shift(
            self, 
            sort: bool = True,
            sort_desc: bool = True, 
            labels: Optional[List[str]] = None,
            to_score: bool= True,
            top_n: Optional[int] = None,
            max_time: Optional[int] = None,
            min_time: Optional[int] = None,
            ) -> List[np.ndarray]:
        
        if max_time is not None and labels is None:
            raise ValueError("Labels must be provided to use max_time")
        
        pair_similarities = []


        for i in range(len(self.embeddings)):
            for j in range(i+1, len(self.embeddings)):
                if max_time is not None and labels[j] - labels[i] > max_time:
                    break
                if min_time is not None and labels[j] - labels[i] < min_time:
                    continue
                
                pair_similarity = self._cosine_similarity(self.embeddings[i], self.embeddings[j])
                if to_score:
                    pair_similarity = self._convert_similarity_to_score(pair_similarity)
                pair = (labels[i], labels[j]) if labels else (i, j)
                pair_similarities.append((pair, pair_similarity))
            
        


        if sort:
            pair_similarities = sorted(pair_similarities, key=lambda x: x[1], reverse= sort_desc)
        
        if top_n:
            pair_similarities = pair_similarities[:top_n]
        
        return pair_similarities


    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


    def _convert_similarity_to_score(self, similarity: float) -> float:
        return 1 - (similarity + 1) / 2


















if __name__ == "__main__":
    pass

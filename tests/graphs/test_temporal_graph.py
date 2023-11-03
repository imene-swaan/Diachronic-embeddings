from unittest.mock import patch, MagicMock
from typing import List
import unittest
from src.graphs.temporal_graph import Nodes

# Your test class
class TestNodes(unittest.TestCase):

    def setUp(self):
        # Mock dependencies
        self.mock_word2vec_inference = MagicMock()
        self.mock_roberta_inference = MagicMock()
        self.mock_bert_inference = MagicMock()

        # Setup the mock for Word2VecInference
        with patch('src.feature_extraction.word2vec.Word2VecInference') as mock_word2vec_class:
            mock_word2vec_class.return_value = self.mock_word2vec_inference
            # Suppose Word2VecInference has a method 'get_top_k_words' that we need to mock.
            self.mock_word2vec_inference.get_top_k_words.return_value = (['context1', 'context2'], None)

            # Similarly setup mocks for RobertaInference and BertInference
            with patch('src.feature_extraction.roberta.RobertaInference') as mock_roberta_class:
                mock_roberta_class.return_value = self.mock_roberta_inference
                self.mock_roberta_inference.get_top_k_words.return_value = ['similar1', 'similar2']

                # Now instantiate your Nodes class with mocked dependencies
                self.nodes = Nodes(
                        target_word='test',
                        dataset=['this is a test', 'this is another test'],
                        level=1,
                        k=2,
                        c=2,
                        word2vec_model_path='path/to/mock/model',
                        mlm_model_path='path/to/mock/mlm_model',
                        mlm_model_type='roberta'  # Assuming we are mocking RobertaInference
                    )

    def test_get_similar_nodes(self):
        # Test that the _get_similar_nodes method returns the expected list
        # Assuming mock_roberta_inference.get_top_k_words returns ['similar1', 'similar2']
        result = self.nodes._get_similar_nodes('test')
        self.assertEqual(result, ['similar1', 'similar2'])
        self.mock_roberta_inference.get_top_k_words.assert_called()

    def test_get_context_nodes(self):
        # Test that the _get_context_nodes method returns the expected list
        # Assuming mock_word2vec_inference.get_top_k_words returns (['context1', 'context2'], None)
        result = self.nodes._get_context_nodes('test')
        self.assertEqual(result, ['context1', 'context2'])
        self.mock_word2vec_inference.get_top_k_words.assert_called()

    def test_get_nodes(self):
        # Test the get_nodes method. This also implicitly tests that the internal methods
        # _get_similar_nodes and _get_context_nodes are being called correctly.
        expected_result = {
            'target_node': 'test',
            'similar_nodes': [['similar1', 'similar2']],
            'context_nodes': [['context1', 'context2']]
        }

        result = self.nodes.get_nodes()
        self.assertEqual(result, expected_result)

        # Ensure that internal methods are being called
        self.mock_roberta_inference.get_top_k_words.assert_called()
        self.mock_word2vec_inference.get_top_k_words.assert_called()


if __name__ == '__main__':
    unittest.main()

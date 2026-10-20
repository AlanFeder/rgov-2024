from pathlib import Path
import sys
import unittest
from unittest.mock import patch, Mock
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b1_rag_fns import b1_all_rag_fns as rag

class RetrievalTests(unittest.TestCase):
    def test_http_embedding_contract_and_ranking(self):
        response = Mock(status_code=200)
        response.json.return_value = {"data": [{"embedding": [1.0, 0.0]}]}
        with patch.object(rag.requests, "post", return_value=response) as post:
            query = rag.do_1_embed("question", "offline-placeholder")
        self.assertEqual("Bearer offline-placeholder", post.call_args.kwargs["headers"]["Authorization"])
        results = rag.do_sort(query, np.array([[0., 1.], [1., 0.]]), ["other", "match"])
        self.assertEqual("match", results[0]["id0"])
        self.assertEqual(1., results[0]["score"])

    def test_streamlit_starts_without_network_calls(self):
        from streamlit.testing.v1 import AppTest
        root = Path(__file__).resolve().parents[1]
        app = AppTest.from_file(str(root / "c1_apps/streamlit_served1.py"))
        app.secrets["OPENAI_API_KEY"] = "offline-placeholder"
        with patch("requests.sessions.Session.request", side_effect=AssertionError("Network disabled")):
            app.run(timeout=30)
        self.assertEqual([], [error.message for error in app.exception])

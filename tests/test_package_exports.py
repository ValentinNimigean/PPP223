from ingest import ChunkRecord, CodeChunk, Loader, LoaderProtocol
from model import ChatInferenceBackend, LocalInferenceEngine, PeftAdapterInferenceEngine, SLMAgent
from rag import HybridRetriever, RetrieverProtocol
from ui import build_parser


def test_direct_module_imports_still_work():
    from ingest.loader import Loader as DirectLoader
    from ingest.metadata import CodeChunk as DirectCodeChunk
    from model.agent import SLMAgent as DirectAgent
    from rag.retriever import HybridRetriever as DirectRetriever

    assert DirectLoader is Loader
    assert DirectCodeChunk is CodeChunk
    assert DirectRetriever is HybridRetriever
    assert DirectAgent is SLMAgent


def test_package_level_exports_are_available():
    assert Loader.__name__ == "Loader"
    assert CodeChunk.__name__ == "CodeChunk"
    assert HybridRetriever.__name__ == "HybridRetriever"
    assert SLMAgent.__name__ == "SLMAgent"
    assert LoaderProtocol.__name__ == "LoaderProtocol"
    assert RetrieverProtocol.__name__ == "RetrieverProtocol"
    assert ChunkRecord.__name__ == "ChunkRecord"
    assert ChatInferenceBackend.__name__ == "ChatInferenceBackend"
    assert LocalInferenceEngine.__name__ == "LocalInferenceEngine"
    assert PeftAdapterInferenceEngine.__name__ == "PeftAdapterInferenceEngine"


def test_ui_package_reexports_cli_parser():
    parser = build_parser()
    args = parser.parse_args(["--repo", ".", "--question", "What class handles hybrid vector search?"])
    assert args.repo == "."
    assert args.question == "What class handles hybrid vector search?"

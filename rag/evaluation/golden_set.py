import json
import pathlib
from dataclasses import dataclass

DEFAULT_GOLDEN_QUERIES_PATH = pathlib.Path(__file__).resolve().parents[2] / "eval" / "golden_queries.json"


@dataclass
class GoldenQuery:
    query: str
    expected_source_substring: str | None
    expected_sufficient: bool


def load_golden_queries(path: pathlib.Path | str = DEFAULT_GOLDEN_QUERIES_PATH) -> list[GoldenQuery]:
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return [
        GoldenQuery(
            query=item["query"],
            expected_source_substring=item["expected_source_substring"],
            expected_sufficient=item["expected_sufficient"],
        )
        for item in data
    ]

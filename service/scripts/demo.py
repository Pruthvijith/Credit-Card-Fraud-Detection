"""Build, register, promote, score and verify recovery in a fresh directory."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fraud_platform.api import ModelStore
from fraud_platform.contracts import vector
from fraud_platform.registry import checked_release, promote, register
from fraud_platform.synthetic import generate


def main() -> None:
    with TemporaryDirectory() as folder:
        root = Path(folder)
        first = register(root, 4000, 42)
        candidate = register(root, 4000, 43)
        for version in (first, candidate, first):
            promote(root, version)
        location = checked_release(root, (root/'active').read_text().strip())
        loaded = ModelStore()
        loaded.load(location)
        examples, _ = generate(100)
        score = float(loaded.model.predict_proba([vector(examples[0])])[0, 1])
        result = {k: v for k, v in loaded.manifest.items() if k.startswith('test_')}
        assert 0 <= score <= 1 and first == location.name
        print(json.dumps(dict(restored=first, sample_score=score, **result), indent=2))


if __name__ == '__main__':
    main()

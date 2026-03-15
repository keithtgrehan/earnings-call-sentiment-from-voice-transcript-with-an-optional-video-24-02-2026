from __future__ import annotations

import json

from earnings_call_sentiment.visual.runtime import multimodal_runtime_status


def main() -> None:
    print(json.dumps(multimodal_runtime_status(), indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
from pathlib import Path

from earnings_call_sentiment.web_backend import create_review_app

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

def create_app():
    return create_review_app(
        template_dir=APP_DIR / 'templates',
        static_dir=APP_DIR / 'static',
        repo_root=REPO_ROOT,
        ui_meta={
            'title': 'Earnings Call Signal Engine',
            'eyebrow': 'Transcript-first AI review',
            'lede': (
                'Transcript-first AI tool for extracting structured signals from earnings call audio '
                'and video sources using NLP.'
            ),
            'variant': 'site',
        },
    )


app = create_app()


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.getenv('PORT', '7861')), debug=False)

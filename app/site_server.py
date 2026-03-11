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
            'title': 'Signal Desk',
            'eyebrow': 'Deterministic earnings-call review',
            'lede': (
                'Transcript-first review surface for guidance shifts, tone changes, and auditable evidence. '
                'This is the cleaner long-term website shell built on the same local deterministic backend.'
            ),
            'variant': 'site',
        },
    )


app = create_app()


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.getenv('PORT', '7861')), debug=False)

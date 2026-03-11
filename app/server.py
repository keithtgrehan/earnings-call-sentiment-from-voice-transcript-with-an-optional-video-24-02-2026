from __future__ import annotations

import os
from pathlib import Path

from earnings_call_sentiment.web_backend import create_review_app

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

def create_app():
    return create_review_app(
        template_dir=APP_DIR / 'backup_templates',
        static_dir=APP_DIR / 'backup_static',
        repo_root=REPO_ROOT,
        ui_meta={
            'title': 'Earnings Call Review Lab',
            'eyebrow': 'Backup review console',
            'lede': (
                'Stable fallback interface for the deterministic earnings-call review workflow. '
                'Use this if you want the original panel-based layout while the newer website shell evolves.'
            ),
            'variant': 'backup',
        },
    )


app = create_app()


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.getenv('PORT', '7860')), debug=False)

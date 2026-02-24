import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Earnings call sentiment pipeline (YouTube URL input).")
    p.add_argument("--youtube-url", required=True)
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--out-dir", default="outputs")
    args = p.parse_args()

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("OK - CLI wired.")
    print("youtube_url:", args.youtube_url)
    print("cache_dir:", args.cache_dir)
    print("out_dir:", args.out_dir)

if __name__ == "__main__":
    main()

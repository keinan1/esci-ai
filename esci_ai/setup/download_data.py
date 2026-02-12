from pathlib import Path

import requests


def download_data(urls: list[str], file_paths: list[Path]) -> None:
    for url, file_path in zip(urls, file_paths):
        print(f"Downloading {url}")
        response = requests.get(url)
        response.raise_for_status()
        file_path.write_bytes(response.content)
        print(f"Finished downloading. Saved to {file_path}")


if __name__ == "__main__":
    project_dir = Path(__file__).parent.parent.parent
    data_dir = project_dir / "data"

    urls = [
        # note: first two datasets use git-lfs, so raw content sits behind "media.githubusercontent.media/..."
        "https://media.githubusercontent.com/media/amazon-science/esci-data/refs/heads/main/shopping_queries_dataset/shopping_queries_dataset_examples.parquet",
        "https://media.githubusercontent.com/media/amazon-science/esci-data/refs/heads/main/shopping_queries_dataset/shopping_queries_dataset_products.parquet",
        "https://raw.githubusercontent.com/amazon-science/esci-data/refs/heads/main/shopping_queries_dataset/shopping_queries_dataset_sources.csv",
    ]
    file_paths = [
        data_dir / "raw" / url.split("shopping_queries_dataset_")[-1] for url in urls
    ]

    download_data(urls, file_paths)

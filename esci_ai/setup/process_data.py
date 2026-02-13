from pathlib import Path

import polars as pl


def process_data(data_dir: Path) -> None:

    # load datasets
    examples = pl.read_parquet(data_dir / "raw" / "examples.parquet")
    products = pl.read_parquet(data_dir / "raw" / "products.parquet")

    # join and save examples-products
    examples_products = examples.join(
        products, on=["product_locale", "product_id"], how="left"
    )
    examples_products.write_parquet(
        data_dir / "processed" / "examples_products.parquet"
    )

    # filter examples-products-subset
    queries_of_interest = [
        "aa batteries 100 pack",
        "kodak photo paper 8.5 x 11 glossy",
        "dewalt 8v max cordless screwdriver kit, gyroscopic",
    ]
    examples_products_subset = examples_products.filter(
        pl.col("esci_label") == "E",
        pl.col("query").is_in(queries_of_interest),
    )
    # check queries of interest are identified
    examples_products_subset_queries = examples_products_subset.group_by(
        ["esci_label", "query"]
    ).agg(
        count=pl.len(),
    )

    assert len(examples_products_subset_queries) == len(queries_of_interest)
    print(examples_products_subset_queries)

    # save examples-products-subset
    examples_products_subset.write_parquet(
        data_dir / "processed" / "examples_products_subset.parquet"
    )

    print(f"Finished. Processed data saved to {data_dir / 'processed'}")


if __name__ == "__main__":
    project_dir = Path(__file__).parent.parent.parent
    data_dir = project_dir / "data"

    process_data(data_dir)

import marimo

__generated_with = "0.19.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ESCI-AI Walkthrough

    ## Problem

    The ESCI dataset consists of a set of query-product pairs.
    - Items labelled "E" are deemed "relevant for the query and satisfies __all__ the query specifications."
    - However, on inspection, some "E" items do not satisfy all the query specifications.

    The task is to identify examples where "E" is misapplied for three specific queries (given below). Specifically:
    - Identify when the query is incorrect for the product.
    - Reformulate the query so that it accurately reflects the product.

    ## Plan and Constraints

    While the task is focused on a small subset (less than 50 observations) of the original dataset, the solution will only be of practical use if it can be used on the entire set of ~2 million "E" queries to identify the mismatches.

    For example, the three outputs -- the product, the close-but-incorrect query, and the corrected query -- can be used to finetune an embedding model to move exact matches closer in the embedding space.

    The problem breaks into two different kinds of tasks:

    - [Task 1]: Binary classification - identify when the query is incorrect
    - [Task 2]: Text Generation - reformulate the query

    The first task can in theory be performed efficiently on 2 million examples if we trained a BERT model on query-product pairs, predicting match/no-match. Since we don't have those labels, we'd use the more expensive LLM approach to generate enough balanced labels (~1000) to train the BERT model.

    The second task requires text generation, so we'll assume LLMs will be required even in the scaled solution. However, the number of cases will be relatively small, since we only have to do this for the mismatched pairs.

    The only goal for this task is to get a working solution for a tiny subset of examples, however, we'll build with the points above in mind. In concrete terms:

    - Aim to use a local LLM, as small as possible to get accurate results. We'll start with [Ministral-3-8b-Instruct](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512) as a baseline with the option to move up to the 14B model, or even down to the 3B model depending on results.
    - Keep the implementation modular by separating out the classification and the text generation tasks. This will allow use of the first component just to scale up (generate labels to train a classifier) and the second to be reused in the scaled solution. It waill also allow using different LLM models if needed between the two tasks.

    ## Development Approach

    We'll use the following development process:

    - Step 1: build quick prototype
    - Step 2: perform error analysis on traces
    - Step 3: create automated evals on results
    - Step 4: iterate until all examples are corrrect

    ## Tooling

    - `uv` for package management
    - `ollama` for local model inference
    - `pydantic` for structured outputs
    - `pydantic-ai` for lightweight llm orchestration
    - `pydantic-logfire` for llm observability
    - `pydantic-evals` to create and run automated evals

    ## Setup

    - Clone the repo

    - If not already installed:
        - Install `uv`

        ```shell
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```

        - Install `ollama` and download the model

        ```shell
        curl -fsSL https://ollama.com/install.sh | sh
        ollama pull ministral-3:8b # ollama pull gpt-oss:latest
        ```

    - Create a `.env` file (optional: only needed for pydantic-logfire integration, ask me for a key)

        ```shell
        touch .env
        echo "PYDANTIC_LOGFIRE=<...project-api-key...>"
        ```

    - Download and process the esci data:

        ```shell
        uv run esci_ai/setup/download_data.py
        uv run esci_ai/setup/process_data.py
        ```

       The processed data will be saved as parquet files in `data/processed/`.


    - A bit self-referential: to run or edit this Marimo notebook:

        ```shell
        uv run marimo run esci_ai/walkthrough.py # read only
        uv run marimo edit esci_ai/walkthrough.py # edit
        ```
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import polars as pl

    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"

    examples_products = pl.read_parquet(
        data_dir / "processed" / "examples_products.parquet"
    )

    examples_products.glimpse()

    examples_products.group_by("esci_label").agg(
        count=pl.count(),
        unique_queries=pl.col("query_id").unique().count(),
        unique_products=pl.col("product_id").unique().count(),
    )
    return data_dir, pl


@app.cell
def _(data_dir, pl):
    products = pl.read_parquet(
        data_dir / "raw" / "products.parquet"
    )
    products
    return


@app.cell
def _(data_dir, pl):
    examples = pl.read_parquet(
        data_dir / "raw" / "examples.parquet"
    )
    examples
    return


@app.cell
def _(data_dir, pl):
    df = pl.read_parquet(
        data_dir / "processed" / "examples_products_subset.parquet"
    )

    df
    return


@app.cell
def _():
    # by manual inspection alone, these are at least some of the incorrect observations (7/24 error rate)

    incorrect_example_ids = [
        # batteries
        142660,  # 60 count
        142666,  # AAA
        # drills
        660823,  # no mention of gyroscopic
        660827,  # no mention of gyroscopic
        660840,  # charger only
        # paper
        1163629,  # matte
        1163641,  # matte
    ]
    return


app._unparsable_cell(
    r"""
    from pydantic import BaseModel


    class QueryInfo(BaseModel):
        query_id: int
        query: str

    class ProductInfo(BaseModel)
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()

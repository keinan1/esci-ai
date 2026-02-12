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

    While the task is focused on a small subset of the original dataset, the solution will only be of practical use if it can be used on the entire set of millions (?) of "E" queries to identify the mismatches. For example, the three outputs -- the product, the close-but-incorrect query, and the corrected query -- can be used to finetune an embedding model to move exact matches closer in the embedding space.

    The problem breaks into two different kinds of tasks:

    - [Task 1]: Binary classification - identify when the query is incorrect
    - [Task 2]: Text Generation - reformulate the query

    The first task can in theory be done cheaply on millions of observations. With enough examples we could train a BERT model on query-product pairs, predicting match/no-match. And to bootstrap into this, we'd use the more expensive LLM approach to generate enough balanced labels (~1000) to train the BERT model.

    The second task requires text generation, so we'll assume LLMs will be required even in the scaled solution. However, the number of cases will be relatively small, since we only have to do this for the mismatched pairs.

    The only goal for this task is to get a working solution for a tiny subset of examples, however, we'll build with the points above in mind. In concrete terms:

    - Aim to use a local LLM, as small as possible to get accurate results. We'll start with GPT-OSS-20B.
    - Keep the implementation modular by separating out the classification and the text generation tasks. This will allow use of the first component just to scale up (generate labels to train a classifier) and the second to be reused in the scaled solution.

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

        - Install `ollama` and download `gpt-oss`

        ```shell
        curl -fsSL https://ollama.com/install.sh | sh
        ollama pull gpt-oss:latest
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
    return


if __name__ == "__main__":
    app.run()

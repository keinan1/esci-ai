# ESCI AI Project

## Problem

The ESCI dataset consists of a set of query-product pairs.
- Items labelled "E" are deemed "relevant for the query and satisfies __all__ the query specifications."
- However, on inspection, some "E" items do not satisfy all the query specifications.

The task is to identify examples where "E" is misapplied for three specific queries (given below). Specifically:
- Identify when the query is incorrect for the product.
- Reformulate the query so that it accurately reflects the product.

## Plan and Constraints

While the task is focused on a small subset of the original dataset (less than 20 observations), the solution will only be of practical use if it can be used on the entire set of ~2 million "E" queries to identify the mismatches.

For example, the three outputs -- the product, the close-but-incorrect query, and the corrected query -- can be used to finetune an embedding model to move exact matches closer in the embedding space.

The problem breaks into two different kinds of tasks:

- [Task 1]: Binary classification - identify when the query is incorrect
- [Task 2]: Text Generation - reformulate the query

The first task can in theory be performed efficiently on 2 million examples if we trained a BERT model on query-product pairs, predicting match/no-match. Since we don't have those labels, we'd use the more expensive LLM approach to generate enough balanced labels (~1000) to train the BERT model.

The second task requires text generation, so we'll assume LLMs will be required even in the scaled solution. However, the number of cases will be relatively small, since we only have to do this for the mismatched pairs.

The only goal for this task is to get a working solution for subset of 20 examples. However, we'll build with the points above in mind. In concrete terms:

- Aim to use a local LLM, as small as possible to get accurate results. We'll start with [Ministral-3-3b-Instruct](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512) as a baseline, with the option to move up to [Ministral-3-8b-Instruct](https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512), [Ministral-3-14b-Instruct](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512), and finally [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b).
- Keep the implementation modular by separating out the classification and the text generation tasks. This will allow use of the first component just to scale up (generate labels to train a classifier) and the second to be reused in the scaled solution. It waill also allow using different LLM models if needed between the two tasks.

## Development Approach

We'll use the following development process:

- Step 1: Build quick prototype with evals
- Step 2: Perform error analysis on traces
- Step 3: Create axial codes and improve prompts
- Step 4: Iterate until all examples are corrrect

## Tooling

See the `pyproject.toml` for full details. The main dependencies will be:

- `uv` for package management
- `ollama` for local model inference
- `pydantic` for structured outputs
- `pydantic-ai` for lightweight llm orchestration
- `pydantic-logfire` for llm observability (optional, not needed to run)

## Setup

- Clone the repo

- If not already installed:
    - Install `uv`

    ```shell
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    - Install `ollama` and download the required model(s)

    ```shell
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull ollama pull gpt-oss:latest
    ```
    Note in testing we also use `ministral-3:3b`, `ministral-3:8b`, and  `ministral-3:14b`. 

- Create a `.env` file 

    ```shell
    touch .env
    echo "OLLAMA_BASE_URL=http://localhost:11434/v1" >> .env
    echo "OLLAMA_NUM_PARALLEL=2" >> .env
    echo "LOGFIRE_TOKEN=pylf..." >> .env # optional, only needed for logfire observability
    ```

- Download and process the esci data:

    ```shell
    uv run esci_ai/setup/download_data.py
    uv run esci_ai/setup/process_data.py
    ```

   The processed data will be saved as parquet files in `data/processed/`.

## Development Process

The workflow leading to the final solution is illustrated in the `notebooks/` directory.

### Step 1 - First Pass Solution

In the first pass solution we define the data models, prompts, and performance metrics. We run inference on the smallest model, `ministral-3:3b`, and record the traces and the performance metrics for the twenty examples (accuracy, precision, recall). You can run the notebook with:

```shell
uv run marimo edit notebooks/01_first_pass_solution.py
```

### Step 2 - Error Analysis

The first pass on `ministral-3:3b` performed poorly: accuracy is 0.67, with 1 false positive (i.e., llm categorized non-match as a match) and 7 false negatives (i.e. llm categorized match as non-match)

We collect the traces for all 8 incorrect inferences and perform a manual error analysis. For each trace, we manually record an observation of what went wrong, and a possible fix. See `notebooks/02_first_pass_error_analysis.md`.

### Step 3 - Axial Coding & Prompt Improvement

Here we use Claude to categorize our manual error analysis into proper axial codes. We then ask it to suggest minimal viable prompt improvements from the axial codes and our prior fix suggestions. See `notebooks/03_first_pass_axial_codes.md`.

### Step 4 - Second Pass Solution

We update the prompts and re-run the experiment. Performance improves: using `ministral-3:3b`, accuracy improves to ..., with only 1 false positive. Using the next most capable model, `ministral-3:8b`, accuracy is ... .

Run the notebook with:

```shell
uv run marimo edit notebooks/02_secon_pass_solution.py
```

## Final Implementation

The final python implementation is in `esci_ai/`. We organize the modules into proper files and classes. The script can be run with:

```

```

Results and performance metrics are written to the `data/results` directory.

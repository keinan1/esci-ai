import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import logfire
import polars as pl
from dotenv import find_dotenv, load_dotenv
from pydantic_ai import AgentRunResult

from esci_ai.agents import (
    create_classifier_agent,
    create_classifier_prompt,
    create_queryfix_agent,
    create_queryfix_prompt,
)
from esci_ai.models import (
    MatchClassification,
    ProductInfo,
    QueryFix,
    QueryInfo,
    QueryProductExample,
    QueryProductMatch,
)
from esci_ai.performance import get_classifier_performance

logger = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"

DEFAULT_DATASET_PATH = DATA_DIR / "processed" / "examples_products_subset.parquet"
# DEFAULT_DATASET_PATH = DATA_DIR / "processed" / "examples_products_random_1000.parquet"

# DEFAULT_MODEL = "ollama:ministral-3:3b"
DEFAULT_MODEL = "ollama:ministral-3:8b"
# DEFAULT_MODEL = "ollama:ministral-3:14b"
# DEFAULT_MODEL = "ollama:gpt-oss:20b"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


def setup_observability():
    logfire.configure()
    logfire.instrument_pydantic_ai()


def load_examples(df_path: Path) -> list[QueryProductExample]:
    df = pl.read_parquet(df_path)

    return [
        QueryProductExample(
            example_id=row["example_id"],
            query_info=QueryInfo(
                query_id=row["query_id"],
                query=row["query"],
            ),
            product_info=ProductInfo(
                product_id=row["product_id"],
                product_title=row["product_title"],
                product_description=row.get("product_description", ""),
                product_bullet_point=row.get("product_bullet_point", ""),
                product_brand=row["product_brand"],
                product_color=row.get("product_color", ""),
            ),
        )
        for row in df.to_dicts()
    ]


async def get_classifications(
    examples: list[QueryProductExample], model: str
) -> list[AgentRunResult[QueryProductMatch]]:
    agent = create_classifier_agent(model)
    prompts = [create_classifier_prompt(e) for e in examples]

    # run classifications; note, should properly set return_exceptions=True and filter failures, etc...
    start = time.time()
    results: list[AgentRunResult[QueryProductMatch]] = await asyncio.gather(
        *[agent.run(p) for p in prompts]
    )
    elapsed = time.time() - start

    logger.info(
        f"Classified {len(examples)} items in {elapsed:.1f}s ({elapsed / len(examples):.2f}s/item)"
    )

    return results


async def get_queryfixes(
    examples: list[QueryProductExample], model: str
) -> list[AgentRunResult[QueryFix]]:
    agent = create_queryfix_agent(model)
    prompts = [create_queryfix_prompt(e) for e in examples]

    # run query fixes; note, should properly set return_exceptions=True and filter failures, etc...
    start = time.time()
    results: list[AgentRunResult[QueryFix]] = await asyncio.gather(
        *[agent.run(p) for p in prompts]
    )
    elapsed = time.time() - start

    logger.info(
        f"Fixed queries for {len(examples)} items in {elapsed:.1f}s ({elapsed / len(examples):.2f}s/item)"
    )

    return results


def write_results(
    results_path: str | Path,
    examples: list[QueryProductExample],
    negative_examples: list[QueryProductExample],
) -> None:
    results_report = {
        "negative_predictions": [e.model_dump() for e in negative_examples],
        "positive_predictions": [
            e.model_dump() for e in examples if e not in negative_examples
        ],
    }
    with open(results_path, "w") as f:
        json.dump(results_report, f, indent=2, default=str)
    logger.info(f"Prediction results written to {results_path}")


async def main(model: str = DEFAULT_MODEL, df_path: Path = DEFAULT_DATASET_PATH):
    load_dotenv(find_dotenv())
    setup_logging()
    setup_observability()

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model}"

    examples = load_examples(df_path)

    # run llm classifier
    classifier_results = await get_classifications(examples, model=model)

    # add results to examples (this is fragile, ignore exceptions etc...)
    for e, r in zip(examples, classifier_results):
        e.query_product_match = r.output

    # generate performance report
    report_path = DATA_DIR / "results" / f"{run_id}_classifier_performance.json"
    classifier_performance = get_classifier_performance(
        examples, report_path=report_path
    )
    logger.info(classifier_performance)
    logger.info(f"Classier performance report generated at {report_path}")

    # filter negative predictions for query fix
    negative_examples: list[QueryProductExample] = [
        e
        for e in examples
        if e.query_product_match.match_classification
        == MatchClassification.NOT_EXACT_MATCH
    ]

    # run llm query fixer
    queryfix_results = await get_queryfixes(negative_examples, model=model)

    # add results to negative predictions (again, fragile...)
    for e, r in zip(negative_examples, queryfix_results):
        e.query_fix = r.output

    results_path = DATA_DIR / "results" / f"{run_id}_predictions.json"
    write_results(results_path, examples, negative_examples)


if __name__ == "__main__":
    asyncio.run(main(model=DEFAULT_MODEL))

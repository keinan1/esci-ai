import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path

import logfire
import polars as pl
from dotenv import find_dotenv, load_dotenv
from pydantic_ai import Agent, AgentRunResult

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
    examples: list[QueryProductExample],
) -> list[AgentRunResult[QueryProductMatch]]:
    agent = create_classifier_agent()
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


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    load_dotenv(find_dotenv())

    # logfire for observability, not essential to run
    logfire.configure()
    logfire.instrument_pydantic_ai()

    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"

    examples = load_examples(
        data_dir / "processed" / "examples_products_subset.parquet"
    )
    classifier_results = await get_classifications(examples)

    # add classifications back into examples (this is a bit fragile, ignore exceptions etc...)
    for e, r in zip(examples, classifier_results):
        e.query_product_match = r.output

    report_path = data_dir / "results" / f"{run_id}_performance_report.json"
    classifier_performance = get_classifier_performance(
        examples, report_path=report_path
    )
    logger.info(classifier_performance)
    logger.info(f"Full performance report generated at {report_path}")


if __name__ == "__main__":
    asyncio.run(main())

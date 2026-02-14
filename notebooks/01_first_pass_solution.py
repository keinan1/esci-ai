import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    # examine full esci dataset
    # requires that esci data is downloaded and processed - see README setup instructions

    from pathlib import Path

    import polars as pl

    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"

    df_all = pl.read_parquet(data_dir / "processed" / "examples_products.parquet")

    df_all.glimpse()
    return data_dir, df_all, pl


@app.cell
def _(df_all, pl):
    # total examples by esci label
    df_all.group_by("esci_label").agg(
        count=pl.len(),
        unique_queries=pl.col("query_id").unique().count(),
        unique_products=pl.col("product_id").unique().count(),
    )
    return


@app.cell
def _(data_dir, pl):
    # examine subset of three queries

    df_subset = pl.read_parquet(
        data_dir / "processed" / "examples_products_subset.parquet"
    )

    df_subset
    return (df_subset,)


@app.cell
def _():
    # by manual inspection, these are the non-exact-matches (33% error rate)

    negative_example_ids = [
        # batteries
        142660,  # 60 count, not 100 count
        142666,  # AAA, not AA
        # drills
        660823,  # no mention of gyroscopic
        660827,  # no mention of gyroscopic
        660838,  # charger only, no drill
        # paper
        1163629,  # matte, not glossy
        1163634,  # ambiguous: brand field lists Doaaler, title lists Kodak
        1163641,  # matte, not glossy
    ]

    ambiguous_example_ids = [
        # batteries
        142661,  # title says 100 count, bullets say 50 bulk packaging (likely meaning 50 x 2)
    ]
    return (negative_example_ids,)


@app.cell
def _():
    # define pydantic models to structure the task inputs (query-product examples) and outputs (query match, query reformulation)

    from enum import Enum

    from pydantic import BaseModel, Field


    class QueryInfo(BaseModel):
        query_id: int
        query: str


    class ProductInfo(BaseModel):
        product_id: str
        product_title: str
        product_description: str | None = None
        product_bullet_point: str | None = None
        product_brand: str
        product_color: str | None = None


    class MatchClassification(Enum):
        EXACT_MATCH = "exact_match"
        NOT_EXACT_MATCH = "not_exact_match"


    class QueryProductMatch(BaseModel):
        match_classification: MatchClassification = Field(
            ...,
            description="Classification of whether the product is an exact match for the query specifications.",
        )
        reasoning: str = Field(
            ...,
            description=f"Succinct reason for the classification. If classified as a {MatchClassification.EXACT_MATCH}, return 'All query specifications satisfied by the product.' If classified as {MatchClassification.NOT_EXACT_MATCH}, cite precisely the query specification(s) not satisfied by the product.",
        )


    class CorrectQuery(BaseModel):
        correct_query: str = Field(
            ...,
            description="Given the product information, formulate a query for which the product would be an exact match",
        )


    class QueryProductExample(BaseModel):
        example_id: int
        query_info: QueryInfo
        product_info: ProductInfo
        query_product_match: QueryProductMatch | None = None
        query_correction: CorrectQuery | None = None

    return (
        MatchClassification,
        ProductInfo,
        QueryInfo,
        QueryProductExample,
        QueryProductMatch,
    )


@app.cell
def _(ProductInfo, QueryInfo, QueryProductExample, df_subset):
    # create list of QueryProductExamples from df_subset

    examples = [
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
        for row in df_subset.to_dicts()
    ]

    examples[:5]
    return (examples,)


@app.cell
async def _():
    # test agent setup
    # requires .env with ollama base url, and downloaded models - see README setup instructions

    from pprint import pprint

    import logfire
    from dotenv import find_dotenv, load_dotenv
    from pydantic_ai import Agent

    load_dotenv(find_dotenv())

    # logfire for observability, not essential to run
    logfire.configure()
    logfire.instrument_pydantic_ai()

    test_settings = {
        "temperature": 0.5,
        "max_tokens": 100,
    }

    test_agent = Agent(
        model="ollama:ministral-3:3b",
        model_settings=test_settings,
    )

    test_result = await test_agent.run("Tell me a joke about data scientists")
    pprint(test_result.output)
    return Agent, pprint


@app.cell
def _(Agent, QueryProductMatch):
    # create classifier agent

    classifier_system_prompt = """
    <TASK OVERVIEW>
    You are a world-class quality assurance agent responsible for improving a product query system. /
    Your task is to examine a query-product pair, and determine whether the product exactly matches the query.
    </TASK OVERVIEW>

    <INSTRUCTIONS>
    To accomplish this task:

    1. Consider all the explicit specifications detailed in the query.
    2. Examine all the explicit product details.
    3. Determine whether all query specifications are explicitly met in the product details.
    4. Output your final answer with a clear, fact-based explanation.
    </INSTRUCTIONS>
    """

    model_settings = {
        "temperature": 0,
        "max_tokens": 150,
    }

    classifier_agent = Agent(
        model="ollama:ministral-3:3b",
        # model="ollama:ministral-3:8b",
        # model="ollama:ministral-3:14b",
        # model="ollama:gpt-oss:20b",
        model_settings=model_settings,
        max_concurrency=10,
        retries=2,
        system_prompt=classifier_system_prompt,
        output_type=QueryProductMatch,
    )
    return (classifier_agent,)


@app.cell
async def _(classifier_agent, examples, pprint):
    # classifier test example

    _e = examples[4]

    _example_prompt = f"""
    <QUERY INFO>
    {_e.query_info.model_dump()}
    </QUERY INFO>
    <PRODUCT INFO>
    {_e.product_info.model_dump()}
    </PRODUCT INFO>
    """

    _result = await classifier_agent.run(_example_prompt)

    pprint(f"EXAMPLE PROMPT: {_example_prompt}")
    pprint(f"RESULT: {_result.output}")
    pprint(f"CLASSIFICATION: {_result.output.match_classification.value}")
    return


@app.cell
async def _(classifier_agent, examples):
    # run all classification examples
    # for pydantic-ai async agent run, see https://ai.pydantic.dev/agent/#__tabbed_9_1

    import asyncio
    import time

    # create prompts
    match_prompts = [
        f"""
    <QUERY INFO>
    {e.query_info.model_dump()}
    </QUERY INFO>
    <PRODUCT INFO>
    {e.product_info.model_dump()}
    </PRODUCT INFO>
    """
        for e in examples
    ]

    # batch run classifier agent
    start = time.time()

    match_results = await asyncio.gather(
        *[classifier_agent.run(mp) for mp in match_prompts]
    )

    end = time.time()
    total_time = end - start
    print(f"Execution time for {len(examples)} items: {round(total_time)} seconds")
    print(f"Average {round(total_time / len(examples))} seconds per item")
    return (match_results,)


@app.cell
def _(
    MatchClassification,
    examples,
    match_results,
    negative_example_ids,
    pprint,
):
    # measure performance

    # add classification to examples
    for e, mr in zip(examples, match_results):
        e.query_product_match = mr.output

    # separate classifications
    exact_matches = [
        e
        for e in examples
        if e.query_product_match.match_classification.value
        == MatchClassification.EXACT_MATCH.value
    ]
    not_exact_matches = [
        e
        for e in examples
        if e.query_product_match.match_classification.value
        == MatchClassification.NOT_EXACT_MATCH.value
    ]
    true_positives = [
        e for e in exact_matches if e.example_id not in negative_example_ids
    ]
    false_positives = [
        e for e in exact_matches if e.example_id in negative_example_ids
    ]
    true_negatives = [
        e for e in not_exact_matches if e.example_id in negative_example_ids
    ]
    false_negatives = [
        e for e in not_exact_matches if e.example_id not in negative_example_ids
    ]

    accuracy = (len(true_positives) + len(true_negatives)) / (len(examples))
    precision = len(true_positives) / (len(true_positives) + len(false_positives))
    recall = len(true_positives) / (len(true_positives) + len(false_negatives))

    pprint(f"ACCURACY: {accuracy}")
    pprint(f"PRECISION: {precision}")
    pprint(f"RECALL: {recall}")
    pprint(
        f"FALSE POSITIVES (non-matches classified as exact matches): {len(false_positives)}"
    )
    for f in false_positives:
        pprint(f.model_dump())
    pprint(
        f"FALSE NEGATIVES (exact matches classified as non-matches): {len(false_negatives)}"
    )
    for f in false_negatives:
        pprint(f.model_dump())
    return


if __name__ == "__main__":
    app.run()

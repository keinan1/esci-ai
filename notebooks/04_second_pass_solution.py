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
        1163641,  # matte, not glossy
    ]

    ambiguous_example_ids = [
        # batteries
        142661,  # ambiguous: title field says 100 count, bullet field says 50 count
        # paper
        1163634,  # ambiguous: title field says Kodak, brand field says Doaaler
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
        reasoning: str = Field(
            ...,
            description=(
                "Succinct reasoning. First list each explicit query specification. "
                "Then for each, state whether the product satisfies it. "
                "If all are satisfied, state 'All query specifications satisfied.' "
                "Otherwise, cite precisely which specification(s) are not met."
            ),
        )
        match_classification: MatchClassification = Field(
            ...,
            description="Classification based on the reasoning above.",
        )


    class QueryFix(BaseModel):
        reasoning: str = Field(
            ...,
            description=(
                "Succinct reasoning. Cite precisely which specification(s) are not met and need to be corrected in the query."
            ),
        )
        corrected_query: str = Field(
            ...,
            description="Provide the corrected query and nothing else.",
        )


    class QueryProductExample(BaseModel):
        example_id: int
        query_info: QueryInfo
        product_info: ProductInfo
        query_product_match: QueryProductMatch | None = None
        query_fix: QueryFix | None = None

    return (
        MatchClassification,
        ProductInfo,
        QueryFix,
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
        "max_tokens": 250,
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
    You are a world-class quality assurance agent responsible for improving a product query system.
    Your task is to examine a query-product pair and determine whether the product is an exact match for the query.
    </TASK OVERVIEW>

    <DEFINITION OF EXACT MATCH>
    A product is an exact match when precisely every specification in the query is satisfied by the product information.
    - The product MAY have additional features, details, or attributes beyond what the query asks for. This is fine — only the query's specifications must be met.
    - Match on MEANING, not exact wording, except in cases where the query specifies technical product codes or model designations.
    - Use common sense and domain knowledge to interpret both queries and product details.
    </DEFINITION OF EXACT MATCH>

    <INSTRUCTIONS>
    1. List every explicit specification in the query.
    2. For each specification, check whether the product information satisfies it (by meaning, not literal text).
    3. Only classify as not_exact_match if a query specification is clearly CONTRADICTED or UNMET by the product information.
    4. Output your reasoning FIRST, then your classification.
    </INSTRUCTIONS>
    """

    model_settings = {
        "temperature": 0,
        "max_tokens": 250,
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
    return classifier_agent, model_settings


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
    return asyncio, match_results, time


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


@app.cell
def _(Agent, QueryFix, model_settings):
    # create query fix agent

    query_fix_system_prompt = """
    <TASK OVERVIEW>
    You are a world-class quality assurance agent responsible for improving a product query system.
    You will be given a query-product pair where the query does not exactly match the product information.
    Your task is to provide the MINIMAL SUBSTANTIVE FIX TO THE INCORRECT QUERY such that the product would be an exact match for it.
    </TASK OVERVIEW>

    <DEFINITION OF EXACT MATCH>
    A product is an exact match when precisely every specification in the query is satisfied by the product information.
    - The product MAY have additional features, details, or attributes beyond what the query asks for. This is fine — only the query's specifications must be met.
    </DEFINITION OF EXACT MATCH>

    <INSTRUCTIONS>
    1. Provide the MINIMAL SUBSTANTIVE FIX TO THE INCORRECT QUERY such that the product would be an exact match for it. To accomplish this, you may REVISE detail(s) that appear in the incorrect query (e.g. brand, counts, features, etc...), but DO NOT ADD specifics of levels of detail beyond the original query.
    2. The corrected_query field must contain your corrected query and nothing else.
    </INSTRUCTIONS>
    """

    query_fix_agent = Agent(
        # model="ollama:ministral-3:3b",
        model="ollama:ministral-3:8b",
        # model="ollama:ministral-3:14b",
        # model="ollama:gpt-oss:20b",
        model_settings=model_settings,
        max_concurrency=10,
        retries=2,
        system_prompt=query_fix_system_prompt,
        output_type=QueryFix,
    )
    return (query_fix_agent,)


@app.cell
async def _(examples, pprint, query_fix_agent):
    # query fix test example

    _e = examples[4]

    _example_prompt = f"""
    <INCORRECT QUERY INFO>
    {_e.query_info.model_dump()}
    </INCORRECT QUERY INFO>
    <PRODUCT INFO>
    {_e.product_info.model_dump()}
    </PRODUCT INFO>
    """

    _result = await query_fix_agent.run(_example_prompt)

    pprint(f"EXAMPLE PROMPT: {_example_prompt}")
    pprint(f"RESULT: {_result.output}")
    pprint(f"FIXED QUERY: {_result.output.corrected_query}")
    return


@app.cell
async def _(MatchClassification, asyncio, examples, query_fix_agent, time):
    # run query fix on all not_exact_match predictions

    predicted_not_exact_match = [
        e
        for e in examples
        if e.query_product_match.match_classification.value
        == MatchClassification.NOT_EXACT_MATCH.value
    ]

    # create prompts
    query_fix_prompts = [
        f"""
    <PRODUCT INFO>
    {e.product_info.model_dump()}
    </PRODUCT INFO>
    <INCORRECT QUERY INFO>
    {e.query_info.model_dump()}
    </INCORRECT QUERY INFO>
    """
        for e in predicted_not_exact_match
    ]

    # batch run classifier agent
    _start = time.time()

    query_fix_results = await asyncio.gather(
        *[query_fix_agent.run(qfp) for qfp in query_fix_prompts]
    )

    _end = time.time()
    _total_time = _end - _start
    print(
        f"Execution time for {len(query_fix_prompts)} items: {round(_total_time)} seconds"
    )
    print(
        f"Average {round(_total_time / len(query_fix_prompts))} seconds per item"
    )
    return predicted_not_exact_match, query_fix_results


@app.cell
def _(query_fix_results):
    query_fix_results
    return


@app.cell
def _(predicted_not_exact_match, query_fix_results):
    # manually eyeball performance

    # add fixed prompt to examples
    for _x, _y in zip(predicted_not_exact_match, query_fix_results):
        _x.query_fix = _y.output

    predicted_not_exact_match
    return


if __name__ == "__main__":
    app.run()

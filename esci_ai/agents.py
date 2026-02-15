from pydantic_ai import Agent

from esci_ai.models import QueryFix, QueryProductExample, QueryProductMatch


def create_classifier_agent(
    model: str = "ollama:ministral-3:8b",
    max_concurrency: int = 10,
    retries: int = 3,
) -> Agent:
    return Agent(
        model=model,
        max_concurrency=max_concurrency,
        retries=retries,
        model_settings={"temperature": 0, "max_tokens": 250},
        system_prompt=CLASSIFIER_SYSTEM_PROMPT,
        output_type=QueryProductMatch,
    )


def create_queryfix_agent(
    model: str = "ollama:ministral-3:8b",
    max_concurrency: int = 10,
    retries: int = 3,
) -> Agent:
    return Agent(
        model=model,
        max_concurrency=max_concurrency,
        retries=retries,
        model_settings={"temperature": 0, "max_tokens": 250},
        system_prompt=QUERYFIX_SYSTEM_PROMPT,
        output_type=QueryFix,
    )


def create_classifier_prompt(example: QueryProductExample) -> str:
    return (
        f"<QUERY INFO>\n{example.query_info.model_dump()}\n</QUERY INFO>\n"
        f"<PRODUCT INFO>\n{example.product_info.model_dump()}\n</PRODUCT INFO>"
    )


def create_queryfix_prompt(example: QueryProductExample) -> str:
    return (
        f"<PRODUCT INFO>\n{example.product_info.model_dump()}\n</PRODUCT INFO>\n"
        f"<INCORRECT QUERY INFO>\n{example.query_info.model_dump()}\n</INCORRECT QUERY INFO>"
    )


CLASSIFIER_SYSTEM_PROMPT = """
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


QUERYFIX_SYSTEM_PROMPT = """
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
</INSTRUCTIONS>
"""

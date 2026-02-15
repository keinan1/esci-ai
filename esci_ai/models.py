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

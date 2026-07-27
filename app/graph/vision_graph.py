"""LangGraph 비전 파이프라인 그래프.

노드 순서:
  identify → search

identify_node 에서 error 가 발생하면 search 를 건너뛰고 종료.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.nodes.identify_node import identify_node
from app.schemas.vision_schemas import GraphState


def build_graph():
    """컴파일된 LangGraph 인스턴스를 반환."""
    builder = StateGraph(GraphState)
    builder.add_node("identify", identify_node)
    builder.add_edge(START, "identify")
    builder.add_edge("identify", END)
    return builder.compile()

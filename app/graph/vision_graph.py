"""LangGraph 비전 파이프라인 그래프.

노드 순서:
  identify → search

identify_node 에서 error 가 발생하면 search 를 건너뛰고 종료.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.nodes.identify_node import identify_node
from app.nodes.search_node import search_node
from app.schemas.vision_schemas import GraphState


def build_graph():
    """컴파일된 LangGraph 인스턴스를 반환.

    search 노드는 state.error 가 있거나 identify_result 가 없으면 스스로
    아무것도 하지 않고 상태를 그대로 넘기므로 분기 엣지를 두지 않는다.
    """
    builder = StateGraph(GraphState)
    builder.add_node("identify", identify_node)
    builder.add_node("search", search_node)
    builder.add_edge(START, "identify")
    builder.add_edge("identify", "search")
    builder.add_edge("search", END)
    return builder.compile()

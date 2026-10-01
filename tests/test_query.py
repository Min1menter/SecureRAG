from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from rag.query import rewrite_question


def test_first_turn_skips_llm():
    llm = FakeListChatModel(responses=["SHOULD NOT BE USED"])
    assert rewrite_question(llm, [HumanMessage("What is Modbus?")]) == "What is Modbus?"
    assert llm.i == 0


def test_followup_is_rewritten():
    llm = FakeListChatModel(responses=['"What are the security weaknesses of Modbus?"'])
    msgs = [HumanMessage("Explain Modbus."), AIMessage("Modbus is a protocol."), HumanMessage("What are its weaknesses?")]
    assert rewrite_question(llm, msgs) == "What are the security weaknesses of Modbus?"


def test_bad_rewrite_falls_back():
    llm = FakeListChatModel(responses=["x" * 1000])
    msgs = [HumanMessage("a"), AIMessage("b"), HumanMessage("and its limits?")]
    assert rewrite_question(llm, msgs) == "and its limits?"

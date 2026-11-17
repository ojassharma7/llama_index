"""Test prompt mixin."""

from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.mixin import (
    PromptDictType,
    PromptMixin,
    PromptMixinType,
)


class MockObject2(PromptMixin):
    def __init__(self) -> None:
        self._prompt_dict_2 = {
            "abc": PromptTemplate("{abc} {def}"),
        }

    def _get_prompts(self) -> PromptDictType:
        return self._prompt_dict_2

    def _get_prompt_modules(self) -> PromptMixinType:
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        if "abc" in prompts:
            self._prompt_dict_2["abc"] = prompts["abc"]


class MockObject1(PromptMixin):
    def __init__(self) -> None:
        self.mock_object_2 = MockObject2()
        self._prompt_dict_1 = {
            "summary": PromptTemplate("{summary}"),
            "foo": PromptTemplate("{foo} {bar}"),
        }

    def _get_prompts(self) -> PromptDictType:
        return self._prompt_dict_1

    def _get_prompt_modules(self) -> PromptMixinType:
        return {"mock_object_2": self.mock_object_2}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        if "summary" in prompts:
            self._prompt_dict_1["summary"] = prompts["summary"]
        if "foo" in prompts:
            self._prompt_dict_1["foo"] = prompts["foo"]


class MockObject0(PromptMixin):
    """Holds MockObject1, so MockObject2 sits two levels down."""

    def __init__(self) -> None:
        self.mock_object_1 = MockObject1()

    def _get_prompts(self) -> PromptDictType:
        return {}

    def _get_prompt_modules(self) -> PromptMixinType:
        return {"mock_object_1": self.mock_object_1}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        pass


def test_prompt_mixin() -> None:
    mock_obj1 = MockObject1()
    prompts = mock_obj1.get_prompts()
    assert prompts == {
        "summary": PromptTemplate("{summary}"),
        "foo": PromptTemplate("{foo} {bar}"),
        "mock_object_2:abc": PromptTemplate("{abc} {def}"),
    }

    assert mock_obj1.mock_object_2.get_prompts() == {
        "abc": PromptTemplate("{abc} {def}"),
    }

    # update prompts
    mock_obj1.update_prompts(
        {
            "summary": PromptTemplate("{summary} testing"),
            "mock_object_2:abc": PromptTemplate("{abc} {def} ghi"),
        }
    )
    assert mock_obj1.get_prompts() == {
        "summary": PromptTemplate("{summary} testing"),
        "foo": PromptTemplate("{foo} {bar}"),
        "mock_object_2:abc": PromptTemplate("{abc} {def} ghi"),
    }


def test_prompt_mixin_nested_two_levels() -> None:
    """
    `update_prompts` must accept the keys `get_prompts` produces.

    `get_prompts` prefixes each sub-module key with the module name as it
    recurses, so a prompt two levels down is keyed "a:b:prompt". Splitting that
    key on every colon raised `ValueError: too many values to unpack`.
    """
    mock_obj0 = MockObject0()

    assert mock_obj0.get_prompts() == {
        "mock_object_1:summary": PromptTemplate("{summary}"),
        "mock_object_1:foo": PromptTemplate("{foo} {bar}"),
        "mock_object_1:mock_object_2:abc": PromptTemplate("{abc} {def}"),
    }

    # Feed back exactly what the getter returned.
    mock_obj0.update_prompts(
        {
            "mock_object_1:summary": PromptTemplate("{summary} testing"),
            "mock_object_1:mock_object_2:abc": PromptTemplate("{abc} {def} ghi"),
        }
    )

    assert mock_obj0.get_prompts() == {
        "mock_object_1:summary": PromptTemplate("{summary} testing"),
        "mock_object_1:foo": PromptTemplate("{foo} {bar}"),
        "mock_object_1:mock_object_2:abc": PromptTemplate("{abc} {def} ghi"),
    }

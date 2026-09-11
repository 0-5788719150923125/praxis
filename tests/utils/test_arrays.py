import pytest

from praxis.utils import coerce_to_list


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, []),
        ("", []),
        ("/data/a", ["/data/a"]),
        ("/data/a,/data/b", ["/data/a", "/data/b"]),
        ("/data/a, /data/b", ["/data/a", "/data/b"]),
        ("a,b,", ["a", "b"]),  # trailing comma tolerated
        (["../platformer", "../rift"], ["../platformer", "../rift"]),
        (["/data/a,/data/b", "/data/c"], ["/data/a", "/data/b", "/data/c"]),
        ('["/data/a", "/data/b"]', ["/data/a", "/data/b"]),
        ([1, 2], ["1", "2"]),  # non-strings stringified
    ],
)
def test_coerce_to_list(value, expected):
    assert coerce_to_list(value) == expected

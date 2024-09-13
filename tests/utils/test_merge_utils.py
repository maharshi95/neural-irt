import unittest

from neural_irt.utils.merge_utils import (
    ListUpdateStrategy,
    deep_merge,
    deep_merge_dict,
    deep_merge_list,
)


class TestMergeUtils(unittest.TestCase):
    def test_deep_merge(self):
        # Test merging dictionaries
        base_dict = {"a": 1, "b": {"c": 2, "d": 3}}
        override_dict = {"b": {"c": 4, "e": 5}, "f": 6}
        expected = {"a": 1, "b": {"c": 4, "d": 3, "e": 5}, "f": 6}
        self.assertEqual(deep_merge(base_dict, override_dict), expected)

        # Test merging lists
        base_list = [1, 2, 3]
        override_list = [4, 5]
        self.assertEqual(deep_merge(base_list, override_list), [4, 5])

        # Test merging non-dict, non-list
        self.assertEqual(deep_merge(1, 2), 2)

    def test_deep_merge_dict(self):
        base = {"a": 1, "b": {"c": 2, "d": [1, 2, 3]}, "e": [4, 5, 6]}
        override = {"b": {"c": 3, "d": [4, 5]}, "e": [7, 8], "f": 9}

        # Test with FULL_REPLACE strategy
        expected_full_replace = {
            "a": 1,
            "b": {"c": 3, "d": [4, 5]},
            "e": [7, 8],
            "f": 9,
        }
        self.assertEqual(deep_merge_dict(base, override), expected_full_replace)

        # Test with APPEND strategy
        expected_append = {
            "a": 1,
            "b": {"c": 3, "d": [1, 2, 3, 4, 5]},
            "e": [4, 5, 6, 7, 8],
            "f": 9,
        }
        self.assertEqual(
            deep_merge_dict(base, override, ListUpdateStrategy.APPEND), expected_append
        )

    def test_deep_merge_list(self):
        base = [1, 2, 3, 4]
        override = [5, 6]

        # Test FULL_REPLACE strategy
        self.assertEqual(deep_merge_list(base, override), [5, 6])

        # Test APPEND strategy
        self.assertEqual(
            deep_merge_list(base, override, ListUpdateStrategy.APPEND),
            [1, 2, 3, 4, 5, 6],
        )

        # Test PREPEND strategy
        self.assertEqual(
            deep_merge_list(base, override, ListUpdateStrategy.PREPEND),
            [5, 6, 1, 2, 3, 4],
        )

        # Test REPLACE strategy
        self.assertEqual(
            deep_merge_list(base, override, ListUpdateStrategy.REPLACE), [5, 6, 3, 4]
        )

        # Test RECURSIVE_REPLACE strategy
        base = [{"a": 1}, {"b": 2}, {"c": 3}]
        override = [{"a": 4}, {"b": 5}]
        expected = [{"a": 4}, {"b": 5}, {"c": 3}]
        self.assertEqual(
            deep_merge_list(base, override, ListUpdateStrategy.RECURSIVE_REPLACE),
            expected,
        )

    def test_edge_cases(self):
        # Test merging empty structures
        self.assertEqual(deep_merge({}, {"a": 1}), {"a": 1})
        self.assertEqual(deep_merge([], [1, 2]), [1, 2])

        # Test merging with None
        self.assertEqual(deep_merge({"a": 1}, None), None)
        self.assertEqual(deep_merge(None, {"a": 1}), {"a": 1})

        # Test merging different types
        self.assertEqual(deep_merge({"a": 1}, [1, 2]), [1, 2])

        # Test invalid list update strategy
        with self.assertRaises(ValueError):
            deep_merge_list([1, 2], [3, 4], "invalid_strategy")

    def test_nested_structures(self):
        base = {"a": [1, 2, {"b": 3}], "c": {"d": [4, 5, 6], "e": {"f": 7}}}
        override = {"a": [8, 9, {"x": 10}], "c": {"d": [11, 12], "e": {"g": 13}}}
        expected = {
            "a": [8, 9, {"b": 3, "x": 10}],
            "c": {"d": [11, 12, 6], "e": {"f": 7, "g": 13}},
        }
        self.assertEqual(
            deep_merge(base, override, ListUpdateStrategy.RECURSIVE_REPLACE), expected
        )


if __name__ == "__main__":
    unittest.main()

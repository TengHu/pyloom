import unittest
from dataclasses import dataclass

from pyloom import Tree


@dataclass
class TestEvent:
    event_type: str
    args: list
    kwargs: dict


class TreeTestCase(unittest.TestCase):
    def test1(self):
        l = Tree()

        self.assertTrue(l.head is not None)
        self.assertEqual(len(l.tails), 1)

        l._create_node(
            event=TestEvent("creation", [], {}),
            data="first",
        )
        self.assertEqual(len(l.tails), 1)
        self.assertEqual(l.current, list(l.tails.values())[0])

        l._create_node(
            event=TestEvent("mutate", [], {}),
            data="second",
        )
        self.assertEqual(len(l.tails), 1)
        self.assertEqual(l.current, list(l.tails.values())[0])

        l._create_node(
            event=TestEvent("mutate", [], {}),
            data="third",
        )
        self.assertEqual(len(l.tails), 1)
        self.assertEqual(l.current, list(l.tails.values())[0])

        self.assertEqual(l.current.data, "third")
        self.assertEqual(l.current.event.event_type, "mutate")

        self.assertEqual(l.head.next_node.data, "first")
        self.assertEqual(l.head.next_node.event.event_type, "creation")

    def test_goto(self):
        l = Tree()

        l._create_node(
            event=TestEvent("creation", [], {}),
            data="first",
        )

        l._create_node(
            event=TestEvent("mutate", [], {}),
            data="second",
        )
        l._create_node(
            event=TestEvent("mutate", [], {}),
            data="third",
        )

    def test_nodes(self):
        l = Tree()

        self.assertEqual(l.index(), 0)
        self.assertEqual(l.reachable_from_head(), [l.head])

        l._create_node(
            event=TestEvent("creation", [], {}),
            data="first",
        )

        self.assertEqual(l.index(), 1)
        self.assertEqual(len(l.reachable_from_head()), 2)
        self.assertEqual(l.reachable_from_head()[-1].data, "first")

        l._create_node(
            event=TestEvent("mutate", [], {}),
            data="second",
        )
        self.assertEqual(l.index(), 2)
        self.assertEqual(len(l.reachable_from_head()), 3)
        self.assertEqual(l.reachable_from_head()[-1].data, "second")

        l._create_node(
            event=TestEvent("mutate", [], {}),
            data="third",
        )

        self.assertEqual(l.index(), 3)
        self.assertEqual(len(l.reachable_from_head()), 4)
        self.assertEqual(l.reachable_from_head()[-1].data, "third")


if __name__ == "__main__":
    unittest.main()

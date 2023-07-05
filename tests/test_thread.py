from __future__ import annotations

import unittest

from pyloom import SnapshotOnEvent, Thread, ThreadCreated, event


class ThreadTestCase(unittest.TestCase):
    class Animal(Thread):
        @event("Registered")
        def __init__(self, name):
            self.name = name
            self.tricks = []
            self.energy = SnapshotOnEvent(50)

        @event("Eat")
        def eat(self, food):
            self.energy += 10

    class Dog(Animal):
        @event("Registered")
        def __init__(self, name):
            super().__init__(name)

        @event("TrickAdded")
        def add_trick(self, trick):
            self.tricks.append(trick)

        @event("Bark")
        def bark(self, msg):
            return f"{self.name} says: {msg}"

    def test_basic(self):
        dog = self.Dog("Fido")
        self.assertEqual(dog.tree.head.data, {})
        self.assertEqual(dog.tree.head.next_node.data, {"energy": 50})

        dog.add_trick(trick="roll over")
        self.assertEqual(dog.name, "Fido")
        self.assertEqual(
            dog.tricks,
            ["roll over"],
        )

        dog.add_trick(trick="roll over again")
        self.assertEqual(dog.tricks, ["roll over", "roll over again"])

        self.assertEqual(dog.tree.head.event.event_name, "Registered")

        self.assertTrue(dog.tree.current.event is None)
        self.assertEqual(dog.tree.current.data["energy"], 50)
        self.assertEqual(
            dog.tree.current.prev_node.prev_node.prev_node.event.event_name,
            "Registered",
        )
        self.assertEqual(
            dog.tree.current.prev_node.prev_node.event.event_name, "TrickAdded"
        )
        self.assertEqual(dog.tree.current.prev_node.prev_node.event.event_args, ())
        self.assertEqual(
            dog.tree.current.prev_node.prev_node.event.event_kwargs,
            {"trick": "roll over"},
        )

        # Rebuild thread from events
        events = dog.events()
        copy = None
        for e in events:
            copy = e.mutate(copy)

        self.assertEqual(copy, dog)

    def test_handler_with_returns(self):
        dog = self.Dog("Fido")

        self.assertEqual(dog.bark("hello"), "Fido says: hello")
        self.assertEqual(dog.last_event().mutate(dog), dog)
        self.assertEqual(
            dog.last_event().mutate(dog, return_response=True), "Fido says: hello"
        )

    def test_override_handler(self):
        class Dog(self.Dog):
            @event("Eat")
            def eat(self, food):
                self.energy += 20

        dog = self.Dog("Tom")
        dog.eat("bone")
        events = dog.events()

        # Rebuild thread from events
        copy = None
        for e in events:
            copy = e.mutate(copy)
        self.assertEqual(copy, dog)

    def test_replay_events_on_different_thread(self):
        class Dog(self.Dog):
            @event("Eat")
            def eat(self, food):
                self.energy += 20

        dog = self.Dog("Tom")
        dog.eat("bone")
        dog.eat("apple")

        self.assertEqual(dog.energy, 70)

        # Rebuild thread from events
        events = dog.events()
        copy = None
        for e in events:
            if isinstance(e, ThreadCreated):
                copy = e.mutate(copy, thread_class=Dog)
            else:
                copy = e.mutate(copy)
        self.assertEqual(copy.energy, 90)

    def test_rewind_forward(self):
        dog = self.Dog("Fido")

        dog.eat("bone")
        dog.eat("apple")

        self.assertEqual(dog.tree.current.prev_node.event.event_args, ("apple",))
        self.assertEqual(
            dog.tree.current.prev_node.prev_node.event.event_args, ("bone",)
        )
        self.assertEqual(dog.energy, 70)

        dog.rewind()

        self.assertFalse(hasattr(dog, "energy"))
        self.assertEqual(dog.tree.current.event.event_name, "Registered")
        self.assertEqual(dog.tree.current.event.event_args, ("Fido",))

        dog.forward(steps=1)

        self.assertEqual(dog.energy, 50)
        self.assertEqual(dog.tree.current.event.event_name, "Eat")
        self.assertEqual(dog.tree.current.event.event_args, ("bone",))

        dog.forward(steps=1)
        self.assertEqual(dog.energy, 60)
        self.assertEqual(dog.tree.current.event.event_name, "Eat")
        self.assertEqual(dog.tree.current.event.event_args, ("apple",))

        dog.forward(steps=1)
        self.assertEqual(dog.energy, 70)
        self.assertTrue(dog.tree.current.event is None)

        # Rebuild thread from events
        events = dog.events()
        copy = None
        for e in events:
            copy = e.mutate(copy)
        self.assertEqual(copy, dog)

    def test_event_chaining(self):
        class Dog(self.Dog):
            @event("Registered")
            def __init__(self, name):
                super().__init__(name)

                self.update_snapshot_on_event()
                self.eat_twice("bone")

            @event("EatTwice")
            def eat_twice(self, food):
                self.eat(food)
                self.eat(food)

        dog = Dog("Fido")
        self.assertEqual(dog.energy, 70)

        # Replay level 0 events should rebuild the thread
        events = dog.events((0,))
        self.assertEqual(len(events), 1)
        copy = None
        for e in events:
            copy = e.mutate(copy)
        self.assertEqual(copy, dog)

        events = dog.events((1,))
        self.assertEqual(len(events), 1)

        events = dog.events((2,))
        self.assertEqual(len(events), 2)

        # Replay level 0,1 events
        events = dog.events((0, 1))
        self.assertEqual(len(events), 2)
        copy = None
        for e in events:
            copy = e.mutate(copy)
        self.assertEqual(copy.energy, 90)

        # Replay level 0, 1, 2 events
        events = dog.events((0, 1, 2))
        self.assertEqual(len(events), 4)
        copy = None
        for e in events:
            copy = e.mutate(copy)
        self.assertEqual(copy.energy, 110)

    def test_throw_exception(self):
        class Dog(self.Dog):
            @event("ThrowException")
            def throw_exception(self):
                raise Exception("This is an exception")

        dog = Dog("Fido")
        dog.eat("bone")

        self.assertTrue(dog.tree.current.event is None)
        self.assertTrue(dog.tree.current.next_node is None)
        self.assertEqual(len(dog.events()), 2)

        with self.assertRaises(Exception) as context:
            dog.throw_exception()

        self.assertEqual(str(context.exception), "This is an exception")

        # When exception is thrown, the current node should store the last event
        self.assertTrue(dog.tree.current.event is not None)
        self.assertTrue(dog.tree.current.next_node is None)
        self.assertEqual(len(dog.events()), 2)

    def test_init_event(self):
        dog1 = self.Dog("Fido")
        init_event = dog1.last_event()
        dog2 = init_event.mutate(None)

        self.assertEqual(init_event.event_name, "Registered")
        self.assertEqual(dog1, dog2)

    def test_snapshot_on_event(self):
        class MyClass(Thread):
            @event("Init")
            def __init__(self, a, b, c):
                self.a = SnapshotOnEvent(a)
                self.b = SnapshotOnEvent(b)
                self.c = SnapshotOnEvent(c)

        my_class = MyClass(1, "str", [1, 2, 3])

        self.assertTrue(isinstance(my_class.a, int))
        self.assertTrue(isinstance(my_class.b, str))
        self.assertTrue(isinstance(my_class.c, list))

    def test_events(self):
        dog = self.Dog("Fido")

        for _ in range(10):
            dog.eat("bone")

        # Rewind 5 steps
        dog.rewind(5)

        # Rebuild up to half way
        copy = None
        for event in dog.events():
            copy = event.mutate(copy)
        self.assertEqual(copy, dog)

        # Replay remaining events
        for event in dog.remain_events():
            copy = event.mutate(copy)
        self.assertEqual(copy, dog.forward())

    def test_apply_events(self):
        dog = self.Dog("Fido")
        dog.eat("bone")
        dog.apply_events([dog.last_event()] * 2, clone=True)
        self.assertEqual(dog.energy, 80)
        # All unique events
        self.assertEqual(len(set([id(event) for event in dog.events()])), 4)

        dog.rewind().forward(1)
        dog.eat("bone")
        dog.apply_events([dog.last_event()] * 2, clone=False)
        self.assertEqual(dog.energy, 80)
        # Three eat events are the same
        self.assertEqual(len(set([id(event) for event in dog.events()])), 2)

    def test_branchout(self):
        dog = self.Dog("Fido")
        dog.eat("bone")
        dog.apply_events([dog.last_event()] * 10, clone=True)
        self.assertEqual(dog.tree.total_nodes(), 13)
        self.assertEqual(len(dog.tree.reachable_from_head()), 13)

        # Branch out
        dog.rewind(4)
        self.assertEqual(dog.tree.index(), 13 - 4 - 1)  # -1 because it's 0-indexed
        dog.apply_events([dog.last_event()] * 10, clone=True)
        self.assertEqual(dog.tree.total_nodes(), 13 + 10)
        self.assertEqual(len(dog.tree.reachable_from_head()), 13 - 4 + 10)

        # Branch out again
        dog.rewind(2)
        self.assertEqual(
            dog.tree.index(), 13 - 4 + 10 - 2 - 1
        )  # -1 because it's 0-indexed
        dog.apply_events([dog.last_event()] * 4, clone=True)
        self.assertEqual(dog.tree.total_nodes(), 13 + 10 + 4)
        self.assertEqual(len(dog.tree.reachable_from_head()), 13 - 4 + 10 - 2 + 4)


if __name__ == "__main__":
    unittest.main()

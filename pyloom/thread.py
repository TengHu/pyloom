from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import FunctionType, WrapperDescriptorType
from typing import Any, Dict, Optional, Sequence, Type, Union

from pyvis.network import Network

from pyloom import tree

from .class_resolver import (
    get_fully_qualified_identifier,
    resolve_fully_qualified_identifier,
)
from .tree import Event, Node, Tree

############################
# Event Definitions
############################


@dataclass
class CanMutateThread(Event):
    """
    Event that mutates the thread and save snapshots of its state in nodes.
    """

    event_mutate_level: int = 0

    def event_depth_decorator(decorated_func):
        def wrapper(event, thread, *args, **kwargs):
            assert event is not None
            assert thread is not None

            thread.event_mutate_level += 1
            event.event_mutate_level = thread.event_mutate_level
            ret = decorated_func(event, thread, *args, **kwargs)
            thread.event_mutate_level -= 1
            return ret

        return wrapper

    @event_depth_decorator
    def mutate(self, thread, return_response=False):
        assert thread is not None

        # Save event in current node
        thread.tree.current.event = self

        # Save data in current node
        thread.tree.current.data = {
            k: thread.attributes_snapshotters[k](v)
            for k, v in thread.__dict__.items()
            if k in thread.attributes_snapshotters
        }

        # Create new node
        thread.tree._create_node(
            data={},  # placeholder
        )

        try:
            response = self.apply(thread)

            # Save data in the latest node
            thread.tree.current.data = {
                k: thread.attributes_snapshotters[k](v)
                for k, v in thread.__dict__.items()
                if k in thread.attributes_snapshotters
            }

        except Exception as e:
            thread.rewind(steps=1)
            thread.tree.current.next_node = None
            raise e

        if return_response:
            return response
        return thread

    def apply(self, thread: Thread) -> Any:
        pass


@dataclass
class CanInitThread(CanMutateThread):
    """
    Event that mutates the thread by initializing its state.
    """

    fully_qualified_class_identifier: Optional[str] = None

    def mutate(self, thread, thread_class=None):
        if thread is None:
            if thread_class is None:
                thread_class = resolve_fully_qualified_identifier(
                    self.fully_qualified_class_identifier
                )

            thread = thread_class.__new__(thread_class)
            thread.attributes_snapshotters = (
                {}
            )  # store snapshot functions for attributes
            thread.tree = tree.Tree()

        # Level of command and event nesting of thread
        thread.event_mutate_level = -1

        # Save event in current node
        thread.tree.current.event = self

        # Save data in current node
        thread.tree.current.data = {
            k: thread.attributes_snapshotters[k](v)
            for k, v in thread.__dict__.items()
            if k in thread.attributes_snapshotters
        }

        # Create new node
        thread.tree._create_node(
            data={},  # placeholder
        )

        try:
            # Increment event_mutate_level
            thread.event_mutate_level += 1
            self.event_mutate_level = thread.event_mutate_level

            # Invoke init method
            thread.__init__(
                *self.__dict__["event_args"], **self.__dict__["event_kwargs"]
            )

            # Decrement event_mutate_level
            thread.event_mutate_level -= 1

            Thread.unwrap_snapshot_on_event(thread)

            # Save data in the latest node
            thread.tree.current.data = {
                k: thread.attributes_snapshotters[k](v)
                for k, v in thread.__dict__.items()
                if k in thread.attributes_snapshotters
            }

        except Exception as e:
            thread.rewind(steps=1)
            thread.tree.current.next_node = None
            raise e
        return thread


@dataclass
class ThreadCreated(CanInitThread):
    pass


@dataclass
class ThreadDecoratedEvent(CanMutateThread):
    def apply(self, thread: Thread) -> Any:
        super().apply(thread)

        command = thread._event_handlers.decorated_event_name_to_command_obj[
            self.event_name
        ]
        return command.decorated_method(thread, *self.event_args, **self.event_kwargs)


############################
# Commands
############################


class Command:
    def __init__(
        self,
        name: str,
        decorated_obj: FunctionType,
    ):
        """
        Initialize a command.

        Parameters:
            name (str): The name of the command event.
            decorated_obj (FunctionType): The decorated method.
        """
        self.decorated_method: Union[FunctionType, WrapperDescriptorType]

        self.event_name = name

        if isinstance(decorated_obj, FunctionType):
            self.decorated_method = decorated_obj

    def __get__(
        self, instance: Optional[Thread], owner: MetaThread[Thread]
    ) -> InstanceCommandMethod:
        """
        Bind a command with an instance.

        Note:
            The `__get__` method is a descriptor method that is called when the command is accessed from an instance.
            It returns an instance-specific command method that is bound to the given instance.
        """
        return InstanceCommandMethod(self, instance)


class InstanceCommandMethod:
    def __init__(self, event_decorator: Command, thread: Thread):
        self.event_decorator = event_decorator
        self.__module__ = event_decorator.decorated_method.__module__
        self.__name__ = event_decorator.decorated_method.__name__
        self.__qualname__ = event_decorator.decorated_method.__qualname__
        self.__annotations__ = event_decorator.decorated_method.__annotations__
        self.__doc__ = event_decorator.decorated_method.__doc__
        self.thread = thread

    def trigger(self, *args: Any, **kwargs: Any) -> Any:
        event_cls = self.thread._event_handlers.decorated_event_name_to_event_cls[
            self.event_decorator.event_name
        ]

        return self.thread.trigger_event(
            event_cls, self.event_decorator.event_name, *args, **kwargs
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.trigger(*args, **kwargs)


############################
# Commands
############################


def event(arg=None):
    """
    Decorator to create an command.

    Parameters:
        arg (str): The name of the event.

    Returns:
        function: The decorator function 'create_command'.

    Example:
        @event("on_click")
        def handle_click(event_data):
            print("Button clicked!")

    """
    if isinstance(arg, str):
        event_name = arg

        def create_command(decorated_obj):
            """
            Method to create a command with the given name.

            Parameters:
                decorated_obj: The object being decorated.

            Returns:
                Command: The Command object containing the event name and obj.
            """
            command = Command(name=event_name, decorated_obj=decorated_obj)
            return command

        return create_command
    else:
        raise ThreadException("Invalid argument to event decorator")


class ThreadException(Exception):
    """
    Custom exception to raise in thread class.
    """


@dataclass
class EventHandlers:
    """
    A class to store event handlers for various decorated events.

    Parameters
    ----------
    decorated_event_name_to_command_obj : Dict[str, Command]
        A dictionary mapping decorated event names to corresponding Command objects
        that generate those events. (Default is an empty dictionary).

    decorated_event_name_to_event_cls : Dict[str, Union[Type[ThreadDecoratedEvent], Type[ThreadCreated]]]
        A dictionary mapping decorated event names to corresponding event classes.
        (Default is an empty dictionary).

    Attributes
    ----------
    decorated_event_name_to_command_obj : Dict[str, Command]
        A dictionary mapping decorated event names to corresponding Command objects
        that generate those events.

    decorated_event_name_to_event_cls : Dict[str, Union[Type[ThreadDecoratedEvent], Type[ThreadCreated]]]
        A dictionary mapping decorated event names to corresponding event classes.
    """

    decorated_event_name_to_command_obj: Dict[str, Command] = field(
        default_factory=dict
    )
    decorated_event_name_to_event_cls: Dict[
        str, Union[Type[ThreadDecoratedEvent], Type[ThreadCreated]]
    ] = field(default_factory=dict)


class SnapshotOnEvent:
    """
    Decorator to wrap an object and take snapshots on events.

    Parameters:
        wrapped (Any): The object to be wrapped.
        snapshotter (Callable[[Any], Any], optional): A function to take a snapshot of the object.
            If not provided, `copy.deepcopy()` is used as the default snapshotter.
    """

    def __init__(self, wrapped, snapshotter=None):
        self.wrapped = wrapped

        if snapshotter is None:
            self.snapshotter = lambda x: copy.deepcopy(x)
        else:
            self.snapshotter = snapshotter


class MetaThread(type):
    def __call__(self, *args: Any, **kwargs: Any):
        if not hasattr(self, "_created_event_name"):
            raise ThreadException("Thread class must have a decorated __init__ method")

        return self._create_thread(
            self._created_event_name,
            *args,
            **kwargs,
        )


class Thread(metaclass=MetaThread):
    """
    Represent a thread object with a doubly-linked list of states.

    When an event is triggered, the event is saved in the current node, applied to the thread,
    and a new node is created to save a snapshot of the thread state.

    Attributes:
        metaclass (MetaThread): The metaclass used for customization.
    """

    _event_handlers = EventHandlers()

    @classmethod
    def unwrap_snapshot_on_event(cls, thread):
        """Unwrap SnapshotOnEvent objects in the thread."""
        for k, v in thread.__dict__.items():
            if isinstance(v, SnapshotOnEvent):
                thread.attributes_snapshotters[k] = v.snapshotter
                v = v.wrapped
                thread.__dict__[k] = v

    def update_snapshot_on_event(self):
        Thread.unwrap_snapshot_on_event(self)

    @classmethod
    def _create_thread(
        cls,
        create_event_name: str,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Create a thread object using the specified create event.

        Parameters:
            create_event_name (str): The name of the create event to trigger.
            *args (Any): Variable-length argument list to pass to the create event.
            **kwargs (Any): Arbitrary keyword arguments to pass to the create event.

        Returns:
            Thread: The created thread object.

        Note:
            The create event should be responsible for initializing the thread object and returning it in its `mutate` method.
        """
        created_event_cls = cls._event_handlers.decorated_event_name_to_event_cls[
            create_event_name
        ]
        created_event = created_event_cls(
            fully_qualified_class_identifier=get_fully_qualified_identifier(cls),
            event_name=create_event_name,
            event_args=args,
            event_kwargs=kwargs,
        )
        thread = created_event.mutate(None)

        assert thread is not None
        return thread

    def __init_subclass__(cls, **kwargs):
        """
        Initialize the event handlers for a subclass of Thread.

        This method is automatically called when a class is subclassed from Thread.
        It allows you to perform custom actions during the subclassing process.

        Parameters:
            cls: The subclass being created.
            **kwargs: Additional keyword arguments passed during subclassing.
        """
        super().__init_subclass__(**kwargs)

        # Inherit event handlers from parent class.
        cls._event_handlers = copy.deepcopy(cls._event_handlers)

        for _, attr_value in tuple(cls.__dict__.items()):
            if isinstance(attr_value, Command):
                command = attr_value

                cls._event_handlers.decorated_event_name_to_command_obj[
                    command.event_name
                ] = command

                if command.decorated_method.__name__ == "__init__":
                    cls._event_handlers.decorated_event_name_to_event_cls[
                        command.event_name
                    ] = ThreadCreated
                    cls._created_event_name = command.event_name

                    # un-decorate __init__
                    cls.__init__ = command.decorated_method
                else:
                    cls._event_handlers.decorated_event_name_to_event_cls[
                        command.event_name
                    ] = ThreadDecoratedEvent

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other) and {
            k: v
            for k, v in self.__dict__.items()
            if k not in ("tree", "attributes_snapshotters")
        } == {
            k: v
            for k, v in other.__dict__.items()
            if k not in ("tree", "attributes_snapshotters")
        }

    def trigger_event(
        self,
        event_class: Type[CanMutateThread],
        event_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Create and trigger an event.

        Parameters:
            event_class (Type[CanMutateThread]): The class representing the event.
            event_name (str): The name of the event.
            *args (Any): Variable-length argument list to pass to the event.
            **kwargs (Any): Arbitrary keyword arguments to pass to the event.

        Note:
            The event_class should be a class implementing the CanMutateThread interface,
            which has a `mutate` method to apply changes to the object.
        """
        new_event = event_class(
            event_name=event_name,
            event_args=args,
            event_kwargs=kwargs,
        )
        return new_event.mutate(self, return_response=True)

    def _is_rewindable(self, input):
        return False

    ############################
    # Event Operations
    ############################

    def _goto(self, hash):
        """
        Go to a specific node in the doubly linked list and update the current instance's attributes.

        Parameters:
            hash: The hash value to identify the desired node in the doubly linked list.

        Returns:
            None

        Raises:
            KeyError: If the provided 'hash' is not present in the '_node_dict'.

        Example:
            # Assuming 'tree' is an instance of a doubly linked list containing nodes with data attributes.
            # The hash is used to find the corresponding node in the doubly linked list and update the current instance.
            self._goto("some_hash_value")
        """
        node = self.tree._node_dict[hash]

        # Update the attributes of the current thread with the data from the node.
        self.__dict__.update(node.data)

        # Set the 'current' attribute of the doubly linked list.
        self.tree.current = node

    def rewind(self, steps=None, hash=None) -> Thread:
        """
        Rewind the state of the thread to a previous point in time.

        Parameters:
            steps (int, optional): Number of steps to rewind. If None, it will rewind to the initial state.
            hash (hashable, optional): A hash value representing a specific state to rewind to.

        Returns:
            Thread: The thread object after rewinding.

        Note:
            This method rewinds the object's state by traversing a doubly-linked list of states.
            If both 'steps' and 'hash' are provided, 'hash' will be used to determine the state to rewind to.
        """

        def _rewind_event_dependencies(curr):
            """
            Helper function to rewind the event dependencies of a node.

            Parameters:
                curr_node (Node): The node whose event dependencies need to be rewound.
            """
            if curr.event:
                for arg in curr.event.event_args:
                    if self._is_rewindable(arg):
                        arg.rewind()
                for arg in curr.event.event_kwargs.values():
                    if self._is_rewindable(arg):
                        arg.rewind()

        if hash is not None:
            self._goto(hash)
        else:
            curr = self.tree.current
            if steps is None:
                # Rewind to the initial node.
                while curr.prev_node:
                    _rewind_event_dependencies(curr)
                    curr = curr.prev_node
            else:
                if steps == 0:
                    return self
                else:
                    curr = self.tree.current
                    # Rewind the specified number of steps, if possible.
                    while steps > 0 and curr.prev_node:
                        _rewind_event_dependencies(curr)
                        curr = curr.prev_node
                        steps -= 1
            self.tree.current = curr

        # Update the thread's attributes with the current node data.
        for k in self.attributes_snapshotters.keys():
            if k in self.tree.current.data:
                self.__dict__[k] = self.tree.current.data[k]
            else:
                if k in self.__dict__:
                    del self.__dict__[k]
        return self

    def forward(self, steps=None) -> Thread:
        """
        Move the object's state forward by a specified number of steps.

        Parameters:
            steps (int, optional): Number of steps to move forward. If None, it will move to the latest node.

        Returns:
            Thread: The thread object after moving forward.
        """
        curr = self.tree.current
        if steps is not None:
            while steps > 0 and curr.next_node:
                curr = curr.next_node
                steps -= 1
        else:
            while curr.next_node:
                curr = curr.next_node
        self.tree.current = curr

        # Update the thread's attributes with the current node data.
        for k in self.attributes_snapshotters.keys():
            if k in self.tree.current.data:
                self.__dict__[k] = self.tree.current.data[k]
            else:
                if k in self.__dict__:
                    del self.__dict__[k]
        return self

    def events(
        self,
        event_mutate_levels=None,
    ) -> Sequence[CanMutateThread]:
        """
        Collect the events to build up to current thread state.

        Replaying level 0 events should guaratee reproduce the same thread state.

        Returns:
            Sequence[CanMutateThread]: A sequence of CanMutateThread objects representing the collected events.
        """
        if event_mutate_levels is None:
            show_all = True
        else:
            show_all = False

        nodes = self.tree.reachable_from_head()
        index = self.tree.index()

        events = []

        for i in range(index):
            node = nodes[i]
            if node.event is not None:
                if show_all or node.event.event_mutate_level in event_mutate_levels:
                    events.append(node.event)
        return events

    def remain_events(self, event_mutate_levels=None):
        if event_mutate_levels is None:
            show_all = True
        else:
            show_all = False

        nodes = self.tree.reachable_from_head()
        index = self.tree.index()

        events = []

        for i in range(index, len(nodes)):
            node = nodes[i]
            if node.event is not None:
                if show_all or node.event.event_mutate_level in event_mutate_levels:
                    events.append(node.event)
        return events

    def last_event(self, event_mutate_levels=None) -> CanMutateThread:
        """Return the last event in the doubly-linked list of thread."""
        events = self.events(event_mutate_levels)

        if events:
            return events[-1]

    def apply_events(self, events: Sequence[CanMutateThread], clone=True) -> Thread:
        for event in events:
            if clone:
                event = event.clone()
            event.mutate(self)
        return self

    def to_networkx(self):
        # TODO: implementation
        # hover on node, show hash
        pass

    ############################
    # tree
    ############################

    def distance_from_start(self):
        return self.tree.distance_from_head(self.tree.current)

    def to_pyvis_network(self):
        reachable_from_head = set(
            [node.hash for node in self.tree.reachable_from_head()]
        )
        g = Network(notebook=True, directed=True)
        hash2index = {}

        for i, (hash, node) in enumerate(self.tree._node_dict.items()):
            hash2index[hash] = i

            label = None
            color = "#CCCCCC"

            if hash in reachable_from_head:
                color = "#00CC99"

            if hash == self.tree.head.hash:
                label = "HEAD"
                color = "#0077B3"

            if hash == self.tree.current.hash:
                label = "CURRENT"
                color = "#cc4a33"

            if label:
                g.add_node(i, label=label, color=color)
            else:
                g.add_node(i, color=color)

            if node.prev_node:
                g.add_edge(
                    hash2index[node.prev_node.hash],
                    hash2index[hash],
                    label="{}_{}".format(
                        node.prev_node.event.event_mutate_level,
                        node.prev_node.event.event_name,
                    ),
                )
        return g

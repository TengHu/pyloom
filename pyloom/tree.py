from __future__ import annotations

import copy
import random
import uuid
from dataclasses import dataclass


@dataclass
class Event:
    event_name: str
    event_args: list
    event_kwargs: dict

    def clone(self):
        return copy.deepcopy(self)


class DoublyLinkedListException(Exception):
    pass


class Node:
    def __init__(self, hash_value, data, context, event=None):
        """
        A node representing a state in the DoublyLinkedList.

        Parameters:
            hash_value (str): The hash value representing the node.
            data (dict): A dictionary containing data representing the state.
            context (dict): A dictionary containing additional context for the state.
            event (Event, optional): An event associated with the state, if applicable.
        """
        self.data = data
        self.context = context

        self.hash = hash_value
        self.event = event

        self.next_node = None
        self.prev_node = None


class Tree:
    def __init__(self):
        """
        A doubly-linked list implementation.
        """
        self.dummy = Node(
            hash_value=self._compute_hash(str(random.randint(0, 10000))),
            data={},
            context={},
        )

        self.head = self.dummy
        # Todo: add tests
        self.tails = {self.dummy.hash: self.dummy}  # Store tail nodes

        self.current = self.head
        self._node_dict = {self.dummy.hash: self.dummy}

    def get_node_by_hash(self, hash):
        """
        Retrieve the node by its hash.

        Parameters:
            hash (str): The hash value of the node.

        Returns:
            Node: The node with the specified hash.
        """
        return self._node_dict[hash]

    def _compute_hash(self, previous_hash):
        return uuid.uuid4().hex

    def _create_node(self, data, event=None, context=None):
        if context is None:
            context = {}

        # Inherit context from previous node
        _context = {}
        if self.current.prev_node:
            _context = copy.deepcopy(self.current.prev_node.context)
        _context.update(context)

        hash_value = self._compute_hash(self.current.hash)
        new_node = Node(
            hash_value=hash_value,
            data=data,
            context=context,
            event=event,
        )
        self.current.next_node = new_node
        new_node.prev_node = self.current
        self.current = new_node

        self._node_dict[self.current.hash] = self.current

        # update tails
        if new_node.prev_node.hash in self.tails:
            del self.tails[new_node.prev_node.hash]
        self.tails[new_node.hash] = new_node

    ############################
    # Utils
    ############################

    def reachable_from_head(self):
        curr = self.head

        ret = []
        while curr is not None:
            ret.append(curr)
            curr = curr.next_node

        return ret

    def index(self):
        curr = self.head

        idx = 0
        while curr is not None:
            if self.current == curr:
                return idx
            curr = curr.next_node
            idx += 1

        return idx

    def total_nodes(self):
        return len(self._node_dict)

    def goto(self, hash):
        """
        Go to the node with the specified hash.

        Parameters:
            hash (str): The hash value of the node.

        Returns:
            DoublyLinkedList: The DoublyLinkedList instance.
        """
        self.current = self.get_node_by_hash(hash)
        return self

    def _commit_on_current(self, context=None):
        if context is None:
            context = {}
        self.current.context.update(context)
        return self

    def add_context(self, context, node=None):
        if node is None:
            node = self.current

        node.context.update(context)
        return self

    def distance_from_head(self, node: Node):
        """
        Compute the distance from the head of the list.

        Returns:
            int: The distance from the head of the list.
        """
        distance = 0
        current_node = node
        while current_node != self.head:
            distance += 1
            current_node = current_node.prev_node
        return distance

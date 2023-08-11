# 🧶 Pyloom

🪡 A event sourcing framework for building large language model applications 🪡

Pyloom is a framework designed to streamline the development of intricate LLM applications. Drawing inspiration from [event sourcing](https://martinfowler.com/eaaDev/EventSourcing.html), Pyloom applies this concept to the LLM agent development, offering a range of powerful features that enables better dev experience.

Developing intricate and non-deterministic LLM agents involves making multiple LLM calls and intricate control structures, similar to constructing a Marble 
Machine (🔮). If an error arises, developers are frequently compelled to rerun the entire agent workflow. Developers care not only the agent's final outcome but also the steps it takes to arrive there. At times, We want to see how tweaks like extra tools or adjusted prompts in-between can affect the final outcome.

Pyloom is crafted to address these challenges.

Features
- **"Git for Agent":** Pyloom tracks state changes and the evolution of the agent at each step.
- **Agent Replay:** With Pyloom, developers can  replay and navigate through the agent's flow. Once encouter an error in the agent flow, you can fix the issue and restart from the exact point of failure. You don't need to make all those llm calls again and waste tokens.
- **Event Sourcing:** Pyloom employs event sourcing, representing agent actions as events. By replaying the event stream, the agent's state can be reconstructed. Furthermore, developer can apply the same event streams to different agents and compare performance.
- **Reproduce Production Issues:** Leveraging event sourcing, Pyloom facilitates the reproducing of production errors by replaying the identical event streams in development environment.
- **Auditability and Traceability:** Pyloom enhances the audit and traceability within LLM applications.
- **Compatibility:** Pyloom can be used with other LLM frameworks like langchain and guidance.

## Installation
You can install Pyloom using pip:

```python
pip install pyloom
```

## Quick Start


## Contributing
We are extremely open to contributions in various forms: bug fixes, new features, improved documentation, and pull requests. Your input is valuable to us!

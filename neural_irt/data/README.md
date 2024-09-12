# IRT Dataset Schema

This document outlines the IRT dataset schema for the neural_irt project.

## Overview

The IRT dataset consists of three main components:
1. Queries (Questions)
2. Agents (Respondents)
3. Responses

Each component is expected to be in a format that can be loaded using the [`load_as_hf_dataset`](../data/datasets.py#L12) function that takes in a name/path to the dataset and returns a Huggingface Dataset object. Currently, this function supports Huggingface dataset names, and disk-based paths to JSON, JSONL, and HuggingFace datasets.

## Schema Details

### 1. Queries

Each query should be a dictionary with the following keys:
- `id`: A unique identifier for the query
- `text`: The text of the query
- `embedding`: (Optional) A pre-computed embedding of the query

Example:
```json
{
"id": "q001",
"text": "What is the capital of France?",
"embedding": [0.1, 0.2, 0.3, ...]
}
```

### 2. Agents

Each agent should be a dictionary with the following keys:
- `id`: A unique identifier for the agent
- `name`: The name of the agent
- `type`: The type of the agent (e.g., 'user', 'expert', 'system', etc.)

Example:
```json
{
"id": "u001",
"name": "John Doe",
"type": "user"
}


### 3. Responses

Each response should be a dictionary with the following keys:
- `query_id`: The ID of the query being responded to
- `agent_id`: The ID of the agent providing the response
- `ruling`: The response or score given by the agent

Example:
```json
{
"query_id": "q001",
"agent_id": "u001",
"ruling": 1
}
```
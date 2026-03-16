"""Convert the JianhaoNJU/IrtNet-Dataset into neural-irt format.

Downloads 5 CSV files from HuggingFace and produces:
  - agents.jsonl         (one entry per model)
  - queries.jsonl        (one entry per prompt)
  - {split}_responses.jsonl  (one entry per model-prompt pair, for each split)
  - query_embeddings.pt  (dict mapping prompt_id -> 768-dim embedding tensor)

Additionally generates a YAML config for each model type (caimira, hpcirt).

Usage:
    python scripts/prepare_irtnet_data.py --output-dir data/irtnet
    python scripts/prepare_irtnet_data.py --output-dir data/irtnet --embed-model all-mpnet-base-v2
    python scripts/prepare_irtnet_data.py --output-dir data/irtnet --skip-embeddings
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import hf_hub_download

REPO_ID = "JianhaoNJU/IrtNet-Dataset"

CSV_FILES = [
    "train.csv",
    "test.csv",
    "val.csv",
    "model_order.csv",
    "question_order.csv",
]


def download_or_locate_csvs(output_dir: str, local_dir: str | None = None) -> dict[str, Path]:
    """Download CSV files from HuggingFace, or locate them in a local directory."""
    if local_dir is not None:
        local = Path(local_dir)
        paths = {}
        for filename in CSV_FILES:
            p = local / filename
            if not p.exists():
                raise FileNotFoundError(f"Expected {p} in --local-dir but not found.")
            paths[filename] = p
        print(f"  Using local CSVs from {local}")
        return paths

    raw_dir = Path(output_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for filename in CSV_FILES:
        local_path = raw_dir / filename
        if local_path.exists():
            print(f"  [skip] {filename} already exists")
        else:
            print(f"  [download] {filename}")
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=str(raw_dir),
            )
        paths[filename] = local_path
    return paths


def build_agents(model_order_path: Path, response_dfs: dict[str, pd.DataFrame]) -> list[dict]:
    """Build agent entries from model_order.csv.

    Each model becomes an agent. The 'agent_type' is derived from the model
    family prefix (e.g., 'llama', 'mistral', 'gemma').
    """
    model_df = pd.read_csv(model_order_path)
    # The CSV has a single column with model names
    col = model_df.columns[0]
    model_names = model_df[col].tolist()

    # Collect all model_ids actually referenced in responses
    all_model_ids = set()
    for df in response_dfs.values():
        all_model_ids.update(df["model_id"].unique())

    agents = []
    for name in model_names:
        if name not in all_model_ids:
            continue
        # Derive agent type from model name prefix
        agent_type = _infer_model_family(name)
        agents.append({
            "id": name,
            "name": name,
            "agent_type": agent_type,
        })
    return agents


def _infer_model_family(model_name: str) -> str:
    """Infer model family from model name for agent_type grouping."""
    name_lower = model_name.lower().replace("-", "_").replace("/", "_")
    # Common model family prefixes
    families = [
        "llama", "mistral", "mixtral", "gemma", "phi", "qwen",
        "falcon", "mpt", "opt", "bloom", "pythia", "gpt",
        "yi", "deepseek", "internlm", "baichuan", "chatglm",
        "vicuna", "alpaca", "codellama", "starcoder", "command",
        "solar", "openchat", "zephyr", "orca", "nous", "wizard",
    ]
    for family in families:
        if family in name_lower:
            return family
    return "other"


def build_queries(
    question_order_path: Path, response_dfs: dict[str, pd.DataFrame]
) -> list[dict]:
    """Build query entries from response CSVs (with text from question_order.csv fallback).

    Only includes queries that are referenced in at least one response split.
    """
    # Collect all prompt_ids referenced in responses
    all_prompt_ids = set()
    for df in response_dfs.values():
        all_prompt_ids.update(df["prompt_id"].astype(str).unique())

    # Build prompt_id -> entry from responses (has prompt text + category)
    prompt_map: dict[str, dict] = {}
    for df in response_dfs.values():
        for _, row in df.drop_duplicates(subset=["prompt_id"]).iterrows():
            pid = str(row["prompt_id"])
            if pid not in prompt_map:
                entry = {"id": pid}
                if "prompt" in row and pd.notna(row["prompt"]):
                    entry["text"] = str(row["prompt"])
                if "category" in row and pd.notna(row["category"]):
                    entry["category"] = str(row["category"])
                prompt_map[pid] = entry

    # Fill in missing text from question_order.csv if available
    if question_order_path.exists():
        question_df = pd.read_csv(question_order_path)
        prompt_col = "prompt" if "prompt" in question_df.columns else question_df.columns[0]
        for pid in all_prompt_ids:
            if pid in prompt_map and "text" not in prompt_map[pid]:
                # Try to find text in question_order by matching
                # (question_order may not have IDs, just ordered prompts)
                pass

    return list(prompt_map.values())


def build_responses(df: pd.DataFrame) -> list[dict]:
    """Convert a response DataFrame to neural-irt format."""
    responses = []
    for _, row in df.iterrows():
        label = float(row["label"])
        ruling = 1 if label >= 0.5 else 0
        responses.append({
            "query_id": str(row["prompt_id"]),
            "agent_id": str(row["model_id"]),
            "ruling": ruling,
        })
    return responses


def compute_embeddings(
    queries: list[dict], model_name: str = "all-mpnet-base-v2"
) -> dict[str, torch.Tensor]:
    """Compute query embeddings using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    texts_with_ids = [(q["id"], q.get("text", "")) for q in queries]
    ids = [t[0] for t in texts_with_ids]
    texts = [t[1] for t in texts_with_ids]

    print(f"  Encoding {len(texts)} queries with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_tensor=True)

    return {qid: emb for qid, emb in zip(ids, embeddings)}


def write_jsonl(data: list[dict], path: Path):
    """Write a list of dicts as JSONL."""
    with open(path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    print(f"  Wrote {len(data)} entries to {path}")


def write_config(output_dir: Path, n_agents: int, n_agent_types: int, embed_dim: int):
    """Write example YAML configs for caimira and hpcirt."""
    config_dir = output_dir / "configs"
    config_dir.mkdir(exist_ok=True)

    base_data_block = f"""\
data:
  train_set:
    queries: "{output_dir}/queries.jsonl"
    agents: "{output_dir}/agents.jsonl"
    responses: "{output_dir}/train_responses.jsonl"
  val_set: null
  val_sets:
    val:
      responses: "{output_dir}/val_responses.jsonl"
    test:
      responses: "{output_dir}/test_responses.jsonl"
  question_input_format: "embedding"
  agent_input_format: "id"
  query_embeddings_path: "{output_dir}/query_embeddings.pt"
"""

    caimira_config = f"""\
run_tag: "irtnet-caimira"

{base_data_block}
model:
  n_agents: {n_agents}
  n_agent_types: {n_agent_types}
  n_dim: 32
  n_dim_item_embed: {embed_dim}
  rel_mode: "linear"
  dif_mode: "linear"
  fit_guess_bias: false
  fit_agent_type_embeddings: true
  characteristics_bounder: null
  rel_temperature: 0.5

trainer:
  c_reg_skill: 1e-5
  c_reg_difficulty: 1e-5
  c_reg_relevance: 1e-5
  batch_size: 256
  max_epochs: 100
  optimizer: "Adam"
  learning_rate: 1e-3
  ckpt_savedir: "./checkpoints/irtnet"

wandb:
  enabled: true
  project: "neural-irt-irtnet"
  save_dir: "."
"""

    hpcirt_config = f"""\
run_tag: "irtnet-hpcirt"

{base_data_block}
model:
  n_agents: {n_agents}
  n_agent_types: {n_agent_types}
  n_dim: 16
  n_dim_item_embed: {embed_dim}
  disc_mode: "linear"
  diff_mode: "linear"
  conj_diff_mode: "linear"
  comp_mode: "linear"
  n_hidden: 128
  relevance_mode: "derived"
  mu_prior_logit: -2.0
  bifactor_reg: true
  fit_guess_bias: false
  fit_agent_type_embeddings: true
  characteristics_bounder: null

trainer:
  c_reg_skill: 1e-5
  c_reg_difficulty: 1e-5
  c_reg_g: 1e-5
  c_reg_mu: 1e-4
  c_reg_disc: 1e-5
  c_reg_bifactor: 1e-4
  batch_size: 256
  max_epochs: 100
  optimizer: "Adam"
  learning_rate: 1e-3
  ckpt_savedir: "./checkpoints/irtnet"

wandb:
  enabled: true
  project: "neural-irt-irtnet"
  save_dir: "."
"""

    (config_dir / "caimira.yaml").write_text(caimira_config)
    (config_dir / "hpcirt.yaml").write_text(hpcirt_config)
    print(f"  Wrote configs to {config_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Convert JianhaoNJU/IrtNet-Dataset to neural-irt format."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/irtnet",
        help="Directory to write the converted dataset.",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="all-mpnet-base-v2",
        help="Sentence transformer model for query embeddings.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip computing embeddings (use if you already have them).",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Path to already-downloaded CSV files (skips HuggingFace download).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download or locate CSVs
    print("Downloading dataset CSVs...")
    csv_paths = download_or_locate_csvs(args.output_dir, args.local_dir)

    # Step 2: Load response CSVs
    print("Loading response data...")
    splits = ["train", "val", "test"]
    response_dfs = {}
    for split in splits:
        df = pd.read_csv(csv_paths[f"{split}.csv"])
        response_dfs[split] = df
        print(f"  {split}: {len(df)} response rows")

    # Step 3: Build agents
    print("Building agents...")
    agents = build_agents(csv_paths["model_order.csv"], response_dfs)
    agent_types = {a["agent_type"] for a in agents}
    print(f"  {len(agents)} agents, {len(agent_types)} agent types: {sorted(agent_types)}")
    write_jsonl(agents, output_dir / "agents.jsonl")

    # Step 4: Build queries
    print("Building queries...")
    queries = build_queries(csv_paths["question_order.csv"], response_dfs)
    print(f"  {len(queries)} queries")
    write_jsonl(queries, output_dir / "queries.jsonl")

    # Step 5: Build responses per split
    print("Building responses...")
    for split in splits:
        responses = build_responses(response_dfs[split])
        write_jsonl(responses, output_dir / f"{split}_responses.jsonl")

    # Step 6: Compute embeddings
    embed_dim = 768  # default for all-mpnet-base-v2
    embed_path = output_dir / "query_embeddings.pt"
    if args.skip_embeddings:
        print("Skipping embedding computation.")
        if embed_path.exists():
            embeddings = torch.load(embed_path, weights_only=True)
            embed_dim = next(iter(embeddings.values())).shape[0]
    else:
        print("Computing query embeddings...")
        queries_with_text = [q for q in queries if q.get("text")]
        if not queries_with_text:
            print("  WARNING: No query texts found, skipping embeddings.")
        else:
            embeddings = compute_embeddings(queries_with_text, args.embed_model)
            embed_dim = next(iter(embeddings.values())).shape[0]
            torch.save(embeddings, embed_path)
            print(f"  Saved {len(embeddings)} embeddings ({embed_dim}-dim) to {embed_path}")

    # Step 7: Write configs
    print("Writing example configs...")
    write_config(output_dir, len(agents), len(agent_types), embed_dim)

    print("\nDone! To train:")
    print(f"  python -m neural_irt.train --model-type caimira --config-paths {output_dir}/configs/caimira.yaml")
    print(f"  python -m neural_irt.train --model-type hpcirt --config-paths {output_dir}/configs/hpcirt.yaml")


if __name__ == "__main__":
    main()

import wandb


def get_wandb_runs_by_experiment(project_name, run_name):
    api = wandb.Api()

    # Fetch all runs for the specified project
    runs = api.runs(f"{wandb.api.default_entity}/{project_name}")

    # Filter runs by experiment name
    matching_runs = [run for run in runs if run.name == run_name]

    return matching_runs


wandb_runs = get_wandb_runs_by_experiment(
    "neurt-testing",
    "caimira-10-dim_ttqe-emb-tiny-test-query-embeds_Adam-lr=1e-03c-reg-skill=1e-6-diff=1e-6-imp=1e-6__test",
)
for run in wandb_runs[:5]:
    run = wandb.Api().run(f"{wandb.api.default_entity}/{run.project}/{run.id}")
    print(f"Run Name: {run.name}")
    # for file in run.files():
    #     print(file)
    # checkpoints_path = f"{run.dir}/{run.id}/checkpoints"
    # for checkpoint in os.listdir(checkpoints_path):
    #     print(checkpoint)
    print(run._base_dir)
    print(run.dir)
    print(run.path)
# %%

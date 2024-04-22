import datasets
from datasets import concatenate_datasets
from pathlib import Path


def push_data_to_hub(organization, data_name, config_name):
    _datasets = []

    for p in Path(f"data/{data_name}/{config_name}").glob("*"):
        if not p.is_dir():
            continue

        model_name = p.name
        dataset = datasets.load_dataset("json", data_files=f"data/{data_name}/{config_name}/{model_name}/*.json")
        # Remove the column "additional_kwargs" from the dataset
        # Check if the column exists before removing it

        if isinstance(dataset, datasets.Dataset):
            if "additional_kwargs" in dataset.column_names:
                # This handles the raw data.
                dataset = dataset.remove_columns("additional_kwargs")

            dataset.add_column("model_name", column=[model_name] * len(dataset))
            _datasets.append(dataset)
        elif isinstance(dataset, datasets.DatasetDict):
            # This handles the processed data.
            for key in dataset.keys():
                if "additional_kwargs" in dataset[key].column_names:
                    dataset[key] = dataset[key].remove_columns("additional_kwargs")

                dataset[key] = dataset[key].add_column("model_name", column=[model_name] * len(dataset[key]))
            _datasets.append(dataset)

    if isinstance(_datasets[0], datasets.Dataset):
        dataset = concatenate_datasets(_datasets)
    elif isinstance(_datasets[0], datasets.DatasetDict):
        keys = set()
        ddict = {}

        for ds in _datasets:
            keys.update(ds.keys())

        for key in keys:
            ddict[key] = []
            for ds in _datasets:
                ddict[key].append(ds[key])

            ddict[key] = concatenate_datasets(ddict[key])

        dataset = datasets.DatasetDict(ddict)

    dataset.push_to_hub(f"{organization}/doc-topics-{data_name}", config_name=config_name)

    return dataset

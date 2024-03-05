import datasets


def push_data_to_hub(organization, data_name, config_name):
    dataset = datasets.load_dataset("json", data_files=f"data/{data_name}/{config_name}/*.json")
    # Remove the column "additional_kwargs" from the dataset
    # Check if the column exists before removing it

    if isinstance(dataset, datasets.Dataset):
        if "additional_kwargs" in dataset.column_names:
            # This handles the raw data.
            dataset = dataset.remove_columns("additional_kwargs")
    elif isinstance(dataset, datasets.DatasetDict):
        # This handles the processed data.
        for key in dataset.keys():
            if "additional_kwargs" in dataset[key].column_names:
                dataset[key] = dataset[key].remove_columns("additional_kwargs")

    dataset.push_to_hub(f"{organization}/doc-topics-{data_name}", config_name=config_name)

    return dataset

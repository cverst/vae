import json
import argparse
import os


def parse_json(source_file: str, target_file: str) -> None:
    """Parse animal-crossing-villagers.json file for dataset creation.

    Args:
        source_file (str): Path to source file.
        target_file (str): Path to target/destination file. Will be overwritten.
    """

    # Load animal-crossing-villagers.json
    with open(source_file) as f:
        villagers = json.load(f)

        # Remove fields that are not needed
        fields_to_remove = [
            "url",
            "alt_name",
            "title_color",
            "text_color",
            "id",
            "image_url",
            "birthday_month",
            "birthday_day",
            "sign",
            "quote",
            "phrase",
            "clothing",
            "islander",
            "debut",
            "prev_phrases",
            "nh_details",
            "appearances",
        ]

        for villager in villagers:
            for field in fields_to_remove:
                villager.pop(field, None)

        # Save to file, make it pretty, and preserve special characters
        with open(target_file, "w") as f:
            json.dump(villagers, f, indent=4, ensure_ascii=False)
            print(f"File parsed and saved to '{target_file}'.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Path to ACNH villagers JSON source file.")
    parser.add_argument(
        "--target",
        default="../data/annotations/animal-crossing-villagers-parsed.json",
        help="Path to JSON target file.",
    )
    args = parser.parse_args()

    if args.target == "../data/annotations/animal-crossing-villagers-parsed.json":
        PATH = "../data/annotations"
        path_exists = os.path.exists(PATH)
        if not path_exists:
            os.makedirs(PATH)
            print("New data directory created.")

    parse_json(
        source_file=args.source,
        target_file=args.target,
    )

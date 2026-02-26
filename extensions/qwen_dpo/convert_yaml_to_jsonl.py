import yaml
import json
import os
import argparse


def convert_yaml_to_jsonl(yaml_path, jsonl_path):
    if not os.path.exists(yaml_path):
        print(f"Error: Cannot find YAML file at {yaml_path}")
        return

    print(f"Reading from {yaml_path}...")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if not data:
        print("Error: YAML file is empty or invalid.")
        return

    count = 0
    print(f"Writing to {jsonl_path}...")

    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(os.path.abspath(jsonl_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(jsonl_path, 'w', encoding='utf-8') as f_out:
        # Iterate through the top-level categories
        for category, items in data.items():
            # Ensure the value is a list of dictionaries
            if isinstance(items, list):
                for item in items:
                    # Convert the dictionary directly to a JSON string
                    json_line = json.dumps(item, ensure_ascii=False)
                    f_out.write(json_line + '\n')
                    count += 1

    print(f"✓ Successfully converted {count} items to JSONL format.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a YAML file to a JSONL file.")

    # Define required arguments
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input YAML file."
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to the output JSONL file."
    )

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Run the conversion
    convert_yaml_to_jsonl(args.input, args.output)
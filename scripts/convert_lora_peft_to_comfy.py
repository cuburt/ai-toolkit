from safetensors.torch import load_file, save_file
import os


def convert_lora_peft_to_comfy(args):
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"File {args.input} does not exist.")

    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("Loading Hugging Face LoRA...")
    state_dict = load_file(args.input)
    new_dict = {}

    for key, value in state_dict.items():
        # 1. Strip the Hugging Face PEFT wrapper
        new_key = key.replace("base_model.model.", "")

        # 2. Convert PEFT A/B notation to ComfyUI down/up notation
        new_key = new_key.replace("lora_A.weight", "lora_down.weight")
        new_key = new_key.replace("lora_B.weight", "lora_up.weight")

        new_dict[new_key] = value

    print("Saving ComfyUI-compatible LoRA...")
    save_file(new_dict, args.output)
    print(f"Success! Your converted file is saved as: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a PEFT LoRA to a Comfy LoRA.")

    # Define required arguments
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input safetensors."
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to the output safetensors."
    )
    # Parse the arguments from the command line
    args = parser.parse_args()
    convert_lora_peft_to_comfy(args)

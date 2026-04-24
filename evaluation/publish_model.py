import tinker
import argparse

def main():
    parser = argparse.ArgumentParser(description="Publish a model")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    args = parser.parse_args()
    sc = tinker.ServiceClient()
    rest_client = sc.create_rest_client()
    rest_client.publish_checkpoint_from_tinker_path(args.checkpoint_path).result()


if __name__ == "__main__":
    main()
import json
import sys
from data_collector_unit_poc.data_storage import PostRepository, Source

def import_data_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        repository = PostRepository()
        for post_data in data:
            post_data['source'] = Source(post_data['source'])
            repository.add_post(post_data)
    print("Data imported successfully from", file_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python import_data_from_file.py <path_to_json_file>")
    else:
        import_data_from_file(sys.argv[1])

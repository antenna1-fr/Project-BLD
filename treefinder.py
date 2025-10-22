import os

EXCLUDES = {"venv311", "__pycache__", "tmp", ".git", "db", "tb"}

def tree(dir_path, prefix=""):
    items = [d for d in os.listdir(dir_path) if d not in EXCLUDES]
    for i, name in enumerate(sorted(items)):
        path = os.path.join(dir_path, name)
        connector = "└── " if i == len(items) - 1 else "├── "
        print(prefix + connector + name)
        if os.path.isdir(path):
            extension = "    " if i == len(items) - 1 else "│   "
            tree(path, prefix + extension)

tree(".")

import os

ignore_dirs = {'.git', '.mypy_cache', '.pytest_cache', '.venv', '__pycache__', 'data', 'docs', 'node_modules', 'build', 'dist'}
ignore_exts = {'.pyc', '.pdf', '.jpg', '.png', '.jpeg', '.mp4', '.zip', '.tar.gz', '.log', '.sqlite3'}

def write_project_to_md():
    with open('codebase_context.md', 'w', encoding='utf-8') as f:
        f.write("# AntiDeepfake Full Codebase\n\n")
        
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for file in files:
                if any(file.endswith(ext) for ext in ignore_exts):
                    continue
                if file in ['generate_codebase_md.py', 'codebase_context.md', 'package-lock.json', 'response.json']:
                    continue
                
                path = os.path.join(root, file)
                # Skip some heavy things
                if os.path.getsize(path) > 500 * 1024: # skip files > 500KB
                    continue
                
                try:
                    with open(path, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                    f.write(f"## File: `{path}`\n\n")
                    f.write(f"```python\n{content}\n```\n\n")
                except UnicodeDecodeError:
                    pass

write_project_to_md()

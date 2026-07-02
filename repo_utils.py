import os
import shutil
import stat
import tempfile
import git

def remove_readonly(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError as e:
        if e.errno == 145:  # WinError 145: directory not empty
            for root, dirs, files in os.walk(path, topdown=False):
                for name in files:
                    try:
                        os.remove(os.path.join(root, name))
                    except Exception:
                        pass
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except Exception:
                        pass
            try:
                os.rmdir(path)
            except Exception:
                pass
        else:
            raise

def clone_and_list_files(repo_url, base_dir="parsed_code"):
    repo_name = repo_url.rstrip("/").split("/")[-1]
    repo_path = os.path.join(base_dir, repo_name)

    # Step 1: Remove existing folder
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path, onerror=remove_readonly)

    # Step 2: Clone to a temp dir first (with .git)
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            git.Repo.clone_from(repo_url, tmp_dir)

            # Step 3: Copy only supported files to final location
            for root, _, files in os.walk(tmp_dir):
                for file in files:
                    if file.endswith((".py", ".js", ".java", ".cpp", ".ts", ".html")):
                        src_path = os.path.join(root, file)
                        rel_path = os.path.relpath(src_path, tmp_dir)
                        dest_path = os.path.join(repo_path, rel_path)

                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        shutil.copy2(src_path, dest_path)
    except git.exc.GitCommandError:
        return None, None

    # Step 4: Return .py file list (for now only answering Python)
    py_files = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                rel_path = os.path.relpath(os.path.join(root, file), repo_path)
                py_files.append(rel_path)

    return repo_name, py_files

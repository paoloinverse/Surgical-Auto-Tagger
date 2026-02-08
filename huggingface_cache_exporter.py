import os
import shutil
from pathlib import Path
from huggingface_hub import scan_cache_dir

def export_readable_cache(target_base_dir):
    target_path = Path(target_base_dir).resolve()
    os.makedirs(target_path, exist_ok=True)
    
    print(f"Scanning default Hugging Face cache...")
    cache_info = scan_cache_dir()
    
    for repo in cache_info.repos:
        repo_id_safe = repo.repo_id.replace("/", "--")
        repo_type = repo.repo_type
        
        for revision in repo.revisions:
            # Create a human-readable folder name: type_org--model_commit
            folder_name = f"{repo_type}_{repo_id_safe}_{revision.commit_hash[:7]}"
            export_dir = target_path / folder_name
            
            print(f"Exporting: {repo.repo_id} (Revision: {revision.commit_hash[:7]})")
            os.makedirs(export_dir, exist_ok=True)
            
            # revision.snapshot_path is the directory containing symlinks
            # We copy it while following symlinks to get the real data
            try:
                if os.path.exists(revision.snapshot_path):
                    for item in os.listdir(revision.snapshot_path):
                        s = os.path.join(revision.snapshot_path, item)
                        d = os.path.join(export_dir, item)
                        if os.path.islink(s):
                            # Resolve symlink and copy actual file
                            shutil.copy2(os.path.realpath(s), d)
                        elif os.path.isdir(s):
                            shutil.copytree(s, d, symlinks=False, dirs_exist_ok=True)
                        else:
                            shutil.copy2(s, d)
                print(f"   [OK] Saved to {export_dir}")
            except Exception as e:
                print(f"   [ERROR] Failed to export {repo.repo_id}: {e}")

if __name__ == "__main__":
    # Change this to your desired safety directory
    SAFE_ZONE = os.path.expanduser("~/HF_OFFLINE_BACKUP")
    export_readable_cache(SAFE_ZONE)




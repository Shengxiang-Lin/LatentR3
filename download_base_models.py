from huggingface_hub import snapshot_download
# Model repository list
repos = [
    "Qwen/Qwen2.5-1.5B-Instruct"
    #"Qwen/Qwen2-7B-Instruct"
    #"huggyllama/llama-7b"
]
for repo in repos:
    print(f"Downloading {repo}...")
    snapshot_download(
        repo_id=repo,
        local_dir=f"./base_models/{repo.split('/')[-1]}",  # # Save to local directory
        revision="main",  # Use the main branch
        cache_dir="./cache",  # cache directory
        resume_download=True,  # Supports resume from breakpoints

    )
print("All base_models downloaded!")
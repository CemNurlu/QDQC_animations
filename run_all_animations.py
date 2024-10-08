import os

repo_root = "anim_scripts"
lecture_dirs = ["Lecture0", "Lecture1", "Lecture2", "Lecture4"]

for lecture_dir in lecture_dirs:
    full_lecture_dir = os.path.join(repo_root, lecture_dir)
    for root, dirs, files in os.walk(full_lecture_dir):
        for f in files:
            if f.endswith(".py"):
                python_file_path = os.path.join(root, f)
                print(f"RUNNING SCRIPT: {python_file_path}")
                try:
                    exec(open(python_file_path).read())
                    print(f"SUCCESS RUNNING SCRIPT: {python_file_path}")
                except:
                    print(f"ERROR RUNNING SCRIPT: {python_file_path}")
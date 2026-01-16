import os, sys, hashlib

def scan(root, case_sensitive=False):
    files = {}
    root = os.path.abspath(root)
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.normpath(os.path.relpath(full, root)).replace('\\','/')
            key = rel if case_sensitive else rel.lower()
            files[key] = (rel, full, os.path.getsize(full))
    return files

def sha256(path, chunk=65536):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()

def compare_dirs(left, right, case_sensitive=False):
    """直接比较两个目录，返回结果字典而非依赖命令行参数"""
    if not os.path.isdir(left) or not os.path.isdir(right):
        raise FileNotFoundError("请确保两个参数都是存在的文件夹路径。")
    L = scan(left, case_sensitive)
    R = scan(right, case_sensitive)

    setL = set(L.keys())
    setR = set(R.keys())

    onlyL = sorted(setL - setR)
    onlyR = sorted(setR - setL)
    common = sorted(setL & setR)

    diffs = []

    for k in common:
        rel, lf, lsize = L[k]
        _, rf, rsize = R[k]
        if lsize != rsize:
            diffs.append((rel, 'SIZE', lsize, rsize))
        else:
            h1 = sha256(lf)
            h2 = sha256(rf)
            if h1 != h2:
                diffs.append((rel, 'CONTENT', None, None))

    result = {
        "only_left": [L[r][0] for r in onlyL],
        "only_right": [R[r][0] for r in onlyR],
        "diffs": diffs,
        "equal": not (onlyL or onlyR or diffs)
    }
    return result

def print_result(res):
    if res["only_left"]:
        print("仅在左侧存在:")
        for p in res["only_left"]:
            print("  " + p)
    if res["only_right"]:
        print("仅在右侧存在:")
        for p in res["only_right"]:
            print("  " + p)
    if res["diffs"]:
        print("内容/大小不一致:")
        for item in res["diffs"]:
            if item[1] == 'SIZE':
                print(f"  {item[0]}  左大小={item[2]}  右大小={item[3]}")
            else:
                print(f"  {item[0]}  内容不同")
    if res["equal"]:
        print("完全相同（按相对路径、大小和内容比较）")

if __name__ == '__main__':
    # 在当前工作目录下列出子文件夹，直接用 os 读取并交互选择
    cwd = os.path.abspath(os.getcwd())
    dirs = [d for d in os.listdir(cwd) if os.path.isdir(os.path.join(cwd, d))]
    if not dirs:
        print("当前目录下没有子文件夹，请在 e:\\3d 下运行或将目标文件夹放入当前目录。")
        sys.exit(2)

    print("请选择要比较的两个文件夹（输入序号或直接输入完整路径），当前目录:", cwd)
    for i, d in enumerate(dirs, 1):
        print(f"  {i}. {d}")
    a = input("左侧(序号或路径): ").strip()
    b = input("右侧(序号或路径): ").strip()

    def resolve(choice):
        if not choice:
            return None
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(dirs):
                return os.path.join(cwd, dirs[idx])
            return None
        return os.path.abspath(choice)

    left = resolve(a)
    right = resolve(b)
    if not left or not right:
        print("未提供有效文件夹。")
        sys.exit(2)

    try:
        res = compare_dirs(left, right)
        print_result(res)
        sys.exit(0 if res["equal"] else 1)
    except FileNotFoundError as e:
        print(e)
        sys.exit(2)
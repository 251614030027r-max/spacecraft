import os

# ================= 配置区域 =================
# 在这里填入你想要导出的文件夹路径或具体文件路径
# 可以是文件夹，也可以是单独的文件
TARGET_PATHS = [
    "docs",
    "dynamics",
    "env",
    "tests",
    "train",
    "DEVIATION_LOG.md",
    "README.md",
    "REPRODUCTION_SPEC.md",
    #"Train",
    # "data/dataset.py" # 你可以继续添加...

]

# 指定只导出这些后缀的文件
ALLOWED_EXTENSIONS = {'.py', '.yaml', '.json', '.sh', '.md'}

# 输出文件名
OUTPUT_FILE = "project_context.txt"


# ===========================================

def export_selected_files():
    print(f"准备导出，目标包含: {TARGET_PATHS}")

    count = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        # 写入头部说明
        outfile.write("Project Context (Selected Files):\n")
        outfile.write("The following content is a compilation of specific source files.\n\n")

        for path in TARGET_PATHS:
            # 1. 如果路径不存在
            if not os.path.exists(path):
                print(f"⚠️ 警告: 路径不存在，已跳过 -> {path}")
                continue

            # 2. 如果是单独的一个文件
            if os.path.isfile(path):
                _write_file_to_txt(path, outfile)
                count += 1

            # 3. 如果是一个文件夹
            elif os.path.isdir(path):
                print(f"正在扫描文件夹: {path} ...")
                for root, _, files in os.walk(path):
                    for file in files:
                        if os.path.splitext(file)[1] in ALLOWED_EXTENSIONS:
                            file_path = os.path.join(root, file)
                            # 避免把自己导出来
                            if file_path == os.path.basename(__file__) or file == OUTPUT_FILE:
                                continue
                            _write_file_to_txt(file_path, outfile)
                            count += 1

    print(f"\n✅ 成功！共合并了 {count} 个文件。")
    print(f"文件已保存至: {os.path.abspath(OUTPUT_FILE)}")


def _write_file_to_txt(file_path, outfile):
    """读取单个文件并写入输出流的辅助函数"""
    try:
        with open(file_path, "r", encoding="utf-8") as infile:
            content = infile.read()

            # 格式化输出，方便 AI 识别
            outfile.write(f"\n{'=' * 50}\n")
            outfile.write(f"==== FILE_PATH: {file_path} ====\n")
            outfile.write(f"{'=' * 50}\n\n")

            outfile.write(content)
            outfile.write("\n")
            print(f"  -> 已添加: {file_path}")
    except Exception as e:
        print(f"  ❌ 读取失败 {file_path}: {e}")


if __name__ == "__main__":
    export_selected_files()
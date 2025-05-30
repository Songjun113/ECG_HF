import openpyxl
import csv
from pathlib import Path

# 任务1：读取Excel数据到字典（显式处理中文）
def read_excel_to_dict_with_chinese(file_path, cell_range, columns):
    wb = openpyxl.load_workbook(file_path)
    sheet = wb.active
    data = {col: [] for col in columns}
    
    for row in sheet[cell_range]:
        for i, cell in enumerate(row):
            # 显式处理中文（openpyxl自动解码，无需额外操作）
            data[columns[i]].append(cell.value)
    
    return data

# 文件路径配置（建议使用原始字符串+pathlib）
input_path = Path(r"D:\three\专设\data\row_data\total_data.xlsx")
output_path = input_path.parent / "labels.csv"

# 读取数据（兼容中文列名和内容）
data = read_excel_to_dict_with_chinese(
    file_path=input_path,
    cell_range="C3:D186",
    columns=["name", "labels"]  # 列名建议用英文，避免后续编码问题
)

# 任务2：修改labels值（逻辑不变）
for i in range(len(data["labels"])):
    if data["labels"][i] in (0, 1):
        data["labels"][i] = -1
    elif data["labels"][i] in (2, 3):
        data["labels"][i] = 1
    else:
        data["labels"][i] = None  # 或按需处理异常值

# 任务3：导出CSV（关键修改：确保中文编码正确）
with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:  # 注意utf-8-sig
    writer = csv.writer(f)
    
    # 写入列名（如果列名需要中文，可直接修改）
    writer.writerow(["name", "labels"])  # 或 ["姓名", "标签"]
    
    # 逐行写入数据（处理中文内容）
    for name, label in zip(data["name"], data["labels"]):
        # 确保name是字符串（避免None报错）
        name_str = str(name) if name is not None else ""
        writer.writerow([name_str, label])

print(f"处理完成！文件已保存到：{output_path}")
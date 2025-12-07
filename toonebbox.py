import os
import re

input_folder = "/root/samurai/scripts/bbox_outputs/"
output_file = "/root/samurai/scripts/bbox_outputs/first_frame_bbox_all.txt"

def extract_number(filename):
    nums = re.findall(r'\d+', filename)
    return int(nums[-1]) if nums else -1

files = sorted(
    [f for f in os.listdir(input_folder) if f.endswith(".txt")],
    key=extract_number
)

with open(output_file, "w") as out:
    for f in files:
        with open(os.path.join(input_folder, f)) as ff:
            line = ff.readline().strip()
            out.write(line + "\n")

print("Merged:", output_file)
print("Order:", files)

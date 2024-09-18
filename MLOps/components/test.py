from pathlib import Path
import sys
from yaml import safe_load

'''
current_path = Path(__file__)
print(current_path)
#print(current_path.dirname)
print(current_path.parts)
print(current_path.parent)
print(current_path.stem)
print(current_path.suffix)
print(current_path.parent.parent)
print(sys.argv[0])
'''

# print(sys.version_info)
# print(sys.path)

# import platform

# # Get the operating system and version
# print(f"Operating system: {platform.system()} {platform.release()}")

# # Get more detailed platform information
# print(f"Platform: {platform.platform()}")

parameters = safe_load(open('params.yaml','r'))['data_ingestion']
print(f"Parameters: {parameters}")

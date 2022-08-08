from pathlib import Path

PARENT_DIR = Path('/home/vid/hdd/projects/PycharmProjects/FACEID_VMX/')
LANDMARKS_COLORS = [(0, 255, 0), (255, 0, 255), (255, 255, 255), (0, 255, 0), (255, 0, 255)]

det_thresh = 0.75  # 0.75
det_nms = 0.1  # 0.4
det_size = 1280  # 640  # 1280 or 960 баланс?
recog_tresh = 0.6  # 0.6 = office, 0.45 - zavod
bright_etalon = 100  # constant 150

from pathlib import Path

PARENT_DIR = Path(__file__).parent.parent  # FACEID_VMX dir
LANDMARKS_COLORS = [(0, 255, 0), (255, 0, 255), (255, 255, 255), (0, 255, 0), (255, 0, 255)]

det_thresh = 0.6  # 0.75
det_nms = 0.4  # 0.4
recog_tresh = 0.85  # 0.65  # 0.71
det_window_size = (1280, 1280)
bright_etalon = 130  # constant 150

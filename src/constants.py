from pathlib import Path

PARENT_DIR = Path(__file__).parent.parent  # FACEID_VMX dir
LANDMARKS_COLORS = [(0, 255, 0), (255, 0, 255), (255, 255, 255), (0, 255, 0), (255, 0, 255)]

det_thresh = 0.6  # 0.75
det_nms = 0.4  # 0.4
det_window_size = (1280, 1280)
recog_tresh = 0.6
bright_etalon = 100  # constant 150

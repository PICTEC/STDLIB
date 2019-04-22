import GPUtil
import os

def set_device():
    try:
        GPUtil.getFirstAvailable(attempts=3, interval=0.5)
    except (RuntimeError, ValueError):
        print("Switching to CPU only as GPU is busy or unavailable")
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

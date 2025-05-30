from dotenv import load_dotenv
import torch

load_dotenv()  # betölti a .env tartalmát az os.environ-be

def check_cuda():
    """
    Kiírja a CUDA elérhetőségét, verzióját és az aktuális device-ot.
    """
    print("CUDA elérhető:", torch.cuda.is_available())
    print("CUDA verzió:", torch.version.cuda)
    print("PyTorch build CUDA-támogatással:", torch.backends.cudnn.enabled)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Alapértelmezett eszköz:", device)
    return device
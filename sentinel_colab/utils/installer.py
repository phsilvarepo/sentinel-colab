# utils/installer.py
import subprocess
import sys

def install_package(package: str):
    """Install a package using pip."""
    print(f"📦 Installing: {package}")
    subprocess.run([sys.executable, "-m", "pip", "install", *package.split(), "-q"], check=True)

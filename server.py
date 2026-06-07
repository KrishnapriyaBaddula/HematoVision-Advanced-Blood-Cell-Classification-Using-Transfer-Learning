# run_server.py
import subprocess
import sys
import os

def check_and_install(package):
    """Check if package is installed, if not install it"""
    try:
        __import__(package)
        print(f"✅ {package} is installed")
        return True
    except ImportError:
        print(f"⚠️  {package} not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True

def main():
    print("\n" + "="*60)
    print("🩸 HemaToVision Setup & Launch Script")
    print("="*60 + "\n")
    
    # Required packages
    packages = ["fastapi", "uvicorn", "pillow", "numpy", "scipy"]
    
    print("Checking dependencies...\n")
    for package in packages:
        check_and_install(package)
    
    print("\n" + "="*60)
    print("🚀 Starting HemaToVision Server...")
    print("="*60 + "\n")
    
    # Change to backend directory and run the server
    backend_path = os.path.join(os.path.dirname(__file__), "backend")
    
    if not os.path.exists(backend_path):
        os.makedirs(backend_path)
        print(f"✅ Created backend folder at: {backend_path}")
    
    # Check if app.py exists in backend
    app_path = os.path.join(backend_path, "app.py")
    if not os.path.exists(app_path):
        print(f"❌ Error: app.py not found in {backend_path}")
        print("Please make sure backend/app.py exists")
        return
    
    # Run the server
    os.chdir(backend_path)
    subprocess.run([sys.executable, "app.py"])

if __name__ == "__main__":
    main()
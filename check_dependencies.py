"""
Dependency checker for CryptoGuard Backend
"""
import sys
import importlib.util

def check_module(module_name, package_name=None):
    """Check if a module is installed"""
    if package_name is None:
        package_name = module_name
    
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return False, f"✗ {package_name} is NOT installed"
    else:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'unknown')
            return True, f"✓ {package_name} is installed (version: {version})"
        except:
            return True, f"✓ {package_name} is installed (version: unknown)"

def main():
    print("=" * 60)
    print("CryptoGuard Backend - Dependency Checker")
    print("=" * 60)
    print()
    
    # Check Python version
    print(f"Python Version: {sys.version}")
    print()
    
    if sys.version_info < (3, 8):
        print("✗ ERROR: Python 3.8 or higher is required!")
        print("  Please upgrade Python")
        return False
    else:
        print("✓ Python version is compatible")
        print()
    
    # Required modules
    required_modules = [
        ('flask', 'Flask'),
        ('flask_cors', 'Flask-CORS'),
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'scikit-learn'),
        ('joblib', 'joblib'),
    ]
    
    print("Checking Required Dependencies:")
    print("-" * 60)
    
    all_installed = True
    missing_packages = []
    
    for module_name, package_name in required_modules:
        installed, message = check_module(module_name, package_name)
        print(message)
        if not installed:
            all_installed = False
            missing_packages.append(package_name)
    
    print("-" * 60)
    print()
    
    if all_installed:
        print("✓ All dependencies are installed!")
        print()
        print("You can now run the backend server:")
        print("  python app.py")
        print()
        return True
    else:
        print("✗ Some dependencies are missing!")
        print()
        print("To install missing packages, run:")
        print("  pip install -r requirements.txt --break-system-packages")
        print()
        print("Or install individually:")
        for package in missing_packages:
            print(f"  pip install {package.lower()} --break-system-packages")
        print()
        return False

if __name__ == "__main__":
    success = main()
    print("=" * 60)
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)

import os
import sys
from datetime import datetime

def check_requirements():
    """Check essential requirements"""
    print("🔍 Checking requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check essential modules
    required_modules = [
        'tkinter',
        'numpy', 
        'gymnasium',
        'datetime',
        'threading',
        'json'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} - OK")
        except ImportError:
            missing_modules.append(module)
            print(f"❌ {module} - MISSING")
    
    # Check optional but important modules
    optional_modules = {
        'MetaTrader5': 'MT5 integration',
        'stable_baselines3': 'RL algorithms',
        'pandas': 'Data processing',
        'matplotlib': 'Charts'
    }
    
    print("\n📋 Optional modules:")
    for module, description in optional_modules.items():
        try:
            __import__(module)
            print(f"✅ {module} - Available ({description})")
        except ImportError:
            print(f"⚠️ {module} - Not available ({description})")
    
    if missing_modules:
        print(f"\n❌ Missing required modules: {', '.join(missing_modules)}")
        print("Please install missing modules before running the application.")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        'core',
        'gui', 
        'models',
        'logs',
        'config',
        'data'
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created: {directory}/")
        except Exception as e:
            print(f"❌ Failed to create {directory}/: {e}")
            return False
    
    return True

def main():
    """Main application entry point"""
    try:
        print("🚀 Starting Simple AI Trading System...")
        print("=" * 50)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)
        
        # Check requirements
        if not check_requirements():
            print("\n❌ Requirements check failed!")
            input("Press Enter to exit...")
            sys.exit(1)
        
        # Create directories
        if not create_directories():
            print("\n❌ Directory creation failed!")
            input("Press Enter to exit...")
            sys.exit(1)
        
        print("\n" + "=" * 50)
        print("🎯 Initializing Simple Trading GUI...")
        
        # Import and run GUI
        try:
            from gui.minimal_gui import MinimalGUI
            
            print("✅ GUI module imported successfully")
            
            # Create and run application
            app = MinimalGUI()
            print("✅ GUI initialized successfully")
            
            print("🚀 Starting application...")
            print("=" * 50)
            
            app.run()
            
        except ImportError as e:
            print(f"❌ GUI import error: {e}")
            print("Make sure gui/minimal_gui.py exists and is properly configured")
            input("Press Enter to exit...")
            sys.exit(1)
            
        except Exception as e:
            print(f"❌ GUI initialization error: {e}")
            import traceback
            traceback.print_exc()
            input("Press Enter to exit...")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⏹️ Application interrupted by user")
        print("👋 Goodbye!")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to show error in GUI if possible
        try:
            import tkinter as tk
            from tkinter import messagebox
            
            root = tk.Tk()
            root.withdraw()  # Hide main window
            
            messagebox.showerror(
                "Startup Error", 
                f"Failed to start AI Trading System:\n\n{str(e)}\n\n"
                "Check console for detailed error information."
            )
            
        except:
            pass
        
        input("Press Enter to exit...")
        sys.exit(1)
    
    finally:
        print("\n🧹 System shutdown complete")
        print("Thank you for using Simple AI Trading System! 🚀")

if __name__ == "__main__":
    main()
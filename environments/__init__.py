# environments/__init__.py - Environment Package

"""
🏗️ Trading Environments Package

ระบบ Environment แยกตาม Trading Mode:
- ConservativeEnvironment: ใช้ RL Agent เดิม (HOLD บ่อย, ปลอดภัย)
- AggressiveEnvironment: ใช้ AI Recovery Brain (เทรดมาก, แก้ไม้อัจฉริยะ)

Usage:
    from environments import ConservativeEnvironment, AggressiveEnvironment
    
    # Conservative Mode
    env = ConservativeEnvironment(mt5_interface, config)
    
    # Aggressive Mode  
    env = AggressiveEnvironment(mt5_interface, config)
"""

# Import environments
try:
    from .conservative_env import ConservativeEnvironment
    print("✅ ConservativeEnvironment loaded")
except ImportError as e:
    print(f"❌ ConservativeEnvironment import error: {e}")
    ConservativeEnvironment = None

try:
    from .aggressive_env import AggressiveEnvironment
    print("✅ AggressiveEnvironment loaded")
except ImportError as e:
    print(f"❌ AggressiveEnvironment import error: {e}")
    AggressiveEnvironment = None

# Available environments
__all__ = [
    'ConservativeEnvironment',
    'AggressiveEnvironment'
]

# Environment factory
def create_environment(mode: str, mt5_interface, config, historical_data=None):
    """
    🏭 Environment Factory - สร้าง environment ตาม mode
    
    Args:
        mode: 'conservative' หรือ 'aggressive'
        mt5_interface: MT5 connector
        config: Configuration dict
        historical_data: Historical data for training
    
    Returns:
        Environment instance
    """
    try:
        if mode.lower() == 'conservative':
            if ConservativeEnvironment is None:
                raise ImportError("ConservativeEnvironment not available")
            return ConservativeEnvironment(mt5_interface, config, historical_data)
            
        elif mode.lower() == 'aggressive':
            if AggressiveEnvironment is None:
                raise ImportError("AggressiveEnvironment not available")
            return AggressiveEnvironment(mt5_interface, config, historical_data)
            
        else:
            raise ValueError(f"Unknown environment mode: {mode}")
            
    except Exception as e:
        print(f"❌ Environment creation error: {e}")
        # Fallback to conservative if available
        if ConservativeEnvironment is not None:
            print("🛡️ Falling back to ConservativeEnvironment")
            return ConservativeEnvironment(mt5_interface, config, historical_data)
        else:
            raise e

# Version info
__version__ = "1.0.0"
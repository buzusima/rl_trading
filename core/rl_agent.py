# core/rl_agent.py - แก้ไข model save/load issues

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Any

# Stable-Baselines3 imports with error handling
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
    print("✅ Stable-Baselines3 available")
except ImportError as e:
    print(f"❌ Stable-Baselines3 not available: {e}")
    print("Install with: pip install stable-baselines3")
    SB3_AVAILABLE = False

class RLAgent:
    """
    RL Agent for Trading
    - PPO algorithm only
    - Training and prediction
    - Fixed model save/load
    """
    
    def __init__(self, environment, config: Dict = None):
        print("🤖 Initializing RL Agent...")
        
        # Core components
        self.env = environment
        self.config = config or {}
        
        # Check if SB3 is available
        if not SB3_AVAILABLE:
            print("❌ Cannot initialize agent without Stable-Baselines3")
            self.model = None
            return
        
        # Agent configuration
        self.algorithm = 'PPO'
        self.learning_rate = self.config.get('learning_rate', 0.0003)
        self.batch_size = self.config.get('batch_size', 64)
        self.n_steps = self.config.get('n_steps', 2048)
        self.n_epochs = self.config.get('n_epochs', 10)
        
        # Training parameters
        self.training_steps = self.config.get('training_steps', 10000)
        self.gamma = self.config.get('gamma', 0.99)
        self.gae_lambda = self.config.get('gae_lambda', 0.95)
        self.clip_range = self.config.get('clip_range', 0.2)
        
        # Model state
        self.model = None
        self.vec_env = None  # เก็บ vectorized environment
        self.is_trained = False
        self.total_timesteps_trained = 0
        
        # Training tracking
        self.training_history = []
        self.episode_rewards = []
        self.training_start_time = None
        
        # File paths - แก้ไข: ให้ชัดเจน
        self.model_save_path = os.path.abspath('models')
        self.log_path = os.path.abspath('logs')
        
        # Create directories
        self._create_directories()
        
        # Initialize model
        self._initialize_model()
        
        print("✅ RL Agent initialized:")
        print(f"   - Algorithm: {self.algorithm}")
        print(f"   - Learning Rate: {self.learning_rate}")
        print(f"   - Training Steps: {self.training_steps}")
        print(f"   - Save Path: {self.model_save_path}")
        print(f"   - Model Ready: {self.model is not None}")

    def _create_directories(self):
        """Create necessary directories"""
        for path in [self.model_save_path, self.log_path]:
            try:
                os.makedirs(path, exist_ok=True)
                print(f"✅ Directory ready: {path}")
            except Exception as e:
                print(f"❌ Failed to create {path}: {e}")
                raise e

    def _initialize_model(self):
        """Initialize PPO model"""
        try:
            if not SB3_AVAILABLE:
                print("❌ Cannot initialize model - SB3 not available")
                return
            
            # Validate environment
            self._validate_environment()
            
            # Create vectorized environment - เก็บไว้ใช้ทั้ง train และ load
            def make_env():
                return Monitor(self.env, self.log_path)
            
            self.vec_env = DummyVecEnv([make_env])
            
            # Create PPO model
            import torch.nn as nn
            
            self.model = PPO(
                policy='MlpPolicy',
                env=self.vec_env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                device='auto',
                verbose=1,
                tensorboard_log=None,
                policy_kwargs={
                    'net_arch': [256, 256],
                    'activation_fn': nn.Tanh
                }
            )
            
            print("✅ PPO model initialized successfully")
            
        except Exception as e:
            print(f"❌ Model initialization error: {e}")
            self.model = None

    def _validate_environment(self):
        """Validate environment compatibility"""
        try:
            obs_shape = self.env.observation_space.shape
            action_shape = self.env.action_space.shape
            
            print(f"🔍 Environment Validation:")
            print(f"   - Observation Shape: {obs_shape}")
            print(f"   - Action Shape: {action_shape}")
            
            # Test environment reset
            obs, info = self.env.reset()
            print(f"   - Reset Test: ✅ Success")
            
            # Test environment step
            valid_action = np.array([0, 0.01, 0.0], dtype=np.float32)
            obs, reward, done, truncated, info = self.env.step(valid_action)
            print(f"   - Step Test: ✅ Success")
            
        except Exception as e:
            print(f"❌ Environment validation failed: {e}")
            raise e

    def train(self, total_timesteps: int = None):
        """Train the RL agent"""
        try:
            if self.model is None:
                print("❌ No model to train - initialization failed")
                return False
                
            timesteps = total_timesteps or self.training_steps
            print(f"🎓 Training PPO agent for {timesteps} timesteps...")
            
            self.training_start_time = datetime.now()
            self.model.learn(total_timesteps=timesteps, progress_bar=True)
            
            training_time = (datetime.now() - self.training_start_time).total_seconds()
            print(f"✅ Training completed in {training_time:.1f} seconds")
            
            self.is_trained = True
            self.total_timesteps_trained += timesteps
            return True
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            return False

    def predict(self, observation):
        """Predict action from observation"""
        try:
            if self.model is None:
                print("❌ No model available for prediction")
                return self.env.action_space.sample()
                
            action, _ = self.model.predict(observation, deterministic=True)
            return action
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return self.env.action_space.sample()

    def save_model(self, filename: str = None):
        """Save trained model - แก้ไข: handle .zip extension properly"""
        try:
            if self.model is None:
                print("❌ No model to save")
                return None
                
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ppo_model_{timestamp}"
                
            # แก้ไข: ให้แน่ใจว่า filename ไม่มี .zip (SB3 จะเพิ่มให้อัตโนมัติ)
            if filename.endswith('.zip'):
                filename = filename[:-4]
                
            filepath = os.path.join(self.model_save_path, filename)
            
            # Save model (SB3 จะเพิ่ม .zip อัตโนมัติ)
            self.model.save(filepath)
            
            # Check if file was actually created
            actual_filepath = filepath + '.zip'
            if os.path.exists(actual_filepath):
                print(f"✅ Model saved: {actual_filepath}")
                return actual_filepath
            else:
                print(f"❌ Model save failed - file not found: {actual_filepath}")
                return None
            
        except Exception as e:
            print(f"❌ Save model error: {e}")
            return None

    def load_model(self, filename: str = None):
        """Load trained model - แก้ไข: handle paths and environment properly"""
        try:
            if not SB3_AVAILABLE:
                print("❌ Cannot load model - SB3 not available")
                return False
                
            # แก้ไข: หา model file ที่ถูกต้อง
            if filename is None:
                # Find latest model
                if not os.path.exists(self.model_save_path):
                    print(f"❌ Model directory not found: {self.model_save_path}")
                    return False
                    
                model_files = [f for f in os.listdir(self.model_save_path) 
                              if f.endswith('.zip') and f.startswith('ppo_model_')]
                if not model_files:
                    print(f"❌ No saved models found in {self.model_save_path}")
                    return False
                    
                # Get latest file by modification time
                model_files_with_time = []
                for f in model_files:
                    filepath = os.path.join(self.model_save_path, f)
                    mtime = os.path.getmtime(filepath)
                    model_files_with_time.append((mtime, f))
                
                model_files_with_time.sort(reverse=True)
                filename = model_files_with_time[0][1]
                print(f"🔍 Using latest model: {filename}")
            
            # แก้ไข: สร้าง full path ที่ถูกต้อง
            if not filename.endswith('.zip'):
                filename += '.zip'
                
            filepath = os.path.join(self.model_save_path, filename)
            
            if not os.path.exists(filepath):
                print(f"❌ Model file not found: {filepath}")
                return False
                
            # แก้ไข: ใช้ vec_env เดิมที่สร้างตอน init
            if self.vec_env is None:
                print("❌ Vectorized environment not available")
                return False
                
            # Load model with correct environment
            self.model = PPO.load(filepath, env=self.vec_env)
            print(f"✅ Model loaded: {filepath}")
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"❌ Load model error: {e}")
            return False

    def list_saved_models(self):
        """List all saved models"""
        try:
            if not os.path.exists(self.model_save_path):
                print(f"❌ Model directory not found: {self.model_save_path}")
                return []
                
            model_files = [f for f in os.listdir(self.model_save_path) 
                          if f.endswith('.zip') and f.startswith('ppo_model_')]
            
            if not model_files:
                print(f"❌ No saved models found in {self.model_save_path}")
                return []
            
            # Sort by modification time (newest first)
            model_info = []
            for f in model_files:
                filepath = os.path.join(self.model_save_path, f)
                mtime = os.path.getmtime(filepath)
                size = os.path.getsize(filepath)
                model_info.append({
                    'filename': f,
                    'path': filepath,
                    'modified': datetime.fromtimestamp(mtime),
                    'size_mb': size / (1024 * 1024)
                })
            
            model_info.sort(key=lambda x: x['modified'], reverse=True)
            
            print(f"📋 Found {len(model_info)} saved models:")
            for i, info in enumerate(model_info):
                print(f"   {i+1}. {info['filename']} ({info['size_mb']:.1f}MB) - {info['modified']}")
                
            return model_info
            
        except Exception as e:
            print(f"❌ List models error: {e}")
            return []

    def get_model_info(self):
        """Get current model information"""
        if self.model is None:
            return {"status": "No model loaded"}
            
        return {
            "status": "Model loaded",
            "algorithm": self.algorithm,
            "is_trained": self.is_trained,
            "total_timesteps": self.total_timesteps_trained,
            "learning_rate": self.learning_rate,
            "policy": str(type(self.model.policy)),
            "observation_space": str(self.env.observation_space.shape),
            "action_space": str(self.env.action_space.shape)
        }

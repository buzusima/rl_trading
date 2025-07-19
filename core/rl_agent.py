import os
from typing import Dict

# Stable-Baselines3 imports with error handling
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
    print("✅ Stable-Baselines3 available")
except ImportError as e:
    print(f"❌ Stable-Baselines3 not available: {e}")
    print("Install with: pip install stable-baselines3")
    SB3_AVAILABLE = False

class RLAgent:
    """
    Simplified RL Agent for Trading
    - PPO algorithm only
    - Training and prediction
    - Essential model management
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
        self.algorithm = 'PPO'  # Only PPO
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
        self.is_trained = False
        self.total_timesteps_trained = 0
        
        # Training tracking
        self.training_history = []
        self.episode_rewards = []
        self.training_start_time = None
        
        # File paths
        self.model_save_path = 'models/'
        self.log_path = 'logs/'
        
        # Create directories
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        
        # Initialize model
        self._initialize_model()
        
        print("✅ RL Agent initialized:")
        print(f"   - Algorithm: {self.algorithm}")
        print(f"   - Learning Rate: {self.learning_rate}")
        print(f"   - Training Steps: {self.training_steps}")
        print(f"   - Observation Space: {self.env.observation_space.shape}")
        print(f"   - Action Space: {self.env.action_space.shape}")
        print(f"   - Model Ready: {self.model is not None}")

    def _initialize_model(self):
        """Initialize PPO model"""
        try:
            if not SB3_AVAILABLE:
                print("❌ Cannot initialize model - SB3 not available")
                return
            
            # Validate environment
            self._validate_environment()
            
            # Create vectorized environment
            def make_env():
                return Monitor(self.env, self.log_path)
            
            vec_env = DummyVecEnv([make_env])
            
            # Create PPO model - FIX: import torch functions properly
            import torch.nn as nn
            
            self.model = PPO(
                policy='MlpPolicy',
                env=vec_env,
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
                tensorboard_log=self.log_path,
                policy_kwargs={
                    'net_arch': [256, 256],  # Simple network
                    'activation_fn': nn.Tanh  # FIX: Use nn.Tanh instead of 'tanh'
                }
            )
            
            print("✅ PPO model initialized successfully")
            
        except Exception as e:
            print(f"❌ Model initialization error: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    def _validate_environment(self):
        """Validate environment compatibility"""
        try:
            # Check observation and action spaces
            obs_shape = self.env.observation_space.shape
            action_shape = self.env.action_space.shape
            
            print(f"🔍 Environment Validation:")
            print(f"   - Observation Shape: {obs_shape}")
            print(f"   - Action Shape: {action_shape}")
            
            # Test environment reset
            obs, info = self.env.reset()
            print(f"   - Reset Test: ✅ Success")
            print(f"   - Sample Observation Shape: {obs.shape}")
            
            # Test environment step
            random_action = self.env.action_space.sample()
            obs, reward, done, truncated, info = self.env.step(random_action)
            print(f"   - Step Test: ✅ Success")
            print(f"   - Sample Reward: {reward}")
            
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
            import traceback
            traceback.print_exc()
            return False

    def predict(self, observation):
        """Predict action from observation"""
        try:
            if self.model is None:
                print("❌ No model available for prediction")
                return self.env.action_space.sample()  # Random action
                
            action, _ = self.model.predict(observation, deterministic=True)
            return action
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return self.env.action_space.sample()

    def save_model(self, filename: str = None):
        """Save trained model"""
        try:
            if self.model is None:
                print("❌ No model to save")
                return None
                
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ppo_model_{timestamp}"
                
            filepath = os.path.join(self.model_save_path, filename)
            self.model.save(filepath)
            print(f"✅ Model saved: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"❌ Save model error: {e}")
            return None

    def load_model(self, filename: str = None):
        """Load trained model"""
        try:
            if filename is None:
                # Find latest model
                model_files = [f for f in os.listdir(self.model_save_path) if f.endswith('.zip')]
                if not model_files:
                    print("❌ No saved models found")
                    return False
                filename = max(model_files)
                
            filepath = os.path.join(self.model_save_path, filename)
            if not os.path.exists(filepath):
                print(f"❌ Model file not found: {filepath}")
                return False
                
            self.model = PPO.load(filepath, env=DummyVecEnv([lambda: self.env]))
            print(f"✅ Model loaded: {filepath}")
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"❌ Load model error: {e}")
            return False
        
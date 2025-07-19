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

class BasicRLAgent:
    """
    Simplified RL Agent for Trading
    - PPO algorithm only
    - Basic training and prediction
    - Essential model management
    """
    
    def __init__(self, environment, config: Dict = None):
        print("🤖 Initializing Basic RL Agent...")
        
        # Core components
        self.env = environment
        self.config = config or {}
        
        # Check if SB3 is available
        if not SB3_AVAILABLE:
            print("❌ Cannot initialize agent without Stable-Baselines3")
            self.model = None
            return
        
        # Basic agent configuration
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
        
        print("✅ Basic RL Agent initialized:")
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
            
            # Create PPO model
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
                    'activation_fn': 'tanh'
                }
            )
            
            print("✅ PPO model initialized successfully")
            
        except Exception as e:
            print(f"❌ Model initialization error: {e}")
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
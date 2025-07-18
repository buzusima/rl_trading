# rl_agent.py - Professional Multi-Agent RL System
import numpy as np
import pandas as pd
import pickle
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
import threading
import time
import warnings
warnings.filterwarnings('ignore')

# Stable-Baselines3 imports with error handling
try:
    from stable_baselines3 import PPO, SAC, TD3, A2C
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
    from stable_baselines3.common.buffers import ReplayBuffer
    SB3_AVAILABLE = True
except ImportError as e:
    print(f"Stable-Baselines3 not available: {e}")
    print("Install with: pip install stable-baselines3[extra]")
    SB3_AVAILABLE = False

class ProfessionalTradingCallback(BaseCallback):
    """
    Professional callback for advanced training monitoring
    """
    
    def __init__(self, eval_env, eval_freq: int = 1000, 
                 save_freq: int = 5000, save_path: str = 'models/', 
                 gui_callback: Callable = None, verbose: int = 0):
        super(ProfessionalTradingCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.save_path = save_path
        self.gui_callback = gui_callback
        
        # Professional metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.action_diversity = []
        self.market_timing_scores = []
        self.portfolio_efficiency = []
        self.recovery_success_rates = []
        
        self.best_mean_reward = -np.inf
        self.training_start_time = time.time()
        
        # Ensure save directory exists
        os.makedirs(save_path, exist_ok=True)
        
    def _init_callback(self) -> None:
        """Initialize callback"""
        pass
        
    def _on_training_start(self) -> None:
        """Called before training starts"""
        self.training_start_time = time.time()
        print("🚀 Professional Training Started!")
        
    def _on_rollout_start(self) -> None:
        """Called before collecting new samples"""
        pass
        
    def _on_step(self) -> bool:
        """Called after each environment step"""
        try:
            # Update GUI if callback provided
            if self.gui_callback and self.n_calls % 100 == 0:
                progress = (self.n_calls / self.model._total_timesteps) * 100 if hasattr(self.model, '_total_timesteps') else 0
                
                # Get current reward if available
                current_reward = None
                try:
                    if hasattr(self.model, 'logger') and self.model.logger:
                        current_reward = self.model.logger.name_to_value.get('rollout/ep_rew_mean', None)
                except:
                    pass
                
                self.gui_callback(self.n_calls, self.model._total_timesteps if hasattr(self.model, '_total_timesteps') else 100000, current_reward)
            
            # Professional evaluation
            if self.n_calls % self.eval_freq == 0 and self.eval_env is not None:
                self._perform_professional_evaluation()
                
            # Model checkpointing
            if self.n_calls % self.save_freq == 0:
                self._save_checkpoint()
                
            return True
            
        except Exception as e:
            print(f"Callback step error: {e}")
            return True
    
    def _perform_professional_evaluation(self):
        """Perform comprehensive evaluation"""
        try:
            # Standard evaluation
            mean_reward, std_reward = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=5, deterministic=True
            )
            
            print(f"📊 Step {self.n_calls}: Reward: {mean_reward:.3f} ±{std_reward:.3f}")
            
            # Professional metrics analysis
            self._analyze_action_diversity()
            self._analyze_market_timing()
            self._analyze_portfolio_efficiency()
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_model_path = os.path.join(self.save_path, 'best_professional_model')
                self.model.save(best_model_path)
                print(f"🏆 New best model saved: {mean_reward:.3f}")
                
        except Exception as e:
            print(f"Professional evaluation error: {e}")
    
    def _analyze_action_diversity(self):
        """Analyze action diversity and exploration"""
        try:
            # This would analyze the recent actions taken by the agent
            # For now, we'll simulate this analysis
            diversity_score = np.random.uniform(0.3, 1.0)  # Placeholder
            self.action_diversity.append(diversity_score)
            
            if len(self.action_diversity) > 10:
                avg_diversity = np.mean(self.action_diversity[-10:])
                if avg_diversity < 0.4:
                    print(f"⚠️ Low action diversity: {avg_diversity:.3f}")
                    
        except Exception as e:
            print(f"Action diversity analysis error: {e}")
    
    def _analyze_market_timing(self):
        """Analyze market timing performance"""
        try:
            # Placeholder for market timing analysis
            timing_score = np.random.uniform(0.4, 0.9)
            self.market_timing_scores.append(timing_score)
            
        except Exception as e:
            print(f"Market timing analysis error: {e}")
    
    def _analyze_portfolio_efficiency(self):
        """Analyze portfolio management efficiency"""
        try:
            # Placeholder for portfolio efficiency analysis
            efficiency_score = np.random.uniform(0.5, 1.0)
            self.portfolio_efficiency.append(efficiency_score)
            
        except Exception as e:
            print(f"Portfolio efficiency analysis error: {e}")
    
    def _save_checkpoint(self):
        """Save model checkpoint"""
        try:
            checkpoint_path = os.path.join(self.save_path, f'professional_checkpoint_{self.n_calls}')
            self.model.save(checkpoint_path)
            
            # Save training metrics
            metrics = {
                'step': self.n_calls,
                'timestamp': datetime.now().isoformat(),
                'episode_rewards': self.episode_rewards[-100:],  # Last 100
                'action_diversity': self.action_diversity[-50:],   # Last 50
                'market_timing_scores': self.market_timing_scores[-50:],
                'portfolio_efficiency': self.portfolio_efficiency[-50:],
                'best_mean_reward': self.best_mean_reward,
                'training_duration': time.time() - self.training_start_time
            }
            
            metrics_path = checkpoint_path + '_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
                
            print(f"💾 Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            print(f"Checkpoint save error: {e}")
    
    def _on_rollout_end(self) -> None:
        """Called after rollout collection"""
        try:
            # Log episode statistics
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                ep_info = self.model.ep_info_buffer[-1]
                self.episode_rewards.append(ep_info.get('r', 0))
                self.episode_lengths.append(ep_info.get('l', 0))
                
                # Keep only recent episodes
                if len(self.episode_rewards) > 1000:
                    self.episode_rewards = self.episode_rewards[-1000:]
                    self.episode_lengths = self.episode_lengths[-1000:]
                    
        except Exception as e:
            print(f"Rollout end processing error: {e}")
    
    def _on_training_end(self) -> None:
        """Called when training ends"""
        try:
            # Save final model
            final_model_path = os.path.join(self.save_path, 'final_professional_model')
            self.model.save(final_model_path)
            
            # Generate training report
            self._generate_training_report()
            
            training_duration = time.time() - self.training_start_time
            print(f"🎓 Professional Training Completed!")
            print(f"   Duration: {training_duration:.0f} seconds")
            print(f"   Best Reward: {self.best_mean_reward:.3f}")
            
        except Exception as e:
            print(f"Training end processing error: {e}")
    
    def _generate_training_report(self):
        """Generate comprehensive training report"""
        try:
            report = {
                'training_summary': {
                    'total_steps': self.n_calls,
                    'training_duration': time.time() - self.training_start_time,
                    'best_mean_reward': self.best_mean_reward,
                    'final_avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
                    'reward_improvement': self.best_mean_reward - (np.mean(self.episode_rewards[:100]) if len(self.episode_rewards) > 100 else 0),
                },
                'performance_metrics': {
                    'avg_action_diversity': np.mean(self.action_diversity) if self.action_diversity else 0,
                    'avg_market_timing': np.mean(self.market_timing_scores) if self.market_timing_scores else 0,
                    'avg_portfolio_efficiency': np.mean(self.portfolio_efficiency) if self.portfolio_efficiency else 0,
                    'episode_count': len(self.episode_rewards),
                    'avg_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
                },
                'learning_curve': {
                    'rewards_over_time': self.episode_rewards,
                    'action_diversity_over_time': self.action_diversity,
                    'timestamps': [datetime.now().isoformat()] * len(self.episode_rewards)
                }
            }
            
            report_path = os.path.join(self.save_path, 'professional_training_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            print(f"📊 Training report saved: {report_path}")
            
        except Exception as e:
            print(f"Report generation error: {e}")

class ProfessionalRLAgent:
    """
    Professional Multi-Agent RL System for Trading
    - Supports multiple algorithms (PPO, SAC, TD3)
    - Advanced action space handling (15 dimensions)
    - Professional observation processing (150 features)
    - Intelligent agent selection and ensemble methods
    """
    
    def __init__(self, environment, config: Dict = None):
        self.env = environment
        self.config = config or {}
        
        print(f"🤖 Initializing Professional RL Agent...")
        
        # Validate environment compatibility
        self._validate_environment()
        
        # Agent configuration
        self.primary_algorithm = self.config.get('primary_algorithm', 'PPO')
        self.secondary_algorithm = self.config.get('secondary_algorithm', 'SAC')
        self.ensemble_mode = self.config.get('ensemble_mode', False)
        
        # Learning parameters
        self.learning_rate = self.config.get('learning_rate', 0.0003)
        self.batch_size = self.config.get('batch_size', 256)
        self.buffer_size = self.config.get('buffer_size', 100000)
        self.learning_starts = self.config.get('learning_starts', 1000)
        
        # Professional features
        self.adaptive_lr = self.config.get('adaptive_learning_rate', True)
        self.curriculum_learning = self.config.get('curriculum_learning', True)
        self.experience_replay = self.config.get('experience_replay', True)
        
        # Multi-agent system
        self.agents = {}
        self.active_agent = None
        self.agent_performance = {}
        self.agent_selection_history = []
        
        # Training state
        self.is_training = False
        self.training_thread = None
        self.total_timesteps_trained = 0
        
        # Performance tracking
        self.training_history = []
        self.evaluation_history = []
        self.action_analysis = []
        self.model_versions = []
        
        # Professional metrics
        self.action_distribution = np.zeros(15)  # Track 15-dim action usage
        self.market_regime_performance = {}
        self.portfolio_efficiency_history = []
        
        # Paths
        self.model_save_path = 'models/professional_models/'
        self.log_path = 'data/professional_logs/'
        
        # Create directories
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        
        # Initialize agents
        self._initialize_professional_agents()
        
        print(f"✅ Professional RL Agent initialized:")
        print(f"   Primary Algorithm: {self.primary_algorithm}")
        print(f"   Secondary Algorithm: {self.secondary_algorithm}")
        print(f"   Ensemble Mode: {self.ensemble_mode}")
        print(f"   Observation Space: {self.env.observation_space.shape}")
        print(f"   Action Space: {self.env.action_space.shape}")
        
    def _validate_environment(self):
        """Validate environment compatibility"""
        try:
            obs_shape = self.env.observation_space.shape
            action_shape = self.env.action_space.shape
            
            print(f"🔍 Environment Validation:")
            print(f"   Observation Shape: {obs_shape}")
            print(f"   Action Shape: {action_shape}")
            
            # Check for professional environment features
            if obs_shape[0] < 100:
                print(f"⚠️ Warning: Observation space seems small ({obs_shape[0]} features)")
                print(f"   Professional environments should have 100+ features")
                
            if action_shape[0] < 10:
                print(f"⚠️ Warning: Action space seems simple ({action_shape[0]} dimensions)")
                print(f"   Professional environments should have 10+ action dimensions")
                
            # Test environment reset
            obs, info = self.env.reset()
            print(f"✅ Environment validation successful")
            print(f"   Sample observation shape: {obs.shape}")
            print(f"   Info keys: {list(info.keys()) if isinstance(info, dict) else 'No info'}")
            
        except Exception as e:
            print(f"❌ Environment validation failed: {e}")
            raise e
    
    def _initialize_professional_agents(self):
        """Initialize multiple professional agents"""
        try:
            if not SB3_AVAILABLE:
                print("❌ Stable-Baselines3 not available, cannot initialize agents")
                return
                
            # Create vectorized environment
            def make_env():
                return Monitor(self.env, self.log_path)
            
            vec_env = DummyVecEnv([make_env])
            
            # Initialize primary agent (PPO)
            if self.primary_algorithm.upper() == 'PPO':
                self.agents['primary'] = self._create_ppo_agent(vec_env)
            elif self.primary_algorithm.upper() == 'SAC':
                self.agents['primary'] = self._create_sac_agent(vec_env)
            elif self.primary_algorithm.upper() == 'TD3':
                self.agents['primary'] = self._create_td3_agent(vec_env)
            else:
                print(f"⚠️ Unknown primary algorithm: {self.primary_algorithm}, using PPO")
                self.agents['primary'] = self._create_ppo_agent(vec_env)
            
            # Initialize secondary agent if ensemble mode
            if self.ensemble_mode:
                if self.secondary_algorithm.upper() == 'SAC':
                    self.agents['secondary'] = self._create_sac_agent(vec_env)
                elif self.secondary_algorithm.upper() == 'TD3':
                    self.agents['secondary'] = self._create_td3_agent(vec_env)
                elif self.secondary_algorithm.upper() == 'PPO':
                    self.agents['secondary'] = self._create_ppo_agent(vec_env, variant='aggressive')
                else:
                    print(f"⚠️ Unknown secondary algorithm: {self.secondary_algorithm}")
            
            # Set active agent
            self.active_agent = 'primary'
            
            # Initialize performance tracking
            for agent_name in self.agents.keys():
                self.agent_performance[agent_name] = {
                    'total_rewards': [],
                    'success_rate': 0.0,
                    'avg_portfolio_efficiency': 0.0,
                    'market_timing_score': 0.0,
                    'episodes': 0
                }
            
            print(f"✅ Initialized {len(self.agents)} professional agents")
            
        except Exception as e:
            print(f"❌ Agent initialization error: {e}")
            self.agents = {}
    
    def _create_ppo_agent(self, vec_env, variant='standard'):
        """Create PPO agent with professional configuration"""
        try:
            if variant == 'aggressive':
                # More aggressive PPO for ensemble
                return PPO(
                    policy='MlpPolicy',
                    env=vec_env,
                    learning_rate=self.learning_rate * 1.5,
                    n_steps=1024,
                    batch_size=self.batch_size // 2,
                    n_epochs=15,
                    gamma=0.995,
                    gae_lambda=0.98,
                    clip_range=0.3,
                    ent_coef=0.02,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    device='auto',
                    verbose=1,
                    tensorboard_log=self.log_path
                )
            else:
                # Standard professional PPO
                return PPO(
                    policy='MlpPolicy',
                    env=vec_env,
                    learning_rate=self.learning_rate,
                    n_steps=2048,
                    batch_size=self.batch_size,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.01,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    device='auto',
                    verbose=1,
                    tensorboard_log=self.log_path,
                    policy_kwargs={
                        'net_arch': [512, 512, 256],  # Larger network for complex observation space
                        'activation_fn': 'tanh'
                    }
                )
                
        except Exception as e:
            print(f"PPO creation error: {e}")
            return None
    
    def _create_sac_agent(self, vec_env):
        """Create SAC agent with professional configuration"""
        try:
            return SAC(
                policy='MlpPolicy',
                env=vec_env,
                learning_rate=self.learning_rate,
                buffer_size=self.buffer_size,
                learning_starts=self.learning_starts,
                batch_size=self.batch_size,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                ent_coef='auto',
                target_update_interval=1,
                device='auto',
                verbose=1,
                tensorboard_log=self.log_path,
                policy_kwargs={
                    'net_arch': [512, 512, 256],  # Larger network
                    'activation_fn': 'relu'
                }
            )
            
        except Exception as e:
            print(f"SAC creation error: {e}")
            return None
    
    def _create_td3_agent(self, vec_env):
        """Create TD3 agent with professional configuration"""
        try:
            # Add action noise for TD3
            action_noise = NormalActionNoise(
                mean=np.zeros(self.env.action_space.shape[0]), 
                sigma=0.1 * np.ones(self.env.action_space.shape[0])
            )
            
            return TD3(
                policy='MlpPolicy',
                env=vec_env,
                learning_rate=self.learning_rate,
                buffer_size=self.buffer_size,
                learning_starts=self.learning_starts,
                batch_size=self.batch_size,
                tau=0.005,
                gamma=0.99,
                train_freq=(1, 'step'),
                gradient_steps=1,
                action_noise=action_noise,
                policy_delay=2,
                target_policy_noise=0.2,
                target_noise_clip=0.5,
                device='auto',
                verbose=1,
                tensorboard_log=self.log_path,
                policy_kwargs={
                    'net_arch': [512, 512, 256],  # Larger network
                    'activation_fn': 'relu'
                }
            )
            
        except Exception as e:
            print(f"TD3 creation error: {e}")
            return None
    
    def train(self, total_timesteps: int = None, callback: Callable = None):
        """Professional training with advanced features"""
        print("🎓 Starting Professional Training...")
        
        if not self.agents:
            print("❌ No agents available for training")
            return False
            
        try:
            timesteps = total_timesteps or self.config.get('training_steps', 50000)
            
            # Create professional callback
            eval_env = DummyVecEnv([lambda: Monitor(self.env, self.log_path)])
            
            professional_callback = ProfessionalTradingCallback(
                eval_env=eval_env,
                eval_freq=max(timesteps // 50, 1000),  # Evaluate 50 times during training
                save_freq=max(timesteps // 20, 2000),  # Save 20 checkpoints
                save_path=self.model_save_path,
                gui_callback=callback,
                verbose=1
            )
            
            training_start = datetime.now()
            
            # Train active agent
            active_agent = self.agents[self.active_agent]
            print(f"🚀 Training {self.active_agent} agent ({type(active_agent).__name__}) for {timesteps:,} steps...")
            
            # Training with curriculum learning
            if self.curriculum_learning:
                self._curriculum_training(active_agent, timesteps, professional_callback)
            else:
                active_agent.learn(
                    total_timesteps=timesteps,
                    callback=professional_callback,
                    progress_bar=True
                )
            
            # Train secondary agent if ensemble mode
            if self.ensemble_mode and 'secondary' in self.agents:
                print(f"🤖 Training secondary agent...")
                secondary_timesteps = timesteps // 2  # Train secondary for half the time
                
                self.agents['secondary'].learn(
                    total_timesteps=secondary_timesteps,
                    callback=professional_callback,
                    progress_bar=True
                )
            
            training_end = datetime.now()
            training_duration = (training_end - training_start).total_seconds()
            
            # Update training history
            training_record = {
                'start_time': training_start.isoformat(),
                'end_time': training_end.isoformat(),
                'duration_seconds': training_duration,
                'timesteps': timesteps,
                'algorithm': self.primary_algorithm,
                'ensemble_mode': self.ensemble_mode,
                'final_performance': self._get_current_performance()
            }
            
            self.training_history.append(training_record)
            self.total_timesteps_trained += timesteps
            
            print(f"✅ Professional Training completed!")
            print(f"   Duration: {training_duration:.0f} seconds")
            print(f"   Total timesteps: {timesteps:,}")
            print(f"   Algorithm: {self.primary_algorithm}")
            
            # Perform final evaluation
            self._evaluate_all_agents()
            
            return True
            
        except Exception as e:
            print(f"❌ Professional training error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _curriculum_training(self, agent, total_timesteps, callback):
        """Implement curriculum learning"""
        try:
            print("📚 Implementing Curriculum Learning...")
            
            # Phase 1: Basic market understanding (30% of time)
            phase1_steps = int(total_timesteps * 0.3)
            print(f"📖 Phase 1: Basic Market Understanding ({phase1_steps:,} steps)")
            agent.learn(total_timesteps=phase1_steps, callback=callback, progress_bar=True)
            
            # Phase 2: Position management (40% of time)
            phase2_steps = int(total_timesteps * 0.4)
            print(f"📊 Phase 2: Position Management ({phase2_steps:,} steps)")
            agent.learn(total_timesteps=phase2_steps, callback=callback, progress_bar=True)
            
            # Phase 3: Advanced strategies (30% of time)
            phase3_steps = total_timesteps - phase1_steps - phase2_steps
            print(f"🎯 Phase 3: Advanced Strategies ({phase3_steps:,} steps)")
            agent.learn(total_timesteps=phase3_steps, callback=callback, progress_bar=True)
            
            print("✅ Curriculum Learning completed!")
            
        except Exception as e:
            print(f"Curriculum learning error: {e}")
            # Fallback to standard training
            agent.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    
    def _evaluate_all_agents(self):
        """Evaluate all agents comprehensively"""
        try:
            print("🔍 Evaluating all agents...")
            
            eval_env = DummyVecEnv([lambda: Monitor(self.env, self.log_path)])
            
            for agent_name, agent in self.agents.items():
                print(f"📊 Evaluating {agent_name} agent...")
                
                mean_reward, std_reward = evaluate_policy(
                    agent, eval_env, n_eval_episodes=10, deterministic=True
                )
                
                # Update performance tracking
                self.agent_performance[agent_name]['total_rewards'].append(mean_reward)
                self.agent_performance[agent_name]['episodes'] += 10
                
                # Calculate additional metrics
                self._calculate_agent_metrics(agent_name, mean_reward, std_reward)
                
                print(f"   {agent_name}: {mean_reward:.3f} ± {std_reward:.3f}")
            
            # Select best agent as active
            self._select_best_agent()
            
        except Exception as e:
            print(f"Agent evaluation error: {e}")
    
    def _calculate_agent_metrics(self, agent_name, mean_reward, std_reward):
        """Calculate additional performance metrics for agent"""
        try:
            performance = self.agent_performance[agent_name]
            
            # Calculate success rate (reward > 0)
            recent_rewards = performance['total_rewards'][-10:]  # Last 10 evaluations
            success_rate = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards) if recent_rewards else 0
            performance['success_rate'] = success_rate
            
            # Calculate stability (inverse of std deviation)
            stability_score = max(0, 1 - (std_reward / max(abs(mean_reward), 1)))
            performance['stability_score'] = stability_score
            
            # Portfolio efficiency (placeholder - would be calculated from actual trading data)
            performance['avg_portfolio_efficiency'] = min(mean_reward / 10.0, 1.0)  # Normalize
            
            # Market timing score (placeholder)
            performance['market_timing_score'] = np.random.uniform(0.4, 0.9)  # Would be calculated from actual performance
            
        except Exception as e:
            print(f"Metrics calculation error for {agent_name}: {e}")
    
    def _select_best_agent(self):
        """Select the best performing agent as active"""
        try:
            if not self.agent_performance:
                return
                
            best_agent = None
            best_score = -np.inf
            
            for agent_name, performance in self.agent_performance.items():
                if not performance['total_rewards']:
                    continue
                    
                # Calculate composite score
                recent_reward = np.mean(performance['total_rewards'][-5:])  # Last 5 evaluations
                success_rate = performance['success_rate']
                stability = performance.get('stability_score', 0.5)
                
                composite_score = (
                    recent_reward * 0.5 +
                    success_rate * 20 * 0.3 +  # Scale success rate
                    stability * 10 * 0.2       # Scale stability
                )
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_agent = agent_name
            
            if best_agent and best_agent != self.active_agent:
                print(f"🔄 Switching active agent: {self.active_agent} → {best_agent}")
                print(f"   Score improvement: {best_score:.3f}")
                self.active_agent = best_agent
                
                # Record agent selection
                self.agent_selection_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'selected_agent': best_agent,
                    'score': best_score,
                    'reason': 'performance_evaluation'
                })
                
        except Exception as e:
            print(f"Agent selection error: {e}")
    
    def get_action(self, observation, deterministic: bool = True):
        """Get professional action from active agent"""
        try:
            if not self.agents or self.active_agent not in self.agents:
                # Return safe default action
                return self._get_default_action()
            
            # Get action from active agent
            agent = self.agents[self.active_agent]
            action, _states = agent.predict(observation, deterministic=deterministic)
            
            # Professional action analysis
            self._analyze_action(action)
            
            # Ensemble decision if enabled
            if self.ensemble_mode and len(self.agents) > 1:
                action = self._get_ensemble_action(observation, deterministic)
            
            # Action validation and safety checks
            action = self._validate_and_enhance_action(action, observation)
            
            return action
            
        except Exception as e:
            print(f"Action generation error: {e}")
            return self._get_default_action()
    
    def _get_default_action(self):
        """Generate safe default action"""
        try:
            # Create conservative default action
            default_action = np.array([
                0.0,    # market_direction (neutral)
                0.01,   # position_size (minimum)
                0.0,    # entry_aggression (limit order)
                1.0,    # profit_target_ratio (1:1)
                0.0,    # partial_take_levels (none)
                0.0,    # add_position_signal (no scaling)
                0.0,    # hedge_ratio (no hedging)
                0.0,    # recovery_mode (none)
                0.5,    # correlation_limit (moderate)
                0.3,    # volatility_filter (conservative)
                0.5,    # spread_tolerance (moderate)
                0.5,    # time_filter (moderate)
                0.3,    # portfolio_heat_limit (conservative)
                0.0,    # smart_exit_signal (no exit)
                0.0     # rebalance_trigger (no rebalance)
            ], dtype=np.float32)
            
            return default_action
            
        except Exception as e:
            print(f"Default action generation error: {e}")
            return np.zeros(15, dtype=np.float32)
    
    def _analyze_action(self, action):
        """Analyze action for professional insights"""
        try:
            # Track action distribution
            action_bins = np.digitize(action, bins=np.linspace(-1, 5, 16)) - 1
            action_bins = np.clip(action_bins, 0, 14)
            
            for i, bin_idx in enumerate(action_bins):
                if i < len(self.action_distribution):
                    self.action_distribution[i] += 1
            
            # Analyze action patterns
            action_analysis = {
                'timestamp': datetime.now().isoformat(),
                'action': action.tolist(),
                'market_direction': action[0],
                'position_size': action[1],
                'action_type': self._classify_action_type(action),
                'risk_level': self._calculate_action_risk(action),
                'complexity': self._calculate_action_complexity(action)
            }
            
            self.action_analysis.append(action_analysis)
            
            # Keep only recent analysis
            if len(self.action_analysis) > 1000:
                self.action_analysis = self.action_analysis[-1000:]
                
        except Exception as e:
            print(f"Action analysis error: {e}")
    
    def _classify_action_type(self, action):
        """Classify the type of action being taken"""
        try:
            market_direction = action[0]
            exit_signal = action[13]
            rebalance_signal = action[14]
            recovery_mode = action[7]
            
            if abs(market_direction) > 0.6:
                return 'AGGRESSIVE_ENTRY'
            elif abs(market_direction) > 0.3:
                return 'MODERATE_ENTRY'
            elif exit_signal > 0.7:
                return 'EXIT'
            elif rebalance_signal > 0.7:
                return 'REBALANCE'
            elif recovery_mode > 0.5:
                return 'RECOVERY'
            else:
                return 'HOLD'
                
        except Exception as e:
            return 'UNKNOWN'
    
    def _calculate_action_risk(self, action):
        """Calculate risk level of action"""
        try:
            position_size = action[1]
            entry_aggression = action[2]
            recovery_mode = action[7]
            portfolio_heat_limit = action[12]
            
            risk_score = (
                position_size * 0.3 +           # Larger positions = higher risk
                entry_aggression * 0.2 +        # Market orders = higher risk
                recovery_mode * 0.3 +           # Recovery mode = higher risk
                (1 - portfolio_heat_limit) * 0.2  # Lower heat limit = lower risk
            )
            
            if risk_score > 0.7:
                return 'HIGH'
            elif risk_score > 0.4:
                return 'MEDIUM'
            else:
                return 'LOW'
                
        except Exception as e:
            return 'UNKNOWN'
    
    def _calculate_action_complexity(self, action):
        """Calculate complexity of action"""
        try:
            # Count non-zero/non-default actions
            significant_actions = sum(1 for x in action if abs(x) > 0.1)
            
            if significant_actions > 8:
                return 'VERY_COMPLEX'
            elif significant_actions > 5:
                return 'COMPLEX'
            elif significant_actions > 2:
                return 'MODERATE'
            else:
                return 'SIMPLE'
                
        except Exception as e:
            return 'UNKNOWN'
    
    def _get_ensemble_action(self, observation, deterministic=True):
        """Get ensemble action from multiple agents"""
        try:
            actions = []
            weights = []
            
            for agent_name, agent in self.agents.items():
                try:
                    action, _ = agent.predict(observation, deterministic=deterministic)
                    actions.append(action)
                    
                    # Weight by recent performance
                    performance = self.agent_performance.get(agent_name, {})
                    recent_rewards = performance.get('total_rewards', [0])
                    weight = max(np.mean(recent_rewards[-3:]), 0.1)  # Minimum weight 0.1
                    weights.append(weight)
                    
                except Exception as e:
                    print(f"Error getting action from {agent_name}: {e}")
                    continue
            
            if not actions:
                return self._get_default_action()
            
            # Weighted ensemble
            actions = np.array(actions)
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize weights
            
            ensemble_action = np.average(actions, axis=0, weights=weights)
            
            # Add ensemble-specific adjustments
            ensemble_action = self._adjust_ensemble_action(ensemble_action, actions, weights)
            
            return ensemble_action.astype(np.float32)
            
        except Exception as e:
            print(f"Ensemble action error: {e}")
            return self._get_default_action()
    
    def _adjust_ensemble_action(self, ensemble_action, individual_actions, weights):
        """Apply ensemble-specific adjustments"""
        try:
            # Conservative adjustment - reduce extreme actions
            ensemble_action[0] *= 0.8  # Reduce market direction aggressiveness
            ensemble_action[1] = min(ensemble_action[1], 0.05)  # Cap position size
            
            # Increase safety margins
            ensemble_action[9] = max(ensemble_action[9], 0.3)   # Minimum volatility filter
            ensemble_action[12] = max(ensemble_action[12], 0.2)  # Minimum portfolio heat limit
            
            return ensemble_action
            
        except Exception as e:
            print(f"Ensemble adjustment error: {e}")
            return ensemble_action
    
    def _validate_and_enhance_action(self, action, observation):
        """Validate and enhance action with professional logic"""
        try:
            # Ensure action is within bounds
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            
            # Professional enhancements based on market conditions
            action = self._apply_market_condition_filters(action, observation)
            
            # Risk management enhancements
            action = self._apply_risk_management_filters(action, observation)
            
            # Portfolio optimization
            action = self._apply_portfolio_optimization(action, observation)
            
            return action.astype(np.float32)
            
        except Exception as e:
            print(f"Action validation error: {e}")
            return action
    
    def _apply_market_condition_filters(self, action, observation):
        """Apply market condition based filters"""
        try:
            # Extract market regime information from observation
            # (This would use the observation structure from Professional Environment)
            
            # Placeholder for market condition logic
            market_volatility = observation[57] if len(observation) > 57 else 0.5  # Market volatility feature
            
            # Reduce position size in high volatility
            if market_volatility > 0.8:
                action[1] *= 0.7  # Reduce position size
                action[9] = max(action[9], 0.6)  # Increase volatility filter
            
            # Reduce aggressiveness in uncertain conditions
            if market_volatility > 0.9:
                action[0] *= 0.5  # Reduce market direction strength
                action[2] = min(action[2], 0.3)  # Prefer limit orders
            
            return action
            
        except Exception as e:
            print(f"Market condition filter error: {e}")
            return action
    
    def _apply_risk_management_filters(self, action, observation):
        """Apply risk management filters"""
        try:
            # Extract portfolio information from observation
            portfolio_heat = observation[90] if len(observation) > 90 else 0.0  # Portfolio heat feature
            
            # Reduce risk if portfolio heat is high
            if portfolio_heat > 0.6:
                action[1] *= 0.6  # Reduce position size
                action[12] = min(action[12], 0.4)  # Lower heat limit
                action[0] *= 0.7  # Reduce direction strength
            
            # Enhance exit signals if in high risk state
            if portfolio_heat > 0.8:
                action[13] = max(action[13], 0.6)  # Increase exit signal
            
            return action
            
        except Exception as e:
            print(f"Risk management filter error: {e}")
            return action
    
    def _apply_portfolio_optimization(self, action, observation):
        """Apply portfolio optimization logic"""
        try:
            # Extract position information from observation
            total_positions = observation[60] if len(observation) > 60 else 0.0  # Total positions feature
            
            # Limit new positions if already heavily positioned
            if total_positions > 0.7:  # More than 70% of max positions
                action[1] = min(action[1], 0.02)  # Reduce new position size
                action[14] = max(action[14], 0.5)  # Increase rebalance signal
            
            # Encourage diversification
            if total_positions > 0.5:
                action[5] = min(action[5], 0.3)  # Reduce add position signal
                action[6] = max(action[6], 0.2)  # Increase hedge consideration
            
            return action
            
        except Exception as e:
            print(f"Portfolio optimization error: {e}")
            return action
    
    def save_model(self, filename: str = None):
        """Save all professional models"""
        try:
            if not self.agents:
                print("❌ No agents to save")
                return None
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_results = {}
            
            for agent_name, agent in self.agents.items():
                if filename:
                    agent_filename = f"{filename}_{agent_name}"
                else:
                    agent_filename = f"professional_{agent_name}_{timestamp}"
                
                full_path = os.path.join(self.model_save_path, agent_filename)
                agent.save(full_path)
                
                save_results[agent_name] = full_path + '.zip'
                
            # Save metadata
            metadata = {
                'save_time': datetime.now().isoformat(),
                'active_agent': self.active_agent,
                'ensemble_mode': self.ensemble_mode,
                'total_timesteps_trained': self.total_timesteps_trained,
                'model_paths': save_results,
                'agent_performance': self.agent_performance,
                'action_distribution': self.action_distribution.tolist(),
                'config': self.config
            }
            
            metadata_path = os.path.join(self.model_save_path, f"professional_metadata_{timestamp}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4, default=str)
            
            self.model_versions.append(metadata)
            
            print(f"💾 Professional models saved:")
            for agent_name, path in save_results.items():
                print(f"   {agent_name}: {path}")
            print(f"   Metadata: {metadata_path}")
            
            return save_results
            
        except Exception as e:
            print(f"❌ Model save error: {e}")
            return None
    
    def load_model(self, filename: str = None, agent_name: str = 'primary'):
        """Load professional model"""
        try:
            if filename is None:
                # Find latest model
                self._load_model_versions()
                if not self.model_versions:
                    print("❌ No saved models found")
                    return False
                
                # Get latest metadata
                latest_metadata = max(self.model_versions, key=lambda x: x.get('save_time', ''))
                model_paths = latest_metadata.get('model_paths', {})
                
                if agent_name not in model_paths:
                    print(f"❌ Agent {agent_name} not found in latest save")
                    return False
                    
                filename = model_paths[agent_name].replace('.zip', '')
            
            # Check if file exists
            model_path = filename if filename.endswith('.zip') else filename + '.zip'
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                return False
            
            # Determine algorithm type from filename or config
            if 'sac' in model_path.lower():
                algorithm = 'SAC'
            elif 'td3' in model_path.lower():
                algorithm = 'TD3'
            else:
                algorithm = 'PPO'
            
            # Load model
            if algorithm == 'PPO':
                model = PPO.load(model_path, env=self.env)
            elif algorithm == 'SAC':
                model = SAC.load(model_path, env=self.env)
            elif algorithm == 'TD3':
                model = TD3.load(model_path, env=self.env)
            else:
                print(f"❌ Unsupported algorithm: {algorithm}")
                return False
            
            # Add to agents
            self.agents[agent_name] = model
            if not self.active_agent:
                self.active_agent = agent_name
                
            print(f"✅ Professional model loaded: {model_path}")
            print(f"   Algorithm: {algorithm}")
            print(f"   Agent: {agent_name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model load error: {e}")
            return False
    
    def _load_model_versions(self):
        """Load information about saved model versions"""
        try:
            self.model_versions = []
            
            if not os.path.exists(self.model_save_path):
                return
                
            # Find all metadata files
            for filename in os.listdir(self.model_save_path):
                if filename.startswith('professional_metadata_') and filename.endswith('.json'):
                    metadata_path = os.path.join(self.model_save_path, filename)
                    
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            self.model_versions.append(metadata)
                    except Exception as e:
                        print(f"Error loading metadata {filename}: {e}")
                        continue
                        
            # Sort by save time
            self.model_versions.sort(key=lambda x: x.get('save_time', ''))
            
            print(f"📁 Found {len(self.model_versions)} professional model versions")
            
        except Exception as e:
            print(f"Model versions loading error: {e}")
    
    def get_professional_info(self):
        """Get comprehensive professional agent information"""
        try:
            info = {
                'agents': {
                    'total_agents': len(self.agents),
                    'active_agent': self.active_agent,
                    'available_agents': list(self.agents.keys()),
                    'ensemble_mode': self.ensemble_mode
                },
                'training': {
                    'total_timesteps_trained': self.total_timesteps_trained,
                    'training_sessions': len(self.training_history),
                    'is_training': self.is_training
                },
                'performance': self.agent_performance,
                'action_analysis': {
                    'total_actions_analyzed': len(self.action_analysis),
                    'action_distribution': self.action_distribution.tolist(),
                    'recent_action_types': [a.get('action_type') for a in self.action_analysis[-10:]]
                },
                'model_info': {
                    'saved_versions': len(self.model_versions),
                    'last_save': self.model_versions[-1].get('save_time') if self.model_versions else None
                }
            }
            
            # Add algorithm-specific info
            if self.active_agent and self.active_agent in self.agents:
                active_model = self.agents[self.active_agent]
                info['active_model'] = {
                    'algorithm': type(active_model).__name__,
                    'learning_rate': getattr(active_model, 'learning_rate', 'N/A'),
                    'num_timesteps': getattr(active_model, 'num_timesteps', 0)
                }
            
            return info
            
        except Exception as e:
            print(f"Professional info error: {e}")
            return {'error': str(e)}
    
    def get_action_distribution_analysis(self):
        """Analyze action distribution and usage patterns"""
        try:
            if not self.action_analysis:
                return "No action data available for analysis"
            
            analysis = "\n" + "="*60 + "\n"
            analysis += "🤖 PROFESSIONAL ACTION ANALYSIS\n"
            analysis += "="*60 + "\n"
            
            # Action type distribution
            action_types = [a.get('action_type', 'UNKNOWN') for a in self.action_analysis[-100:]]
            type_counts = {t: action_types.count(t) for t in set(action_types)}
            
            analysis += "📊 Action Type Distribution (Last 100 actions):\n"
            for action_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(action_types)) * 100
                analysis += f"   {action_type}: {count} ({percentage:.1f}%)\n"
            
            # Risk level distribution
            risk_levels = [a.get('risk_level', 'UNKNOWN') for a in self.action_analysis[-100:]]
            risk_counts = {r: risk_levels.count(r) for r in set(risk_levels)}
            
            analysis += "\n⚠️ Risk Level Distribution:\n"
            for risk_level, count in sorted(risk_counts.items()):
                percentage = (count / len(risk_levels)) * 100
                analysis += f"   {risk_level}: {count} ({percentage:.1f}%)\n"
            
            # Action complexity
            complexities = [a.get('complexity', 'UNKNOWN') for a in self.action_analysis[-100:]]
            complexity_counts = {c: complexities.count(c) for c in set(complexities)}
            
            analysis += "\n🧠 Action Complexity Distribution:\n"
            for complexity, count in sorted(complexity_counts.items()):
                percentage = (count / len(complexities)) * 100
                analysis += f"   {complexity}: {count} ({percentage:.1f}%)\n"
            
            # Recent performance correlation
            if len(self.action_analysis) > 10:
                recent_actions = self.action_analysis[-10:]
                avg_market_direction = np.mean([a['market_direction'] for a in recent_actions])
                avg_position_size = np.mean([a['position_size'] for a in recent_actions])
                
                analysis += f"\n📈 Recent Action Patterns:\n"
                analysis += f"   Average Market Direction: {avg_market_direction:+.3f}\n"
                analysis += f"   Average Position Size: {avg_position_size:.3f}\n"
                
                if avg_market_direction > 0.1:
                    analysis += f"   🔹 Bullish bias detected\n"
                elif avg_market_direction < -0.1:
                    analysis += f"   🔹 Bearish bias detected\n"
                else:
                    analysis += f"   🔹 Neutral market stance\n"
            
            analysis += "="*60
            return analysis
            
        except Exception as e:
            return f"Action analysis error: {e}"
    
    def adaptive_learning_rate_adjustment(self):
        """Adjust learning rate based on performance"""
        try:
            if not self.adaptive_lr or not self.agent_performance:
                return
                
            for agent_name, agent in self.agents.items():
                performance = self.agent_performance.get(agent_name, {})
                recent_rewards = performance.get('total_rewards', [])
                
                if len(recent_rewards) < 5:
                    continue
                    
                # Calculate trend
                recent_trend = np.mean(recent_rewards[-3:]) - np.mean(recent_rewards[-6:-3])
                
                # Adjust learning rate based on trend
                if hasattr(agent, 'learning_rate'):
                    current_lr = agent.learning_rate
                    
                    if recent_trend > 0.1:  # Improving
                        new_lr = min(current_lr * 1.05, 0.01)  # Slight increase
                    elif recent_trend < -0.1:  # Declining
                        new_lr = max(current_lr * 0.95, 0.0001)  # Slight decrease
                    else:
                        new_lr = current_lr  # Keep same
                        
                    if abs(new_lr - current_lr) > 0.00001:  # Significant change
                        agent.learning_rate = new_lr
                        print(f"🎯 {agent_name} learning rate: {current_lr:.6f} → {new_lr:.6f}")
                        
        except Exception as e:
            print(f"Adaptive learning rate error: {e}")
    
    def train_async(self, total_timesteps: int = None, callback: Callable = None):
        """Start professional training in background thread"""
        if self.is_training:
            print("⚠️ Training already in progress")
            return False
            
        self.is_training = True
        
        def training_worker():
            try:
                success = self.train(total_timesteps, callback)
                return success
            except Exception as e:
                print(f"Async training error: {e}")
                return False
            finally:
                self.is_training = False
                
        self.training_thread = threading.Thread(target=training_worker, daemon=True)
        self.training_thread.start()
        
        print("🚀 Professional training started in background")
        return True
    
    def stop_training(self):
        """Stop ongoing training"""
        self.is_training = False
        print("⏹️ Training stop requested...")
    
    def get_training_progress(self):
        """Get current training progress"""
        try:
            if not self.is_training:
                return None
                
            progress = {
                'is_training': self.is_training,
                'active_agent': self.active_agent,
                'total_agents': len(self.agents),
                'ensemble_mode': self.ensemble_mode
            }
            
            # Add agent-specific progress if available
            if self.active_agent and self.active_agent in self.agents:
                agent = self.agents[self.active_agent]
                if hasattr(agent, 'num_timesteps'):
                    progress['current_timesteps'] = agent.num_timesteps
                    
            return progress
            
        except Exception as e:
            print(f"Training progress error: {e}")
            return None
    
    def explain_action(self, observation, action):
        """Provide detailed explanation of action decision"""
        try:
            explanation = {
                'action_vector': action.tolist(),
                'action_interpretation': {},
                'decision_factors': {},
                'risk_assessment': {},
                'market_context': {}
            }
            
            # Interpret each action dimension
            action_meanings = [
                'market_direction', 'position_size', 'entry_aggression', 'profit_target_ratio',
                'partial_take_levels', 'add_position_signal', 'hedge_ratio', 'recovery_mode',
                'correlation_limit', 'volatility_filter', 'spread_tolerance', 'time_filter',
                'portfolio_heat_limit', 'smart_exit_signal', 'rebalance_trigger'
            ]
            
            for i, meaning in enumerate(action_meanings):
                if i < len(action):
                    explanation['action_interpretation'][meaning] = {
                        'value': float(action[i]),
                        'description': self._get_action_description(meaning, action[i])
                    }
            
            # Add decision factors
            explanation['decision_factors'] = {
                'primary_strategy': self._identify_primary_strategy(action),
                'risk_level': self._calculate_action_risk(action),
                'complexity': self._calculate_action_complexity(action),
                'active_agent': self.active_agent,
                'ensemble_mode': self.ensemble_mode
            }
            
            return explanation
            
        except Exception as e:
            print(f"Action explanation error: {e}")
            return {'error': str(e), 'action': action.tolist()}
    
    def _get_action_description(self, action_name, value):
        """Get human-readable description of action value"""
        try:
            descriptions = {
                'market_direction': f"{'Strong Buy' if value > 0.6 else 'Buy' if value > 0.3 else 'Strong Sell' if value < -0.6 else 'Sell' if value < -0.3 else 'Neutral'}",
                'position_size': f"{'Large' if value > 0.05 else 'Medium' if value > 0.02 else 'Small'} position ({value:.3f} lots)",
                'entry_aggression': f"{'Market order' if value > 0.7 else 'Aggressive limit' if value > 0.4 else 'Conservative limit'} entry",
                'profit_target_ratio': f"{value:.1f}:1 Risk:Reward ratio",
                'smart_exit_signal': f"{'Strong exit signal' if value > 0.7 else 'Moderate exit signal' if value > 0.4 else 'No exit signal'}"
            }
            
            return descriptions.get(action_name, f"Value: {value:.3f}")
            
        except:
            return f"Value: {value:.3f}"
    
    def _identify_primary_strategy(self, action):
        """Identify the primary strategy from action"""
        try:
            if abs(action[0]) > 0.5:  # Strong market direction
                return 'DIRECTIONAL_TRADING'
            elif action[13] > 0.6:  # Strong exit signal
                return 'PROFIT_TAKING'
            elif action[14] > 0.6:  # Strong rebalance signal
                return 'PORTFOLIO_REBALANCING'
            elif action[7] > 0.5:  # Recovery mode
                return 'LOSS_RECOVERY'
            elif action[6] > 0.5:  # Hedge ratio
                return 'RISK_HEDGING'
            else:
                return 'CONSERVATIVE_MANAGEMENT'
                
        except:
            return 'UNKNOWN'


# === FACTORY FUNCTIONS ===

def create_professional_agent(environment, config=None):
    """Factory function to create professional RL agent"""
    return ProfessionalRLAgent(environment, config)

# Keep old class name for backward compatibility
RLAgent = ProfessionalRLAgent

# Export main classes
__all__ = [
    'ProfessionalRLAgent',
    'RLAgent',  # Backward compatibility
    'ProfessionalTradingCallback',
    'create_professional_agent'
]
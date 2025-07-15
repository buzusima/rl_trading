# rl_agent.py - Reinforcement Learning Agent Manager
import numpy as np
import pandas as pd
import pickle
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
import threading
import time

# Stable-Baselines3 imports with error handling
try:
    from stable_baselines3 import PPO, DQN, A2C
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.utils import set_random_seed
    SB3_AVAILABLE = True
except ImportError as e:
    print(f"Stable-Baselines3 not available: {e}")
    print("Install with: pip install stable-baselines3[extra]")
    SB3_AVAILABLE = False
    
class TradingCallback(BaseCallback):
    """
    Custom callback for training monitoring
    """
    
    def __init__(self, eval_env, eval_freq: int = 1000, 
                 save_freq: int = 5000, save_path: str = 'models/', verbose: int = 0):
        super(TradingCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.save_path = save_path
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_mean_reward = -np.inf
        
        # Ensure save directory exists
        os.makedirs(save_path, exist_ok=True)
        
    def _init_callback(self) -> None:
        """
        This method is called when the callback is initialized
        """
        pass
        
    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass
        
    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass
        
    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        
        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.
        
        :return: If the callback returns False, training is aborted early.
        """
        # Evaluate policy periodically
        if self.n_calls % self.eval_freq == 0 and self.eval_env is not None:
            try:
                from stable_baselines3.common.evaluation import evaluate_policy
                mean_reward, std_reward = evaluate_policy(
                    self.model, self.eval_env, n_eval_episodes=3, deterministic=True
                )
                
                print(f"Step {self.n_calls}: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
                
                # Save best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    best_model_path = os.path.join(self.save_path, 'best_model')
                    self.model.save(best_model_path)
                    print(f"New best model saved with reward: {mean_reward:.2f}")
                    
            except Exception as e:
                print(f"Evaluation error: {e}")
                
        # Save model periodically
        if self.n_calls % self.save_freq == 0:
            try:
                checkpoint_path = os.path.join(self.save_path, f'model_step_{self.n_calls}')
                self.model.save(checkpoint_path)
                print(f"Model checkpoint saved at step {self.n_calls}")
            except Exception as e:
                print(f"Save error: {e}")
                
        return True
        
    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # Log training statistics
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            try:
                ep_info = self.model.ep_info_buffer[-1]
                self.episode_rewards.append(ep_info.get('r', 0))
                self.episode_lengths.append(ep_info.get('l', 0))
            except:
                pass
                
    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        # Save final model
        try:
            final_model_path = os.path.join(self.save_path, 'final_model')
            self.model.save(final_model_path)
            print("Final model saved")
        except Exception as e:
            print(f"Final save error: {e}")

class RLAgent:
    """
    Reinforcement Learning Agent for trading
    Manages model training, inference, and performance tracking
    """
    
    def __init__(self, environment, config: Dict = None):
        self.env = environment
        self.config = config or {}
        
        # Model parameters
        self.algorithm = self.config.get('algorithm', 'PPO')
        self.learning_rate = self.config.get('learning_rate', 0.0003)
        self.policy = self.config.get('policy', 'MlpPolicy')
        self.device = self.config.get('device', 'auto')
        
        # Training parameters
        self.total_timesteps = self.config.get('training_steps', 10000)
        self.batch_size = self.config.get('batch_size', 64)
        self.n_epochs = self.config.get('n_epochs', 10)
        self.gamma = self.config.get('gamma', 0.99)
        self.gae_lambda = self.config.get('gae_lambda', 0.95)
        
        # Model and training state
        self.model = None
        self.is_training = False
        self.training_thread = None
        self.training_callback = None
        
        # Performance tracking
        self.training_history = []
        self.evaluation_history = []
        self.model_versions = []
        
        # Paths
        self.model_save_path = 'models/trained_models/'
        self.log_path = 'data/training_logs/'
        
        # Create directories
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        
        # Initialize model
        self.initialize_model()
        
    def initialize_model(self):
        """
        Initialize the RL model based on configuration
        """
        try:
            # Wrap environment
            vec_env = DummyVecEnv([lambda: Monitor(self.env, self.log_path)])
            
            # Model-specific parameters
            if self.algorithm.upper() == 'PPO':
                self.model = PPO(
                    policy=self.policy,
                    env=vec_env,
                    learning_rate=self.learning_rate,
                    n_steps=2048,
                    batch_size=self.batch_size,
                    n_epochs=self.n_epochs,
                    gamma=self.gamma,
                    gae_lambda=self.gae_lambda,
                    clip_range=0.2,
                    ent_coef=0.01,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    device=self.device,
                    verbose=1
                )
                
            elif self.algorithm.upper() == 'DQN':
                self.model = DQN(
                    policy=self.policy,
                    env=vec_env,
                    learning_rate=self.learning_rate,
                    buffer_size=100000,
                    learning_starts=1000,
                    batch_size=self.batch_size,
                    tau=1.0,
                    gamma=self.gamma,
                    train_freq=4,
                    gradient_steps=1,
                    target_update_interval=1000,
                    exploration_fraction=0.1,
                    exploration_initial_eps=1.0,
                    exploration_final_eps=0.05,
                    device=self.device,
                    verbose=1
                )
                
            elif self.algorithm.upper() == 'A2C':
                self.model = A2C(
                    policy=self.policy,
                    env=vec_env,
                    learning_rate=self.learning_rate,
                    n_steps=5,
                    gamma=self.gamma,
                    gae_lambda=self.gae_lambda,
                    ent_coef=0.01,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    device=self.device,
                    verbose=1
                )
                
            else:
                raise ValueError(f"Unsupported algorithm: {self.algorithm}")
                
            print(f"Initialized {self.algorithm} model with {self.policy} policy")
            
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            self.model = None
            
    def train(self, total_timesteps: int = None, callback: Callable = None):
        """
        Train the RL model
        """
        if self.model is None:
            print("Model not initialized")
            return False
            
        try:
            timesteps = total_timesteps or self.total_timesteps
            self.model.learn(
                total_timesteps=timesteps,
                progress_bar=True
            )
        
            # Setup callback
            callbacks = []
            
            # Add our custom callback if eval env is available
            try:
                eval_env = DummyVecEnv([lambda: self.env])
                self.training_callback = TradingCallback(
                    eval_env=eval_env,
                    save_path=self.model_save_path,
                    verbose=1
                )
                callbacks.append(self.training_callback)
            except Exception as e:
                print(f"Warning: Could not create evaluation callback: {e}")
                self.training_callback = None
            
            # Add custom callback if provided
            if callback:
                if hasattr(callback, '_init_callback'):
                    callbacks.append(callback)
                else:
                    # If it's a function, wrap it in a simple callback
                    class FunctionCallback(BaseCallback):
                        def __init__(self, func, verbose=0):
                            super().__init__(verbose)
                            self.func = func
                            
                        def _init_callback(self):
                            pass
                            
                        def _on_step(self):
                            try:
                                return self.func(locals(), globals())
                            except:
                                return True
                                
                    callbacks.append(FunctionCallback(callback))
                
            print(f"Starting training for {timesteps} timesteps...")
            
            # Record training start
            training_start = datetime.now()
            
            # Train the model
            if callbacks:
                self.model.learn(
                    total_timesteps=timesteps,
                    callback=callbacks,
                    progress_bar=True
                )
            else:
                self.model.learn(
                    total_timesteps=timesteps,
                    progress_bar=True
                )
            
            # Record training completion
            training_end = datetime.now()
            training_duration = (training_end - training_start).total_seconds()
            
            # Save training record
            training_record = {
                'algorithm': self.algorithm,
                'start_time': training_start.isoformat(),
                'end_time': training_end.isoformat(),
                'duration_seconds': training_duration,
                'total_timesteps': timesteps,
                'final_learning_rate': self.learning_rate,
                'model_path': self.save_model()
            }
            
            self.training_history.append(training_record)
            self.save_training_history()
            
            print(f"Training completed in {training_duration:.0f} seconds")
            return True
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    def train_async(self, total_timesteps: int = None, callback: Callable = None):
        """
        Start training in a separate thread
        """
        if self.is_training:
            print("Training already in progress")
            return False
            
        self.is_training = True
        
        def training_worker():
            try:
                self.train(total_timesteps, callback)
            finally:
                self.is_training = False
                
        self.training_thread = threading.Thread(target=training_worker)
        self.training_thread.daemon = True
        self.training_thread.start()
        
        return True
        
    def stop_training(self):
        """
        Stop ongoing training
        """
        self.is_training = False
        if self.training_thread and self.training_thread.is_alive():
            # Note: This is a graceful stop request
            # The actual stopping depends on the training loop
            print("Training stop requested...")
            
    def get_action(self, observation, deterministic: bool = True):
        """
        Get action from trained model
        """
        if self.model is None:
            # Return random action if no model
            return self.env.action_space.sample()
            
        try:
            action, _states = self.model.predict(
                observation, 
                deterministic=deterministic
            )
            return action
            
        except Exception as e:
            print(f"Error getting action: {str(e)}")
            return self.env.action_space.sample()
            
    def evaluate_model(self, n_episodes: int = 10):
        """
        Evaluate model performance
        """
        if self.model is None:
            print("No model to evaluate")
            return None
            
        try:
            eval_env = DummyVecEnv([lambda: self.env])
            
            mean_reward, std_reward = evaluate_policy(
                self.model, 
                eval_env, 
                n_eval_episodes=n_episodes,
                deterministic=True
            )
            
            evaluation_result = {
                'timestamp': datetime.now().isoformat(),
                'n_episodes': n_episodes,
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'algorithm': self.algorithm
            }
            
            self.evaluation_history.append(evaluation_result)
            
            print(f"Evaluation: {mean_reward:.2f} +/- {std_reward:.2f}")
            return evaluation_result
            
        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            return None
            
    def save_model(self, filename: str = None):
        """
        Save the trained model
        """
        if self.model is None:
            print("No model to save")
            return None
            
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.algorithm.lower()}_model_{timestamp}"
                
            full_path = os.path.join(self.model_save_path, filename)
            self.model.save(full_path)
            
            # Save model metadata
            metadata = {
                'algorithm': self.algorithm,
                'policy': self.policy,
                'learning_rate': self.learning_rate,
                'save_time': datetime.now().isoformat(),
                'training_timesteps': self.total_timesteps,
                'model_path': full_path + '.zip'
            }
            
            metadata_path = full_path + '_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
                
            self.model_versions.append(metadata)
            
            print(f"Model saved: {full_path}")
            return full_path
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return None
            
    def load_model(self, filename: str = None):
        """
        Load a trained model
        """
        try:
            if filename is None:
                # Find latest model
                if not self.model_versions:
                    self.load_model_versions()
                    
                if not self.model_versions:
                    print("No saved models found")
                    return False
                    
                # Get latest model
                latest_model = max(self.model_versions, 
                                 key=lambda x: x.get('save_time', ''))
                filename = latest_model['model_path'].replace('.zip', '')
                
            # Check if file exists
            model_path = filename if filename.endswith('.zip') else filename + '.zip'
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                return False
                
            # Load model based on algorithm
            if self.algorithm.upper() == 'PPO':
                self.model = PPO.load(model_path, env=self.env)
            elif self.algorithm.upper() == 'DQN':
                self.model = DQN.load(model_path, env=self.env)
            elif self.algorithm.upper() == 'A2C':
                self.model = A2C.load(model_path, env=self.env)
            else:
                print(f"Unsupported algorithm for loading: {self.algorithm}")
                return False
                
            print(f"Model loaded: {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
            
    def load_model_versions(self):
        """
        Load information about saved model versions
        """
        try:
            self.model_versions = []
            
            if not os.path.exists(self.model_save_path):
                return
                
            # Find all metadata files
            for filename in os.listdir(self.model_save_path):
                if filename.endswith('_metadata.json'):
                    metadata_path = os.path.join(self.model_save_path, filename)
                    
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        self.model_versions.append(metadata)
                        
            # Sort by save time
            self.model_versions.sort(key=lambda x: x.get('save_time', ''))
            
            print(f"Found {len(self.model_versions)} saved models")
            
        except Exception as e:
            print(f"Error loading model versions: {str(e)}")
            
    def get_model_info(self):
        """
        Get information about current model
        """
        if self.model is None:
            return None
            
        info = {
            'algorithm': self.algorithm,
            'policy': self.policy,
            'learning_rate': self.learning_rate,
            'device': self.device,
            'is_training': self.is_training
        }
        
        # Add algorithm-specific info
        if hasattr(self.model, 'num_timesteps'):
            info['num_timesteps'] = self.model.num_timesteps
            
        if hasattr(self.model, 'learning_rate'):
            info['current_lr'] = self.model.learning_rate
            
        return info
        
    def get_training_progress(self):
        """
        Get current training progress
        """
        if not self.is_training or self.model is None:
            return None
            
        progress = {
            'is_training': self.is_training,
            'algorithm': self.algorithm,
            'num_timesteps': getattr(self.model, 'num_timesteps', 0),
            'target_timesteps': self.total_timesteps
        }
        
        if hasattr(self.model, 'num_timesteps') and self.total_timesteps > 0:
            progress['progress_percent'] = (self.model.num_timesteps / self.total_timesteps) * 100
            
        return progress
        
    def hyperparameter_search(self, param_grid: Dict, n_trials: int = 10):
        """
        Perform hyperparameter optimization
        """
        try:
            try:
                import optuna
            except ImportError:
                print("Optuna not installed. Install with: pip install optuna")
                return None
            
            def objective(trial):
                # Suggest hyperparameters
                lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
                gamma = trial.suggest_uniform('gamma', 0.9, 0.999)
                
                if self.algorithm.upper() == 'PPO':
                    clip_range = trial.suggest_uniform('clip_range', 0.1, 0.3)
                    ent_coef = trial.suggest_loguniform('ent_coef', 1e-8, 1e-1)
                    
                    # Create temporary model
                    temp_env = DummyVecEnv([lambda: self.env])
                    temp_model = PPO(
                        policy=self.policy,
                        env=temp_env,
                        learning_rate=lr,
                        gamma=gamma,
                        clip_range=clip_range,
                        ent_coef=ent_coef,
                        verbose=0
                    )
                    
                # Train for shorter period
                temp_model.learn(total_timesteps=5000)
                
                # Evaluate
                mean_reward, _ = evaluate_policy(temp_model, temp_env, n_eval_episodes=5)
                
                return mean_reward
                
            # Create study
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            
            # Update config with best parameters
            best_params = study.best_params
            self.config.update(best_params)
            
            print(f"Best parameters: {best_params}")
            print(f"Best value: {study.best_value}")
            
            return best_params
            
        except ImportError:
            print("Optuna not installed for hyperparameter optimization")
            return None
        except Exception as e:
            print(f"Hyperparameter search error: {str(e)}")
            return None
            
    def save_training_history(self):
        """
        Save training history to file
        """
        try:
            history_data = {
                'training_history': self.training_history,
                'evaluation_history': self.evaluation_history,
                'model_versions': self.model_versions,
                'config': self.config
            }
            
            history_file = os.path.join(self.log_path, 'training_history.json')
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=4, default=str)
                
        except Exception as e:
            print(f"Error saving training history: {str(e)}")
            
    def load_training_history(self):
        """
        Load training history from file
        """
        try:
            history_file = os.path.join(self.log_path, 'training_history.json')
            
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                    
                self.training_history = history_data.get('training_history', [])
                self.evaluation_history = history_data.get('evaluation_history', [])
                self.model_versions = history_data.get('model_versions', [])
                
                print("Training history loaded")
                
        except Exception as e:
            print(f"Error loading training history: {str(e)}")
            
    def get_performance_metrics(self):
        """
        Calculate comprehensive performance metrics
        """
        if not self.evaluation_history:
            return None
            
        try:
            rewards = [eval_data['mean_reward'] for eval_data in self.evaluation_history]
            
            metrics = {
                'total_evaluations': len(self.evaluation_history),
                'latest_reward': rewards[-1] if rewards else 0,
                'best_reward': max(rewards) if rewards else 0,
                'average_reward': np.mean(rewards) if rewards else 0,
                'reward_std': np.std(rewards) if rewards else 0,
                'improvement_trend': self._calculate_trend(rewards),
                'training_sessions': len(self.training_history)
            }
            
            # Add training time statistics
            if self.training_history:
                durations = [session['duration_seconds'] for session in self.training_history]
                metrics.update({
                    'total_training_time': sum(durations),
                    'average_training_time': np.mean(durations),
                    'last_training_time': durations[-1] if durations else 0
                })
                
            return metrics
            
        except Exception as e:
            print(f"Error calculating performance metrics: {str(e)}")
            return None
            
    def _calculate_trend(self, values: List[float], window: int = 5):
        """
        Calculate trend in performance values
        """
        if len(values) < window:
            return 0.0
            
        try:
            recent_values = values[-window:]
            x = np.arange(len(recent_values))
            slope = np.polyfit(x, recent_values, 1)[0]
            return float(slope)
            
        except:
            return 0.0
            
    def create_ensemble(self, model_paths: List[str]):
        """
        Create ensemble of multiple trained models
        """
        try:
            self.ensemble_models = []
            
            for path in model_paths:
                if self.algorithm.upper() == 'PPO':
                    model = PPO.load(path)
                elif self.algorithm.upper() == 'DQN':
                    model = DQN.load(path)
                elif self.algorithm.upper() == 'A2C':
                    model = A2C.load(path)
                else:
                    continue
                    
                self.ensemble_models.append(model)
                
            print(f"Ensemble created with {len(self.ensemble_models)} models")
            return len(self.ensemble_models) > 0
            
        except Exception as e:
            print(f"Error creating ensemble: {str(e)}")
            return False
            
    def get_ensemble_action(self, observation, deterministic: bool = True):
        """
        Get action from ensemble of models (majority vote)
        """
        if not hasattr(self, 'ensemble_models') or not self.ensemble_models:
            return self.get_action(observation, deterministic)
            
        try:
            actions = []
            
            for model in self.ensemble_models:
                action, _ = model.predict(observation, deterministic=deterministic)
                actions.append(action)
                
            # For continuous actions, take average
            if len(actions[0].shape) > 0:  # Continuous action space
                ensemble_action = np.mean(actions, axis=0)
            else:  # Discrete action space - majority vote
                ensemble_action = np.round(np.mean(actions))
                
            return ensemble_action
            
        except Exception as e:
            print(f"Error getting ensemble action: {str(e)}")
            return self.get_action(observation, deterministic)
            
    def adaptive_learning_rate(self, performance_window: int = 10):
        """
        Adapt learning rate based on recent performance
        """
        if len(self.evaluation_history) < performance_window:
            return
            
        try:
            recent_rewards = [eval_data['mean_reward'] 
                            for eval_data in self.evaluation_history[-performance_window:]]
            
            # Calculate trend
            trend = self._calculate_trend(recent_rewards)
            
            # Adjust learning rate
            if trend < -0.1:  # Performance declining
                self.learning_rate *= 0.9  # Reduce learning rate
                print(f"Learning rate reduced to: {self.learning_rate:.6f}")
            elif trend > 0.1:  # Performance improving
                self.learning_rate *= 1.05  # Slightly increase learning rate
                print(f"Learning rate increased to: {self.learning_rate:.6f}")
                
            # Update model learning rate if possible
            if self.model and hasattr(self.model, 'learning_rate'):
                self.model.learning_rate = self.learning_rate
                
        except Exception as e:
            print(f"Error adapting learning rate: {str(e)}")
            
    def get_action_probabilities(self, observation):
        """
        Get action probabilities from model
        """
        if self.model is None:
            return None
            
        try:
            if hasattr(self.model, 'policy'):
                # Get action probabilities
                obs_tensor = self.model.policy.obs_to_tensor(observation)[0]
                
                if hasattr(self.model.policy, 'get_distribution'):
                    distribution = self.model.policy.get_distribution(obs_tensor)
                    
                    if hasattr(distribution, 'probs'):
                        return distribution.probs.detach().cpu().numpy()
                    elif hasattr(distribution, 'distribution'):
                        # For continuous distributions
                        return {
                            'mean': distribution.distribution.mean.detach().cpu().numpy(),
                            'std': distribution.distribution.stddev.detach().cpu().numpy()
                        }
                        
            return None
            
        except Exception as e:
            print(f"Error getting action probabilities: {str(e)}")
            return None
            
    def explain_action(self, observation, action):
        """
        Provide explanation for taken action
        """
        try:
            # Get action probabilities
            action_probs = self.get_action_probabilities(observation)
            
            # Basic explanation
            explanation = {
                'action': action,
                'algorithm': self.algorithm,
                'confidence': 'Unknown'
            }
            
            if action_probs is not None:
                if isinstance(action_probs, dict):
                    # Continuous action space
                    explanation['action_mean'] = action_probs.get('mean')
                    explanation['action_std'] = action_probs.get('std')
                else:
                    # Discrete action space
                    if len(action_probs) > int(action):
                        explanation['confidence'] = f"{action_probs[int(action)]:.3f}"
                        
            return explanation
            
        except Exception as e:
            print(f"Error explaining action: {str(e)}")
            return {'action': action, 'error': str(e)}
            
    def reset_training(self):
        """
        Reset training state and reinitialize model
        """
        self.stop_training()
        self.initialize_model()
        
        # Clear history
        self.training_history = []
        self.evaluation_history = []
        
        print("Training state reset")
        
    def export_model_for_deployment(self, filename: str = None):
        """
        Export model in deployment-ready format
        """
        if self.model is None:
            print("No model to export")
            return None
            
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"deployment_model_{timestamp}"
                
            # Save model
            model_path = self.save_model(filename)
            
            # Create deployment package
            deployment_info = {
                'model_path': model_path + '.zip',
                'algorithm': self.algorithm,
                'policy': self.policy,
                'observation_space': {
                    'shape': self.env.observation_space.shape,
                    'low': self.env.observation_space.low.tolist(),
                    'high': self.env.observation_space.high.tolist()
                },
                'action_space': {
                    'shape': self.env.action_space.shape,
                    'low': self.env.action_space.low.tolist(),
                    'high': self.env.action_space.high.tolist()
                },
                'performance_metrics': self.get_performance_metrics(),
                'export_time': datetime.now().isoformat()
            }
            
            deployment_file = model_path + '_deployment.json'
            with open(deployment_file, 'w') as f:
                json.dump(deployment_info, f, indent=4, default=str)
                
            print(f"Model exported for deployment: {deployment_file}")
            return deployment_file
            
        except Exception as e:
            print(f"Error exporting model: {str(e)}")
            return None
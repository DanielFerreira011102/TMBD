import torch
import logging
from pathlib import Path
import pika
import io
import json
from typing import Dict, Set
import mlflow
import os
import time
from datetime import datetime
from pika.exceptions import AMQPConnectionError, AMQPChannelError

from src.model.classifier import PneumoniaClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedOrchestrator:
    def __init__(
        self,
        num_clients: int,
        model_dir: str,
        rabbitmq_host: str = "localhost",
        min_clients: int = 2,
        rounds: int = 10,
        wait_time: int = 3600,
        heartbeat: int = 600,  # Increased heartbeat timeout to 10 minutes
        connection_retry_delay: int = 5,
        max_connection_retries: int = 3
    ):
        self.num_clients = num_clients
        self.min_clients = min_clients
        self.rounds = rounds
        self.wait_time = wait_time
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.rabbitmq_host = rabbitmq_host
        self.heartbeat = heartbeat
        self.connection_retry_delay = connection_retry_delay
        self.max_connection_retries = max_connection_retries

        # Initialize model
        self.global_model = PneumoniaClassifier(use_pretrained_weights=True)
        self.current_round = 0
        self.received_updates = {}
        self.registered_clients = set()
        
        # Connection state tracking
        self.connection = None
        self.channel = None
        self.should_reconnect = False
        self.was_consuming = False

        # Setup initial connection
        self.connect()

        # Setup MLflow
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("federated_learning")

    def _get_connection_parameters(self):
        """Create connection parameters with retry and timeout settings"""
        return pika.ConnectionParameters(
            host=self.rabbitmq_host,
            heartbeat=self.heartbeat,
            blocked_connection_timeout=300,
            connection_attempts=3,
            retry_delay=5,
            socket_timeout=300
        )

    def connect(self):
        """Setup RabbitMQ connection with improved retry logic and error handling"""
        for attempt in range(self.max_connection_retries):
            try:
                # Close existing connection if any
                if self.connection and not self.connection.is_closed:
                    try:
                        self.connection.close()
                    except Exception as e:
                        logger.warning(f"Error closing existing connection: {str(e)}")

                # Create new connection
                self.connection = pika.BlockingConnection(self._get_connection_parameters())
                self.channel = self.connection.channel()

                # Enable publisher confirms
                self.channel.confirm_delivery()

                # Declare exchange and queues with persistence
                self.channel.exchange_declare(
                    exchange='global_model_exchange',
                    exchange_type='fanout',
                    durable=True
                )
                self.channel.queue_declare(
                    queue='model_updates',
                    durable=True
                )
                
                # Add registration queue
                self.registration_queue = 'client_registrations'
                self.channel.queue_declare(
                    queue=self.registration_queue,
                    durable=True
                )
                
                logger.info("Successfully connected to RabbitMQ")
                self.should_reconnect = False
                return True

            except Exception as e:
                logger.error(f"Failed to connect to RabbitMQ (attempt {attempt + 1}/{self.max_connection_retries}): {str(e)}")
                if attempt < self.max_connection_retries - 1:
                    time.sleep(self.connection_retry_delay)
                else:
                    self.should_reconnect = True
                    raise

    def _process_heartbeat(self):
        """Process RabbitMQ heartbeat with error handling"""
        try:
            self.connection.process_data_events()
            return True
        except Exception as e:
            logger.warning(f"Error processing heartbeat: {str(e)}")
            self.should_reconnect = True
            return False

    def _handle_connection_error(self):
        """Handle connection errors with reconnection logic"""
        if self.should_reconnect:
            logger.info("Attempting to reconnect...")
            try:
                self.connect()
                return True
            except Exception as e:
                logger.error(f"Failed to reconnect: {str(e)}")
                return False
        return False

    def wait_for_clients(self):
        """Wait for clients to register with improved error handling"""
        logger.info(f"Waiting for at least {self.min_clients} clients to register...")
        start_time = time.time()
        
        def registration_callback(ch, method, properties, body):
            try:
                client_info = json.loads(body.decode())
                client_id = client_info['client_id']
                if client_id not in self.registered_clients:
                    self.registered_clients.add(client_id)
                    logger.info(f"Client {client_id} registered. Total registered: {len(self.registered_clients)}")
            except Exception as e:
                logger.error(f"Error processing registration: {str(e)}")
            finally:
                ch.basic_ack(delivery_tag=method.delivery_tag)

        while len(self.registered_clients) < self.min_clients:
            try:
                if time.time() - start_time > self.wait_time:
                    raise TimeoutError(f"Timeout waiting for clients. Only {len(self.registered_clients)} registered.")

                if not self.connection or self.connection.is_closed:
                    self._handle_connection_error()
                
                self.channel.basic_consume(
                    queue=self.registration_queue,
                    on_message_callback=registration_callback
                )
                
                self.connection.process_data_events(time_limit=1)
                
            except Exception as e:
                logger.error(f"Error in registration loop: {str(e)}")
                if not self._handle_connection_error():
                    time.sleep(self.connection_retry_delay)

        self.channel.stop_consuming()
        logger.info(f"Successfully registered {len(self.registered_clients)} clients")
        return True

    def _broadcast_global_model(self):
        """Broadcast global model with improved error handling and persistence"""
        for attempt in range(self.max_connection_retries):
            try:
                if not self.connection or self.connection.is_closed:
                    self._handle_connection_error()

                # Save state dict to buffer
                buffer = io.BytesIO()
                torch.save(self.global_model.state_dict(), buffer)
                
                # Create complete message
                message = {
                    'round': self.current_round,
                    'state_dict': buffer.getvalue().hex()
                }

                # Send with persistence and mandatory flag
                self.channel.basic_publish(
                    exchange='global_model_exchange',
                    routing_key='',
                    body=json.dumps(message).encode(),
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # make message persistent
                        content_type='application/json'
                    ),
                    mandatory=True
                )

                logger.info(f"Broadcasted global model for round {self.current_round + 1}")
                return True

            except Exception as e:
                logger.error(f"Failed to broadcast model (attempt {attempt + 1}/{self.max_connection_retries}): {str(e)}")
                if attempt < self.max_connection_retries - 1:
                    if not self._handle_connection_error():
                        time.sleep(self.connection_retry_delay)
                else:
                    raise

    def _collect_client_updates(self):
        """Collect updates from clients with improved error handling"""
        start_time = time.time()
        collected_clients = set()
        failed_clients = set()
        active_clients = self.registered_clients.copy()

        def callback(ch, method, properties, body):
            client_id = None
            try:
                message = json.loads(body.decode())
                client_id = message['client_id']
                
                if client_id not in active_clients:
                    logger.warning(f"Received update from unregistered client {client_id}")
                    return
                
                state_dict = torch.load(
                    io.BytesIO(bytes.fromhex(message['state_dict'])),
                    weights_only=True
                )
                
                self.received_updates[client_id] = {
                    'client_id': client_id,
                    'round': message['round'],
                    'state_dict': state_dict,
                    'metrics': message['metrics']
                }
                
                collected_clients.add(client_id)
                logger.info(f"Received update from client {client_id}")
                
            except Exception as e:
                error_msg = f"Error processing client update"
                if client_id:
                    error_msg += f" from client {client_id}"
                    failed_clients.add(client_id)
                    if client_id in active_clients:
                        active_clients.remove(client_id)
                logger.error(f"{error_msg}: {str(e)}")
            finally:
                ch.basic_ack(delivery_tag=method.delivery_tag)

        while len(collected_clients) < min(self.min_clients, len(active_clients)):
            try:
                if time.time() - start_time > self.wait_time:
                    logger.warning(f"Timeout waiting for client updates. "
                                f"Collected {len(collected_clients)} of {len(active_clients)} active clients")
                    break

                if not self.connection or self.connection.is_closed:
                    if not self._handle_connection_error():
                        time.sleep(self.connection_retry_delay)
                        continue

                self.channel.basic_consume(
                    queue='model_updates',
                    on_message_callback=callback
                )

                self.connection.process_data_events(time_limit=1)

                # Check if we've collected updates from all remaining active clients
                if collected_clients == active_clients:
                    break

            except Exception as e:
                logger.error(f"Error in collection loop: {str(e)}")
                if not self._handle_connection_error():
                    time.sleep(self.connection_retry_delay)

        self.channel.stop_consuming()

        # Check if we have enough active clients to continue
        if len(active_clients) < self.min_clients:
            logger.error(f"Not enough active clients to continue training. "
                      f"Active: {len(active_clients)}, Required: {self.min_clients}")
            return False

        # If we have enough active clients but some didn't respond, continue anyway
        return len(collected_clients) >= min(self.min_clients, len(active_clients))

    def start_training(self):
        """Start federated learning process with improved error handling"""
        logger.info("Initializing federated learning training")
        
        try:
            self.wait_for_clients()
        except TimeoutError as e:
            logger.error(str(e))
            return

        logger.info("Starting federated learning training")

        with mlflow.start_run(run_name=f"federated_run_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.log_params({
                "num_clients": self.num_clients,
                "min_clients": self.min_clients,
                "rounds": self.rounds,
                "wait_time": self.wait_time,
                "model_type": self.global_model.__class__.__name__
            })

            for round_num in range(self.rounds):
                try:
                    self.current_round = round_num
                    logger.info(f"Starting round {round_num + 1}/{self.rounds}")

                    if not self._broadcast_global_model():
                        logger.error("Failed to broadcast global model")
                        break

                    if not self._collect_client_updates():
                        logger.warning("Failed to collect enough client updates. Ending training.")
                        break

                    self._aggregate_updates()
                    self._aggregate_and_log_metrics(round_num)
                    self._save_model(round_num)
                    self.received_updates.clear()

                except Exception as e:
                    logger.error(f"Error in training round {round_num + 1}: {str(e)}")
                    if not self._handle_connection_error():
                        break

            # Save final model
            try:
                self._save_model()
                
                summary = {
                    "total_rounds_completed": self.current_round + 1,
                    "total_clients_participated": len(self.registered_clients),
                    "training_duration": time.time() - mlflow.active_run().info.start_time,
                }
                mlflow.log_dict(summary, "training_summary.json")
            
            except Exception as e:
                logger.error(f"Error saving final model: {str(e)}")

    def _aggregate_updates(self):
        """Aggregate client updates using FedAvg algorithm"""
        if not self.received_updates:
            return

        aggregated_state = {}
        first_state = next(iter(self.received_updates.values()))['state_dict']

        for key in first_state.keys():
            aggregated_state[key] = torch.zeros_like(first_state[key], dtype=first_state[key].dtype)

        num_updates = len(self.received_updates)
        for update in self.received_updates.values():
            state_dict = update['state_dict']
            for key in aggregated_state.keys():
                if state_dict[key].dtype == torch.long:
                    aggregated_state[key] += (state_dict[key].float() / num_updates).round().to(state_dict[key].dtype)
                else:
                    aggregated_state[key] += state_dict[key] / num_updates

        self.global_model.load_state_dict(aggregated_state)
        logger.info(f"Updated global model with {num_updates} client updates")

    def _save_model(self, round_num: int = None):
        """Save model with error handling"""
        try:
            model_path = self.model_dir / 'final_federated_model.pth'
            
            self.global_model.save_checkpoint(
                path=str(model_path),
                epoch=round_num if round_num is not None else self.rounds,
                optimizer=None,
                loss=0.0
            )
            logger.info(f"Saved model to {model_path}")
            
            # Log to MLflow with appropriate tags and metadata
            tags = {
                "round": str(round_num) if round_num is not None else "final",
                "timestamp": datetime.now().isoformat()
            }
            
            if round_num is not None:
                mlflow.pytorch.log_model(
                    self.global_model, 
                    f"model_round_{round_num + 1}",
                    registered_model_name="federated_model",
                    tags=tags
                )
            else:
                mlflow.pytorch.log_model(
                    self.global_model,
                    "final_model",
                    registered_model_name="federated_model",
                    tags=tags
                )
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def _aggregate_and_log_metrics(self, round_num: int):
        """Aggregate and log metrics from all clients with improved MLflow tracking"""
        if not self.received_updates:
            return

        try:
            # Calculate aggregate statistics
            losses = [m['metrics']['loss'] for m in self.received_updates.values()]
            accuracies = [m['metrics']['accuracy'] for m in self.received_updates.values()]
            total_samples = sum(m['metrics']['total_samples'] for m in self.received_updates.values())

            # Global metrics for this round
            global_metrics = {
                "global/loss": sum(losses) / len(losses),
                "global/accuracy": sum(accuracies) / len(accuracies),
                "global/min_loss": min(losses),
                "global/max_loss": max(losses),
                "global/min_accuracy": min(accuracies),
                "global/max_accuracy": max(accuracies),
                "global/std_loss": torch.tensor(losses).std().item(),
                "global/std_accuracy": torch.tensor(accuracies).std().item(),
                "training/active_clients": len(self.received_updates),
                "training/total_samples": total_samples
            }
            
            # Log global metrics with step
            mlflow.log_metrics(global_metrics, step=round_num)

            # Per-client metrics
            for client_id, update in self.received_updates.items():
                metrics = update['metrics']
                
                # Main client metrics
                client_metrics = {
                    f"clients/{client_id}/loss": metrics['loss'],
                    f"clients/{client_id}/accuracy": metrics['accuracy'],
                    f"clients/{client_id}/min_loss": metrics['min_loss'],
                    f"clients/{client_id}/max_loss": metrics['max_loss'],
                    f"clients/{client_id}/std_loss": metrics['std_loss'],
                    f"clients/{client_id}/min_accuracy": metrics['min_accuracy'],
                    f"clients/{client_id}/max_accuracy": metrics['max_accuracy'],
                    f"clients/{client_id}/std_accuracy": metrics['std_accuracy'],
                    f"clients/{client_id}/total_samples": metrics['total_samples']
                }
                mlflow.log_metrics(client_metrics, step=round_num)

                # Per-epoch metrics for each client
                for epoch_data in metrics['epoch_metrics']:
                    epoch = epoch_data['epoch']
                    epoch_metrics = {
                        f"clients/{client_id}/epochs/loss_{epoch}": epoch_data['loss'],
                        f"clients/{client_id}/epochs/accuracy_{epoch}": epoch_data['accuracy'],
                        f"clients/{client_id}/epochs/min_batch_loss_{epoch}": epoch_data['min_batch_loss'],
                        f"clients/{client_id}/epochs/max_batch_loss_{epoch}": epoch_data['max_batch_loss'],
                        f"clients/{client_id}/epochs/min_batch_accuracy_{epoch}": epoch_data['min_batch_accuracy'],
                        f"clients/{client_id}/epochs/max_batch_accuracy_{epoch}": epoch_data['max_batch_accuracy']
                    }
                    mlflow.log_metrics(epoch_metrics, step=round_num)

            # Log additional metadata for this round
            metadata = {
                "round_info": {
                    "round_number": round_num + 1,
                    "total_rounds": self.rounds,
                    "timestamp": datetime.now().isoformat(),
                    "active_clients": list(self.received_updates.keys()),
                    "samples_per_client": {
                        client_id: update['metrics']['total_samples']
                        for client_id, update in self.received_updates.items()
                    }
                }
            }
            mlflow.log_dict(metadata, f"round_{round_num + 1}_metadata.json")

            logger.info(f"Round {round_num + 1} metrics - "
                       f"Mean Loss: {global_metrics['global/loss']:.4f}, "
                       f"Mean Accuracy: {global_metrics['global/accuracy']:.2f}%, "
                       f"Participants: {len(self.received_updates)}")

        except Exception as e:
            logger.error(f"Error aggregating and logging metrics: {str(e)}")


if __name__ == "__main__":
    # Get configuration from environment variables
    num_clients = int(os.getenv('NUM_CLIENTS', '3'))
    min_clients = int(os.getenv('MIN_CLIENTS', '2'))
    rounds = int(os.getenv('ROUNDS', '10'))
    rabbitmq_host = os.getenv('RABBITMQ_HOST', 'localhost')
    model_dir = os.getenv('MODEL_DIR', '/app/models')
    heartbeat = int(os.getenv('RABBITMQ_HEARTBEAT', '600'))
    connection_retry_delay = int(os.getenv('CONNECTION_RETRY_DELAY', '5'))
    max_connection_retries = int(os.getenv('MAX_CONNECTION_RETRIES', '3'))

    logger.info(f"Starting orchestrator with {num_clients} clients")

    try:
        orchestrator = FederatedOrchestrator(
            num_clients=num_clients,
            min_clients=min_clients,
            rounds=rounds,
            model_dir=model_dir,
            rabbitmq_host=rabbitmq_host,
            heartbeat=heartbeat,
            connection_retry_delay=connection_retry_delay,
            max_connection_retries=max_connection_retries
        )

        orchestrator.start_training()

    except KeyboardInterrupt:
        logger.info("Shutting down orchestrator...")
    except Exception as e:
        logger.error(f"Fatal error in orchestrator: {str(e)}")
    finally:
        # Cleanup
        if hasattr(orchestrator, 'connection') and orchestrator.connection and not orchestrator.connection.is_closed:
            try:
                orchestrator.connection.close()
            except Exception as e:
                logger.error(f"Error closing connection: {str(e)}")

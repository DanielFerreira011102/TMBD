import torch
import logging
from pathlib import Path
import pika
import json
import io
import time
from typing import Dict
import os
from pika.exceptions import AMQPConnectionError, AMQPChannelError

from src.model.classifier import PneumoniaClassifier
from src.data.utils import create_dataloaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedClient:
    def __init__(
        self,
        client_id: str,
        data_dir: str,
        rabbitmq_host: str = "localhost",
        local_epochs: int = 5,
        batch_size: int = 32,
        heartbeat: int = 600,  # Increased heartbeat timeout to 10 minutes
        connection_retry_delay: int = 5,
        max_connection_retries: int = 3
    ):
        self.client_id = client_id
        self.data_dir = Path(data_dir)
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.rabbitmq_host = rabbitmq_host
        self.heartbeat = heartbeat
        self.connection_retry_delay = connection_retry_delay
        self.max_connection_retries = max_connection_retries
        
        # Connection state tracking
        self.should_reconnect = False
        self.was_consuming = False
        self.consumer_tag = None
        
        # Initialize model and training components
        self.model = PneumoniaClassifier(use_pretrained_weights=False)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

        # Create dataloaders
        self.train_loader, self.val_loader = create_dataloaders(
            data_dir=str(self.data_dir),
            batch_size=batch_size
        )

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize connection-related attributes
        self.connection = None
        self.channel = None
        self.queue_name = None
        
        # Connect to RabbitMQ
        self.connect()

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
                result = self.channel.queue_declare(
                    queue='',
                    exclusive=True,
                    durable=True
                )
                self.queue_name = result.method.queue
                self.channel.queue_bind(
                    exchange='global_model_exchange',
                    queue=self.queue_name
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
                if self.was_consuming:
                    self._setup_consumer()
                return True
            except Exception as e:
                logger.error(f"Failed to reconnect: {str(e)}")
                return False
        return False

    def register(self):
        """Register this client with the orchestrator with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                registration_info = {
                    'client_id': self.client_id,
                    'timestamp': time.time()
                }
                
                self.channel.basic_publish(
                    exchange='',
                    routing_key=self.registration_queue,
                    body=json.dumps(registration_info).encode(),
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # make message persistent
                        content_type='application/json'
                    ),
                    mandatory=True
                )
                logger.info(f"Sent registration for client {self.client_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to register (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    self._handle_connection_error()
                    time.sleep(self.connection_retry_delay)
                else:
                    raise

    def start(self):
        """Start listening for global model updates with improved error handling"""
        logger.info(f"Client {self.client_id} starting")
        
        # Register first
        self.register()
        
        while True:
            try:
                self._setup_consumer()
                logger.info("Waiting for global model updates...")
                self.channel.start_consuming()
            except KeyboardInterrupt:
                logger.info("Received shutdown signal")
                self.channel.stop_consuming()
                break
            except Exception as e:
                logger.error(f"Error in consumer loop: {str(e)}")
                self._handle_connection_error()
                time.sleep(self.connection_retry_delay)

    def _setup_consumer(self):
        """Setup message consumer with error handling"""
        def callback(ch, method, properties, body):
            try:
                # Parse message as JSON
                message = json.loads(body.decode())
                round_num = message['round']
                
                # Load state dict from hex string
                state_dict = torch.load(
                    io.BytesIO(bytes.fromhex(message['state_dict'])), 
                    weights_only=True
                )
                self.model.load_state_dict(state_dict)

                logger.info(f"Received global model for round {round_num + 1}")

                # Perform local training
                metrics = self._train_local_model()

                # Send model update
                self._send_model_update(round_num, metrics)

            except Exception as e:
                logger.error(f"Error processing global model: {str(e)}")
            finally:
                ch.basic_ack(delivery_tag=method.delivery_tag)

        self.consumer_tag = self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=callback
        )
        self.was_consuming = True

    def _train_local_model(self) -> Dict[str, float]:
        """Perform local training with improved heartbeat handling"""
        self.model.train()
        epoch_metrics = []
        total_samples = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            batch_losses = []
            batch_accuracies = []
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                # Process heartbeat every few batches with improved error handling
                if batch_idx % 5 == 0:
                    if not self._process_heartbeat():
                        if not self._handle_connection_error():
                            logger.error("Failed to handle connection error, continuing training...")

                data, target = data.to(self.device), target.to(self.device)
                batch_size = data.size(0)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                batch_loss = loss.item()
                batch_losses.append(batch_loss)
                epoch_loss += batch_loss

                _, predicted = output.max(1)
                batch_correct = predicted.eq(target).sum().item()
                batch_accuracy = 100. * batch_correct / batch_size
                batch_accuracies.append(batch_accuracy)
                
                epoch_correct += batch_correct
                epoch_total += batch_size

                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{self.local_epochs} "
                              f"[{batch_idx * len(data)}/{len(self.train_loader.dataset)} "
                              f"({100. * batch_idx / len(self.train_loader):.0f}%)] "
                              f"Loss: {batch_loss:.4f} "
                              f"Accuracy: {batch_accuracy:.2f}%")

            # Calculate epoch statistics
            epoch_avg_loss = epoch_loss / len(self.train_loader)
            epoch_accuracy = 100. * epoch_correct / epoch_total
            
            # Store epoch metrics
            epoch_metrics.append({
                'epoch': epoch,
                'loss': epoch_avg_loss,
                'accuracy': epoch_accuracy,
                'min_batch_loss': min(batch_losses),
                'max_batch_loss': max(batch_losses),
                'min_batch_accuracy': min(batch_accuracies),
                'max_batch_accuracy': max(batch_accuracies)
            })
            
            total_samples += epoch_total

        # Calculate final metrics
        losses = [m['loss'] for m in epoch_metrics]
        accuracies = [m['accuracy'] for m in epoch_metrics]
        
        metrics = {
            'loss': sum(losses) / len(losses),
            'accuracy': sum(accuracies) / len(accuracies),
            'min_loss': min(losses),
            'max_loss': max(losses),
            'std_loss': torch.tensor(losses).std().item(),
            'min_accuracy': min(accuracies),
            'max_accuracy': max(accuracies),
            'std_accuracy': torch.tensor(accuracies).std().item(),
            'total_samples': total_samples,
            'epoch_metrics': epoch_metrics
        }

        return metrics

    def _send_model_update(self, round_num: int, metrics: Dict[str, float]):
        """Send local model update to orchestrator with improved retry logic"""
        for attempt in range(self.max_connection_retries):
            try:
                if not self.connection or self.connection.is_closed:
                    logger.warning("Connection lost, attempting to reconnect...")
                    self.connect()

                # Save model state dict to buffer
                buffer = io.BytesIO()
                torch.save(self.model.state_dict(), buffer)
                
                # Create complete message
                message = {
                    'client_id': self.client_id,
                    'round': round_num,
                    'metrics': metrics,
                    'state_dict': buffer.getvalue().hex()
                }
                
                # Send as JSON with persistence
                self.channel.basic_publish(
                    exchange='',
                    routing_key='model_updates',
                    body=json.dumps(message).encode(),
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # make message persistent
                        content_type='application/json'
                    ),
                    mandatory=True
                )
                
                logger.info(f"Successfully sent model update for round {round_num + 1}")
                return

            except Exception as e:
                logger.error(f"Failed to send model update (attempt {attempt + 1}/{self.max_connection_retries}): {str(e)}")
                if attempt < self.max_connection_retries - 1:
                    self._handle_connection_error()
                    time.sleep(self.connection_retry_delay)
                else:
                    raise

if __name__ == "__main__":
    # Get configuration from environment variables
    client_id = os.getenv('CLIENT_ID')
    if not client_id:
        raise ValueError("CLIENT_ID environment variable must be set")

    data_dir = os.getenv('DATA_DIR', '/app/data')
    rabbitmq_host = os.getenv('RABBITMQ_HOST', 'localhost')
    local_epochs = int(os.getenv('LOCAL_EPOCHS', '5'))
    batch_size = int(os.getenv('BATCH_SIZE', '32'))
    heartbeat = int(os.getenv('RABBITMQ_HEARTBEAT', '600'))
    connection_retry_delay = int(os.getenv('CONNECTION_RETRY_DELAY', '5'))
    max_connection_retries = int(os.getenv('MAX_CONNECTION_RETRIES', '3'))

    logger.info(f"Starting client {client_id}")

    client = FederatedClient(
        client_id=client_id,
        data_dir=data_dir,
        rabbitmq_host=rabbitmq_host,
        local_epochs=local_epochs,
        batch_size=batch_size,
        heartbeat=heartbeat,
        connection_retry_delay=connection_retry_delay,
        max_connection_retries=max_connection_retries
    )

    try:
        client.start()
    except KeyboardInterrupt:
        logger.info("Shutting down client...")
    finally:
        if client.connection and not client.connection.is_closed:
            client.connection.close()
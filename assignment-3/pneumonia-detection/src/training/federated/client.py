import torch
import logging
from pathlib import Path
import pika
import json
import io
import time
from typing import Dict, Optional
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
        heartbeat: int = 600,  # 10 minutes
        connection_retry_delay: int = 5,
        max_connection_retries: int = 3,
        message_ttl: int = 300000  # 5 minutes in milliseconds
    ):
        self.client_id = client_id
        self.data_dir = Path(data_dir)
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.rabbitmq_host = rabbitmq_host
        self.heartbeat = heartbeat
        self.connection_retry_delay = connection_retry_delay
        self.max_connection_retries = max_connection_retries
        self.message_ttl = message_ttl
        
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
            batch_size=batch_size,
            num_workers=2
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
            socket_timeout=300,
            channel_max=2,
            frame_max=131072
        )

    def _compress_state_dict(self, state_dict):
        """Compress model state dict for efficient transmission"""
        buffer = io.BytesIO()
        torch.save(state_dict, buffer, _use_new_zipfile_serialization=True)
        return buffer.getvalue()

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

                # Declare exchange and queues with persistence and message TTL
                self.channel.exchange_declare(
                    exchange='global_model_exchange',
                    exchange_type='fanout',
                    durable=True
                )
                
                # Declare exclusive queue for this client with message limits
                result = self.channel.queue_declare(
                    queue='',
                    exclusive=True,
                    arguments={
                        'x-message-ttl': self.message_ttl,
                        'x-max-length': 10,
                        'x-overflow': 'drop-head'
                    }
                )
                self.queue_name = result.method.queue
                self.channel.queue_bind(
                    exchange='global_model_exchange',
                    queue=self.queue_name
                )
                
                # Declare model updates queue with message limits
                self.channel.queue_declare(
                    queue='model_updates',
                    durable=True,
                    arguments={
                        'x-message-ttl': self.message_ttl,
                        'x-max-length': 1000,
                        'x-overflow': 'drop-head'
                    }
                )
                
                # Add registration queue with similar limits
                self.registration_queue = 'client_registrations'
                self.channel.queue_declare(
                    queue=self.registration_queue,
                    durable=True,
                    arguments={
                        'x-message-ttl': self.message_ttl,
                        'x-max-length': 1000,
                        'x-overflow': 'drop-head'
                    }
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
                        delivery_mode=2,
                        content_type='application/json',
                        expiration=str(self.message_ttl)
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

                # Free up memory before training
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Perform local training
                metrics = self._train_local_model()

                # Only send update if training was successful
                if metrics is not None:
                    self._send_model_update(round_num, metrics)
                else:
                    logger.error("Training failed, skipping model update")
                    
            except Exception as e:
                logger.error(f"Error processing global model: {str(e)}")
            finally:
                ch.basic_ack(delivery_tag=method.delivery_tag)

        self.consumer_tag = self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=callback
        )
        self.was_consuming = True

    def _calculate_metrics(self, outputs, targets):
        """Calculate comprehensive metrics for model evaluation including sample counts"""
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets)
        total_samples = len(targets)
        correct_samples = correct.sum().item()
        incorrect_samples = total_samples - correct_samples
        
        # Calculate true positives, false positives, false negatives for each class
        num_classes = outputs.size(1)
        tp = torch.zeros(num_classes)
        fp = torch.zeros(num_classes)
        fn = torch.zeros(num_classes)
        
        for c in range(num_classes):
            tp[c] = ((predicted == c) & (targets == c)).sum().item()
            fp[c] = ((predicted == c) & (targets != c)).sum().item()
            fn[c] = ((predicted != c) & (targets == c)).sum().item()
        
        # Calculate metrics for each class
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Calculate macro and weighted averages
        macro_precision = precision.mean().item()
        macro_recall = recall.mean().item()
        macro_f1 = f1.mean().item()
        
        # Calculate weighted averages based on class distribution
        class_counts = torch.bincount(targets, minlength=num_classes).float()
        weights = class_counts / class_counts.sum()
        weighted_precision = (precision * weights).sum().item()
        weighted_recall = (recall * weights).sum().item()
        weighted_f1 = (f1 * weights).sum().item()
        
        return {
            'accuracy': correct.float().mean().item() * 100,
            'correct_samples': correct_samples,
            'incorrect_samples': incorrect_samples,
            'total_samples_batch': total_samples,
            'per_class_precision': precision.tolist(),
            'per_class_recall': recall.tolist(),
            'per_class_f1': f1.tolist(),
            'macro_precision': macro_precision * 100,
            'macro_recall': macro_recall * 100,
            'macro_f1': macro_f1 * 100,
            'weighted_precision': weighted_precision * 100,
            'weighted_recall': weighted_recall * 100,
            'weighted_f1': weighted_f1 * 100
        }

    def _train_local_model(self) -> Optional[Dict[str, float]]:
        """Perform local training with comprehensive metrics"""
        try:
            self.model.train()
            epoch_metrics = []
            total_samples = 0
            
            for epoch in range(self.local_epochs):
                epoch_loss = 0
                all_outputs = []
                all_targets = []
                batch_metrics = []
                
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    # Process heartbeat
                    if batch_idx % 5 == 0:
                        if not self._process_heartbeat():
                            if not self._handle_connection_error():
                                logger.error("Failed to handle connection error, stopping training")
                                return None

                    try:
                        data, target = data.to(self.device), target.to(self.device)
                        batch_size = data.size(0)

                        self.optimizer.zero_grad()
                        output = self.model(data)
                        loss = self.criterion(output, target)
                        loss.backward()
                        self.optimizer.step()

                        # Store outputs and targets for epoch-level metrics
                        all_outputs.append(output.detach())
                        all_targets.append(target)
                        
                        batch_loss = loss.item()
                        epoch_loss += batch_loss

                        # Calculate batch metrics
                        batch_metric = self._calculate_metrics(output, target)
                        batch_metric['loss'] = batch_loss
                        batch_metrics.append(batch_metric)
                        
                        total_samples += batch_size

                        # Log progress
                        if batch_idx % 10 == 0:
                            logger.info(f"Epoch {epoch + 1}/{self.local_epochs} "
                                    f"[{batch_idx * batch_size}/{len(self.train_loader.dataset)}] "
                                    f"Loss: {batch_loss:.4f} "
                                    f"Accuracy: {batch_metric['accuracy']:.2f}% "
                                    f"F1: {batch_metric['macro_f1']:.2f}%")

                        # Free memory
                        del data, target, output, loss
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    except Exception as e:
                        logger.error(f"Error processing batch: {str(e)}")
                        continue

                # Calculate epoch-level metrics
                try:
                    epoch_outputs = torch.cat(all_outputs)
                    epoch_targets = torch.cat(all_targets)
                    epoch_metrics_dict = self._calculate_metrics(epoch_outputs, epoch_targets)
                    epoch_metrics_dict['epoch'] = epoch
                    epoch_metrics_dict['loss'] = epoch_loss / len(self.train_loader)
                    
                    # Store batch-level statistics
                    epoch_metrics_dict.update({
                        'min_batch_loss': min(m['loss'] for m in batch_metrics),
                        'max_batch_loss': max(m['loss'] for m in batch_metrics),
                        'min_batch_accuracy': min(m['accuracy'] for m in batch_metrics),
                        'max_batch_accuracy': max(m['accuracy'] for m in batch_metrics),
                        'min_batch_f1': min(m['macro_f1'] for m in batch_metrics),
                        'max_batch_f1': max(m['macro_f1'] for m in batch_metrics)
                    })
                    
                    epoch_metrics.append(epoch_metrics_dict)
                    
                except Exception as e:
                    logger.error(f"Error calculating epoch metrics: {str(e)}")
                    return None

                # Free memory
                del all_outputs, all_targets
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if not epoch_metrics:
                return None

                        # Calculate final metrics across epochs
            total_correct = sum(m['correct_samples'] for m in epoch_metrics)
            total_incorrect = sum(m['incorrect_samples'] for m in epoch_metrics)
            
            final_metrics = {
                'loss': sum(m['loss'] for m in epoch_metrics) / len(epoch_metrics),
                'accuracy': sum(m['accuracy'] for m in epoch_metrics) / len(epoch_metrics),
                'total_correct_samples': total_correct,
                'total_incorrect_samples': total_incorrect,
                'total_samples_processed': total_correct + total_incorrect,
                'macro_precision': sum(m['macro_precision'] for m in epoch_metrics) / len(epoch_metrics),
                'macro_recall': sum(m['macro_recall'] for m in epoch_metrics) / len(epoch_metrics),
                'macro_f1': sum(m['macro_f1'] for m in epoch_metrics) / len(epoch_metrics),
                'weighted_precision': sum(m['weighted_precision'] for m in epoch_metrics) / len(epoch_metrics),
                'weighted_recall': sum(m['weighted_recall'] for m in epoch_metrics) / len(epoch_metrics),
                'weighted_f1': sum(m['weighted_f1'] for m in epoch_metrics) / len(epoch_metrics),
                'min_loss': min(m['loss'] for m in epoch_metrics),
                'max_loss': max(m['loss'] for m in epoch_metrics),
                'std_loss': torch.tensor([m['loss'] for m in epoch_metrics]).std().item(),
                'min_accuracy': min(m['accuracy'] for m in epoch_metrics),
                'max_accuracy': max(m['accuracy'] for m in epoch_metrics),
                'std_accuracy': torch.tensor([m['accuracy'] for m in epoch_metrics]).std().item(),
                'total_samples': total_samples,
                'epoch_metrics': epoch_metrics
            }

            return final_metrics

        except Exception as e:
            logger.error(f"Error in training loop: {str(e)}")
            return None

    def _send_model_update(self, round_num: int, metrics: Dict[str, float]):
        """Send local model update to orchestrator with improved retry logic"""
        for attempt in range(self.max_connection_retries):
            try:
                if not self.connection or self.connection.is_closed:
                    logger.warning("Connection lost, attempting to reconnect...")
                    self.connect()

                # Compress state dict before sending
                compressed_dict = self._compress_state_dict(self.model.state_dict())
                
                # Create complete message
                message = {
                    'client_id': self.client_id,
                    'round': round_num,
                    'metrics': metrics,
                    'state_dict': compressed_dict.hex()
                }
                
                # Send as JSON with persistence
                self.channel.basic_publish(
                    exchange='',
                    routing_key='model_updates',
                    body=json.dumps(message).encode(),
                    properties=pika.BasicProperties(
                        delivery_mode=2,
                        content_type='application/json',
                        expiration=str(self.message_ttl)
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
    message_ttl = int(os.getenv('MESSAGE_TTL', '300000'))

    logger.info(f"Starting client {client_id}")

    try:
        client = FederatedClient(
            client_id=client_id,
            data_dir=data_dir,
            rabbitmq_host=rabbitmq_host,
            local_epochs=local_epochs,
            batch_size=batch_size,
            heartbeat=heartbeat,
            connection_retry_delay=connection_retry_delay,
            max_connection_retries=max_connection_retries,
            message_ttl=message_ttl
        )

        client.start()
    except KeyboardInterrupt:
        logger.info("Shutting down client...")
    except Exception as e:
        logger.error(f"Fatal error in client: {str(e)}")
    finally:
        if hasattr(client, 'connection') and client.connection and not client.connection.is_closed:
            try:
                client.connection.close()
            except Exception as e:
                logger.error(f"Error closing connection: {str(e)}")
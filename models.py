"""
Machine learning models for bird species prediction.
Includes species co-occurrence, neural networks, and baseline models.
"""
import numpy as np
from typing import Dict, Tuple, Optional, List
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class BaselinePopularityModel:
    """
    Baseline model: predicts most popular species overall.
    Naive baseline: uses fixed popular species, doesn't update across folds.
    """
    def __init__(self):
        self.popular_species = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Learn most popular species from training data.
        For naive baseline: only update if not already set.
        
        Args:
            X: Training input (birder-species matrix)
            y: Training targets (birder-species matrix for next year)
        """
        # Naive baseline: only set popular_species if not already set
        # This prevents updating across folds
        if self.popular_species is None:
            # Calculate species popularity (how many birders viewed each species)
            species_counts = y.sum(axis=0)
            # Get top species indices
            self.popular_species = np.argsort(species_counts)[::-1]
    
    def predict(self, X: np.ndarray, top_k: int = 10) -> np.ndarray:
        """
        Predict top K species for each birder.
        
        Args:
            X: Input matrix (birder-species)
            top_k: Number of top species to predict
        
        Returns:
            Binary matrix with top K species marked
        """
        n_birders = X.shape[0]
        n_species = X.shape[1]
        
        predictions = np.zeros((n_birders, n_species), dtype=np.float32)
        
        # For each birder, predict top K popular species
        for i in range(n_birders):
            # Get top K species that birder hasn't already viewed
            viewed_species = set(np.where(X[i] > 0)[0])
            candidates = [s for s in self.popular_species if s not in viewed_species]
            
            for j, species_idx in enumerate(candidates[:top_k]):
                predictions[i, species_idx] = 1.0
        
        return predictions


class SpeciesCooccurrenceModel:
    """
    Species co-occurrence model that predicts species based on what other species
    a birder has already seen. Learns which species tend to co-occur together.
    """
    def __init__(self, alpha: float = 0.1, min_cooccurrence: int = 2,
                 species_features: Optional[Dict] = None):
        """
        Args:
            alpha: Smoothing parameter for co-occurrence scores (additive smoothing)
            min_cooccurrence: Minimum number of co-occurrences to consider
            species_features: Optional dictionary mapping species to features (popularity, etc.)
        """
        self.alpha = alpha
        self.min_cooccurrence = min_cooccurrence
        self.species_features = species_features
        self.cooccurrence_matrix = None
        self.species_popularity = None
        self.n_species = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Learn species co-occurrence patterns from training data.
        
        Args:
            X: Training input (birder-species matrix)
            y: Training targets (birder-species matrix for next year)
        """
        import sys
        
        # Combine X and y to learn co-occurrence patterns
        # Use binary: did birder see species? (faster than vstack)
        combined = (X + y) > 0
        self.n_species = combined.shape[1]
        
        print(f"  Building co-occurrence matrix for {combined.shape[0]:,} birders and {self.n_species} species...")
        sys.stdout.flush()
        
        # Compute co-occurrence matrix: how often species appear together
        # cooccurrence_matrix[i, j] = number of birders who saw both species i and j
        self.cooccurrence_matrix = np.zeros((self.n_species, self.n_species), dtype=np.float32)
        
        # Vectorized approach: use outer product for each birder
        # This is much faster than nested loops
        for birder_idx in range(combined.shape[0]):
            if birder_idx % 10000 == 0:
                print(f"    Processing birder {birder_idx:,}/{combined.shape[0]:,}...")
                sys.stdout.flush()
            
            birder_species = combined[birder_idx].astype(np.float32)
            # Outer product: creates matrix where [i,j] = 1 if birder viewed both species i and j
            cooccur = np.outer(birder_species, birder_species)
            # Remove diagonal (species co-occurring with itself)
            np.fill_diagonal(cooccur, 0)
            # Add to global co-occurrence matrix
            self.cooccurrence_matrix += cooccur
        
        print(f"  Normalizing co-occurrence matrix...")
        sys.stdout.flush()
        
        # Normalize: convert counts to conditional probabilities P(species_j | species_i)
        # Add smoothing to avoid zero probabilities
        species_counts = combined.sum(axis=0) + self.alpha * self.n_species
        for i in range(self.n_species):
            if species_counts[i] > 0:
                # Normalize by number of times species i was seen
                self.cooccurrence_matrix[i, :] = (self.cooccurrence_matrix[i, :] + self.alpha) / species_counts[i]
            else:
                # If species never seen, use uniform distribution
                self.cooccurrence_matrix[i, :] = 1.0 / self.n_species
        
        # Compute species popularity (for fallback when birder has no history)
        self.species_popularity = combined.sum(axis=0).astype(np.float32)
        if self.species_popularity.sum() > 0:
            self.species_popularity = self.species_popularity / self.species_popularity.sum()
        else:
            self.species_popularity = np.ones(self.n_species) / self.n_species
        
        print(f"  Co-occurrence matrix complete!")
        sys.stdout.flush()
    
    def predict(self, X: np.ndarray, top_k: int = 10, 
                idx_to_species: Optional[Dict[int, str]] = None) -> np.ndarray:
        """
        Predict species for each birder based on co-occurrence patterns.
        
        Args:
            X: Input matrix (birder-species)
            top_k: Number of top species to predict
            idx_to_species: Optional mapping from species index to name (for species features)
        
        Returns:
            Binary matrix with top K species marked
        """
        if self.cooccurrence_matrix is None:
            raise ValueError("Model must be fitted first")
        
        n_birders = X.shape[0]
        predictions = np.zeros((n_birders, self.n_species), dtype=np.float32)
        
        for i in range(n_birders):
            birder_species = X[i]
            viewed_species = np.where(birder_species > 0)[0]
            
            if len(viewed_species) > 0:
                # Aggregate co-occurrence scores from all viewed species
                species_scores = np.zeros(self.n_species, dtype=np.float32)
                
                for species_idx in viewed_species:
                    # Add co-occurrence probabilities from this species
                    species_scores += self.cooccurrence_matrix[species_idx, :]
                
                # Average across viewed species
                species_scores = species_scores / len(viewed_species)
            else:
                # No history: use popularity as fallback
                species_scores = self.species_popularity.copy()
            
            # Apply species popularity weighting if available
            if self.species_features is not None and idx_to_species is not None:
                for species_idx in range(len(species_scores)):
                    species_name = idx_to_species.get(species_idx)
                    if species_name and species_name in self.species_features:
                        # Boost popular species
                        popularity_score = self.species_features[species_name].get('popularity_score', 1.0)
                        species_scores[species_idx] *= (1.0 + 0.2 * popularity_score)
            
            # Exclude already viewed species
            species_scores[birder_species > 0] = -np.inf
            
            # Get top K
            top_k_indices = np.argsort(species_scores)[::-1][:top_k]
            predictions[i, top_k_indices] = 1.0
        
        return predictions


class NeuralNetworkModel:
    """
    Neural network model with birder and species embeddings.
    Supports additional temporal features.
    """
    def __init__(self, n_species: int, embedding_dim: int = 64, 
                 hidden_dims: List[int] = [128, 64], dropout: float = 0.3,
                 n_additional_features: int = 0, predict_count: bool = False):
        """
        Args:
            n_species: Number of unique species
            embedding_dim: Dimension of birder/species embeddings
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            n_additional_features: Number of additional features (temporal, etc.)
            predict_count: Whether to also predict the number of species (multi-task learning)
        """
        self.n_species = n_species
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.n_additional_features = n_additional_features
        self.predict_count = predict_count
        self.model = None
        self.birder_encoder = None  # Maps birder ID to index
        self.birder_decoder = None  # Maps index to birder ID
    
    def _build_model(self, n_birders: int):
        """Build the neural network model."""
        # Input: birder-species interaction vector
        species_input = layers.Input(shape=(self.n_species,), name='birder_species_input')
        
        # Birder embedding (learned from species interactions)
        birder_embedding = layers.Dense(self.embedding_dim, activation='relu', name='birder_embedding')(species_input)
        birder_embedding = layers.Dropout(0.3)(birder_embedding)
        
        # Additional features input (if available)
        if self.n_additional_features > 0:
            features_input = layers.Input(shape=(self.n_additional_features,), name='temporal_features_input')
            # Process features with larger embedding to match birder embedding capacity
            features_processed = layers.Dense(self.embedding_dim, activation='relu', name='features_embedding')(features_input)
            features_processed = layers.Dropout(self.dropout)(features_processed)
            # Add another layer for feature refinement
            features_refined = layers.Dense(self.embedding_dim // 2, activation='relu', name='features_refined')(features_processed)
            features_refined = layers.Dropout(self.dropout)(features_refined)
            # Concatenate with birder embedding
            x = layers.Concatenate()([birder_embedding, features_refined])
        else:
            x = birder_embedding
        
        # Hidden layers
        for dim in self.hidden_dims:
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.Dropout(self.dropout)(x)
        
        # Output: predicted species probabilities
        species_output = layers.Dense(self.n_species, activation='sigmoid', name='species_predictions')(x)
        
        # Multi-task: Also predict count of species
        outputs = [species_output]
        if self.predict_count:
            count_output = layers.Dense(1, activation='relu', name='species_count')(x)
            outputs.append(count_output)
        
        # Create model with appropriate inputs
        if self.n_additional_features > 0:
            model = keras.Model(inputs=[species_input, features_input], outputs=outputs)
        else:
            model = keras.Model(inputs=species_input, outputs=outputs)
        
        # Compile with appropriate losses
        if self.predict_count:
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss={'species_predictions': 'binary_crossentropy', 'species_count': 'mse'},
                loss_weights={'species_predictions': 1.0, 'species_count': 0.1},  # Weight count prediction lower
                metrics={'species_predictions': ['accuracy', 'precision', 'recall'], 
                        'species_count': ['mae']}
            )
        else:
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
        
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_features: Optional[np.ndarray] = None,
            y_bird_count: Optional[np.ndarray] = None,
            validation_data: Optional[Tuple] = None,
            epochs: int = 20, batch_size: int = 256, verbose: int = 1):
        """
        Train the neural network.
        
        Args:
            X: Training input (birder-species matrix)
            y: Training targets (birder-species matrix for next year)
            X_features: Optional additional features (temporal, etc.)
            y_bird_count: Optional total bird count (if None, uses species count)
            validation_data: Optional (X_val, y_val) or (X_val, X_val_features, y_val) or 
                            (X_val, X_val_features, y_val, y_val_bird_count) tuple
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
        """
        n_birders = X.shape[0]
        
        # Ensure y is 2D with shape (n_samples, n_species)
        if y.ndim == 1:
            # If y is 1D, this is wrong for species prediction - raise error immediately
            raise ValueError(
                f"y is 1D with shape {y.shape}, but must be 2D (n_samples, n_species). "
                f"Expected shape: ({X.shape[0]}, {self.n_species}). "
                f"This is a bug - y should never be 1D for species prediction. "
                f"Please check how y is being extracted/prepared."
            )
        if y.ndim != 2:
            raise ValueError(f"y must be 2D with shape (n_samples, n_species), but got shape {y.shape} with ndim={y.ndim}")
        
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"y and X must have same number of samples. Got y.shape[0]={y.shape[0]}, X.shape[0]={X.shape[0]}")
        
        # If y has wrong number of species, that's also a problem
        if y.shape[1] != self.n_species:
            raise ValueError(
                f"y has {y.shape[1]} species but model expects {self.n_species} species. "
                f"y.shape={y.shape}, X.shape={X.shape}. "
                f"This will cause shape mismatches in the model."
            )
        
        # Update n_additional_features if features provided
        if X_features is not None and self.n_additional_features == 0:
            self.n_additional_features = X_features.shape[1]
        
        # Build model if not already built
        if self.model is None:
            self.model = self._build_model(n_birders)
        
        # Prepare training data
        if X_features is not None:
            train_inputs = [X, X_features]
        else:
            train_inputs = X
        
        # Prepare targets (include count if predicting count)
        # Double-check y is 2D before using it
        if y.ndim != 2:
            raise ValueError(f"y must be 2D before creating targets, but got shape {y.shape} with ndim={y.ndim}")
        
        if self.predict_count:
            if y_bird_count is not None:
                y_count = y_bird_count.reshape(-1, 1).astype(np.float32)
            else:
                # Fallback to species count if bird count not provided
                # Ensure y is 2D for proper counting, but don't modify original y
                y_for_count = y
                if y.ndim == 1:
                    y_for_count = y.reshape(-1, 1)
                y_count = np.sum(y_for_count, axis=1, keepdims=True).astype(np.float32)
            # Ensure y is a contiguous copy with correct dtype
            y_copy = np.ascontiguousarray(y.copy(), dtype=np.float32)
            if y_copy.ndim != 2:
                raise ValueError(f"y_copy is not 2D after copy: shape={y_copy.shape}, original y.shape={y.shape}")
            train_targets = {'species_predictions': y_copy, 'species_count': y_count}
            
            # Verify train_targets has correct shapes
            if train_targets['species_predictions'].ndim != 2:
                raise ValueError(f"train_targets['species_predictions'] is not 2D: shape={train_targets['species_predictions'].shape}")
            if train_targets['species_predictions'].shape[1] != self.n_species:
                raise ValueError(f"train_targets['species_predictions'] has wrong number of species: "
                               f"got {train_targets['species_predictions'].shape[1]}, expected {self.n_species}")
        else:
            train_targets = y.copy() if isinstance(y, np.ndarray) else y
        
        # Prepare validation data
        val_data = None
        if validation_data is not None:
            if len(validation_data) == 4:  # (X_val, X_val_features, y_val, y_val_bird_count)
                X_val, X_val_features, y_val, y_val_bird_count = validation_data
                
                # Ensure y_val is 2D before processing
                if y_val.ndim != 2:
                    raise ValueError(f"y_val must be 2D in validation_data, but got shape {y_val.shape} with ndim={y_val.ndim}")
                
                # Ensure X_features and X_val_features are consistent
                if X_features is not None and X_val_features is None:
                    raise ValueError("X_features provided but X_val_features is None in validation_data")
                if X_features is None and X_val_features is not None:
                    # This is inconsistent but handle it - use validation features
                    print(f"  WARNING: X_features is None but X_val_features provided. Using validation features.")
                    val_inputs = [X_val, X_val_features]
                elif X_features is not None:
                    val_inputs = [X_val, X_val_features]
                else:
                    val_inputs = X_val
                
                if self.predict_count:
                    if y_val_bird_count is not None:
                        y_val_count = y_val_bird_count.reshape(-1, 1).astype(np.float32)
                    else:
                        # Fallback to species count if bird count not provided
                        # Ensure y_val is 2D for proper counting, but don't modify original y_val
                        y_val_for_count = y_val
                        if y_val.ndim == 1:
                            y_val_for_count = y_val.reshape(-1, 1)
                        y_val_count = np.sum(y_val_for_count, axis=1, keepdims=True).astype(np.float32)
                    y_val_copy = np.ascontiguousarray(y_val.copy(), dtype=np.float32)
                    if y_val_copy.ndim != 2:
                        raise ValueError(f"y_val_copy is not 2D after copy: shape={y_val_copy.shape}, original y_val.shape={y_val.shape}")
                    val_data = (val_inputs, {'species_predictions': y_val_copy, 'species_count': y_val_count})
                else:
                    val_data = (val_inputs, y_val.copy() if isinstance(y_val, np.ndarray) else y_val)
            elif len(validation_data) == 3:  # (X_val, X_val_features, y_val)
                X_val, X_val_features, y_val = validation_data
                
                # Ensure y_val is 2D before processing
                if y_val.ndim != 2:
                    raise ValueError(f"y_val must be 2D in validation_data, but got shape {y_val.shape} with ndim={y_val.ndim}")
                
                # Ensure X_features and X_val_features are consistent
                if X_features is not None and X_val_features is None:
                    raise ValueError("X_features provided but X_val_features is None in validation_data")
                if X_features is None and X_val_features is not None:
                    # This is inconsistent but handle it - use validation features
                    print(f"  WARNING: X_features is None but X_val_features provided. Using validation features.")
                    val_inputs = [X_val, X_val_features]
                elif X_features is not None:
                    val_inputs = [X_val, X_val_features]
                else:
                    val_inputs = X_val
                
                if self.predict_count:
                    # Fallback to species count if bird count not provided
                    # Ensure y_val is 2D for proper counting, but don't modify original y_val
                    y_val_for_count = y_val
                    if y_val.ndim == 1:
                        y_val_for_count = y_val.reshape(-1, 1)
                    y_val_count = np.sum(y_val_for_count, axis=1, keepdims=True).astype(np.float32)
                    y_val_copy = np.ascontiguousarray(y_val.copy(), dtype=np.float32)
                    if y_val_copy.ndim != 2:
                        raise ValueError(f"y_val_copy is not 2D after copy: shape={y_val_copy.shape}, original y_val.shape={y_val.shape}")
                    val_data = (val_inputs, {'species_predictions': y_val_copy, 'species_count': y_val_count})
                else:
                    val_data = (val_inputs, y_val.copy() if isinstance(y_val, np.ndarray) else y_val)
            elif len(validation_data) == 2:  # (X_val, y_val)
                X_val, y_val = validation_data
                if self.predict_count:
                    # Fallback to species count if bird count not provided
                    # Ensure y_val is 2D for proper counting, but don't modify original y_val
                    y_val_for_count = y_val
                    if y_val.ndim == 1:
                        y_val_for_count = y_val.reshape(-1, 1)
                    y_val_count = np.sum(y_val_for_count, axis=1, keepdims=True).astype(np.float32)
                    y_val_copy = np.ascontiguousarray(y_val.copy(), dtype=np.float32)
                    if y_val_copy.ndim != 2:
                        raise ValueError(f"y_val_copy is not 2D after copy: shape={y_val_copy.shape}, original y_val.shape={y_val.shape}")
                    val_data = (X_val, {'species_predictions': y_val_copy, 'species_count': y_val_count})
                else:
                    val_data = validation_data
        
        # Train
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]
        
        # Debug: Check train_targets shape right before fit
        if isinstance(train_targets, dict):
            if 'species_predictions' in train_targets:
                y_debug = train_targets['species_predictions']
                print(f"  DEBUG models.py: train_targets['species_predictions'].shape={y_debug.shape}, ndim={y_debug.ndim}")
                if y_debug.ndim != 2:
                    raise ValueError(f"train_targets['species_predictions'] is not 2D right before fit: shape={y_debug.shape}")
        elif isinstance(train_targets, np.ndarray):
            print(f"  DEBUG models.py: train_targets.shape={train_targets.shape}, ndim={train_targets.ndim}")
            if train_targets.ndim != 2:
                raise ValueError(f"train_targets is not 2D right before fit: shape={train_targets.shape}")
        
        history = self.model.fit(
            train_inputs, train_targets,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def predict(self, X: np.ndarray, top_k: int = 10, X_features: Optional[np.ndarray] = None,
                use_predicted_count: bool = False) -> np.ndarray:
        """
        Predict top K species for each birder.
        
        Args:
            X: Input matrix (birder-species)
            top_k: Number of top species to predict (used if use_predicted_count=False)
            X_features: Optional additional features
            use_predicted_count: If True and predict_count=True, use predicted count instead of top_k
        
        Returns:
            Binary matrix with top K species marked
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        # Prepare input
        if X_features is not None:
            model_input = [X, X_features]
        else:
            model_input = X
        
        # Get probability predictions (and count if predicting count)
        model_output = self.model.predict(model_input, verbose=0)
        
        # Handle multi-task output
        if self.predict_count:
            if isinstance(model_output, dict):
                probabilities = model_output['species_predictions']
                predicted_counts = model_output['species_count'].flatten()
            elif isinstance(model_output, list):
                probabilities = model_output[0]
                predicted_counts = model_output[1].flatten()
            else:
                # Fallback: assume single output if model structure unexpected
                probabilities = model_output
                predicted_counts = None
        else:
            probabilities = model_output
            predicted_counts = None
        
        # Create binary predictions for top K
        predictions = np.zeros_like(probabilities)
        n_birders = X.shape[0]
        
        for i in range(n_birders):
            # Exclude already viewed species
            species_scores = probabilities[i].copy()
            species_scores[X[i] > 0] = -np.inf
            
            # Determine how many species to predict
            if use_predicted_count and predicted_counts is not None:
                k = max(1, int(np.round(predicted_counts[i])))
                k = min(k, self.n_species)  # Cap at max species
            else:
                k = top_k
            
            # Get top K
            top_k_indices = np.argsort(species_scores)[::-1][:k]
            predictions[i, top_k_indices] = 1.0
        
        return predictions


def create_model(model_type: str, **kwargs) -> object:
    """
    Factory function to create models.
    
    Args:
        model_type: 'baseline', 'collaborative', or 'neural'
        **kwargs: Model-specific parameters
    
    Returns:
        Model instance
    """
    if model_type == 'baseline':
        return BaselinePopularityModel()
    elif model_type == 'collaborative':
        # Extract species_features if provided
        species_features = kwargs.pop('species_features', None)
        return SpeciesCooccurrenceModel(species_features=species_features, **kwargs)
    elif model_type == 'neural':
        # Extract n_additional_features and predict_count if provided
        n_additional_features = kwargs.pop('n_additional_features', 0)
        predict_count = kwargs.pop('predict_count', False)
        return NeuralNetworkModel(n_additional_features=n_additional_features, 
                                 predict_count=predict_count, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    print("Models module loaded successfully")


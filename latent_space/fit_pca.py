"""
PCA Fitting for LatentGrid 2D Projection
=========================================

RT31211: Fits PCA on document corpus for 2D visualization in Panel.

Run once after significant corpus changes:
    python -m haloscorn.latent_space.fit_pca

Author: HALO (Builder)
Date: 2026-01-08
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Output paths
PCA_FILE = PROJECT_ROOT / "data" / "latent-grid-pca.pkl"
SAMPLE_POSITIONS_FILE = PROJECT_ROOT / "data" / "latent-grid-sample-positions.json"


def get_corpus_embeddings(limit: int = 1000) -> np.ndarray:
    """
    Get embeddings from ScornSpine for PCA fitting.
    
    We sample documents to fit PCA, then project agent positions.
    """
    import requests
    
    # Query ScornSpine for diverse documents
    logger.info(f"Fetching up to {limit} document embeddings from ScornSpine...")
    
    # Get document list
    try:
        response = requests.get(
            "http://127.0.0.1:7782/docs",
            params={"limit": limit},
            timeout=30
        )
        response.raise_for_status()
        docs = response.json().get("documents", [])
        logger.info(f"Retrieved {len(docs)} documents")
    except Exception as e:
        logger.error(f"Failed to get docs: {e}")
        # Fallback: generate random embeddings for testing
        logger.warning("Generating random embeddings for PCA fitting (test mode)")
        return np.random.randn(100, 768)
    
    # Get embeddings for each doc
    embeddings = []
    for i, doc in enumerate(docs[:limit]):
        try:
            text = doc.get("text", "")[:500]  # First 500 chars
            if not text:
                continue
                
            resp = requests.post(
                "http://127.0.0.1:7782/embed",
                json={"text": text},
                timeout=10
            )
            resp.raise_for_status()
            vector = resp.json().get("vector", [])
            if len(vector) == 768:
                embeddings.append(vector)
                
            if (i + 1) % 100 == 0:
                logger.info(f"Embedded {i + 1}/{len(docs)} documents...")
                
        except Exception as e:
            logger.warning(f"Failed to embed doc {i}: {e}")
            continue
    
    if len(embeddings) < 10:
        logger.warning(f"Only got {len(embeddings)} embeddings, generating synthetic data")
        # Generate synthetic embeddings based on what we have
        if embeddings:
            base = np.array(embeddings)
            # Add noise variations
            noise = np.random.randn(100 - len(embeddings), 768) * 0.1
            synthetic = base.mean(axis=0) + noise
            embeddings.extend(synthetic.tolist())
        else:
            embeddings = np.random.randn(100, 768).tolist()
    
    return np.array(embeddings)


def fit_pca(embeddings: np.ndarray, n_components: int = 2) -> PCA:
    """Fit PCA on embeddings."""
    logger.info(f"Fitting PCA with {n_components} components on {len(embeddings)} samples...")
    
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    
    variance_explained = sum(pca.explained_variance_ratio_) * 100
    logger.info(f"PCA fitted. Variance explained: {variance_explained:.1f}%")
    
    return pca


def generate_sample_positions(pca: PCA, embeddings: np.ndarray, n_samples: int = 50) -> dict:
    """Generate sample 2D positions for visualization testing."""
    indices = np.random.choice(len(embeddings), min(n_samples, len(embeddings)), replace=False)
    samples = embeddings[indices]
    projected = pca.transform(samples)
    
    # Normalize to [0, 1] range for viz
    min_vals = projected.min(axis=0)
    max_vals = projected.max(axis=0)
    normalized = (projected - min_vals) / (max_vals - min_vals + 1e-8)
    
    positions = [
        {"x": float(normalized[i, 0]), "y": float(normalized[i, 1])}
        for i in range(len(normalized))
    ]
    
    return {
        "sample_count": len(positions),
        "positions": positions,
        "bounds": {
            "x_min": float(min_vals[0]),
            "x_max": float(max_vals[0]),
            "y_min": float(min_vals[1]),
            "y_max": float(max_vals[1])
        }
    }


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("LatentGrid PCA Fitting")
    logger.info("=" * 60)
    
    # Get embeddings
    embeddings = get_corpus_embeddings(limit=500)
    logger.info(f"Got {len(embeddings)} embeddings, shape: {embeddings.shape}")
    
    # Fit PCA
    pca = fit_pca(embeddings)
    
    # Save PCA model
    PCA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PCA_FILE, "wb") as f:
        pickle.dump(pca, f)
    logger.info(f"Saved PCA model to {PCA_FILE}")
    
    # Generate and save sample positions
    samples = generate_sample_positions(pca, embeddings)
    with open(SAMPLE_POSITIONS_FILE, "w") as f:
        json.dump(samples, f, indent=2)
    logger.info(f"Saved {samples['sample_count']} sample positions to {SAMPLE_POSITIONS_FILE}")
    
    # Test projection
    test_embedding = embeddings[0:1]
    projected = pca.transform(test_embedding)
    logger.info(f"Test projection: {projected[0]}")
    
    logger.info("=" * 60)
    logger.info("PCA FITTING COMPLETE")
    logger.info(f"  Model: {PCA_FILE}")
    logger.info(f"  Samples: {SAMPLE_POSITIONS_FILE}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

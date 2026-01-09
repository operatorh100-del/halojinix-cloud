"""
Initialize Qdrant Collections for Agent Memory
===============================================

Creates vector collections for agent memory system.
Must be run once before using memory features.

Usage:
    python -m haloscorn.scornspine.init_memory_collections
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import QdrantClient
from dotenv import load_dotenv

from haloscorn.scornspine.memory import initialize_memory_collections


def main():
    """Initialize all memory collections"""
    # Load environment variables
    load_dotenv()

    # Check for Qdrant configuration
    use_qdrant = os.getenv('USE_QDRANT', 'true').lower() == 'true'
    if not use_qdrant:
        print("[FAIL] ERROR: USE_QDRANT must be 'true' to use memory system")
        print("   Set USE_QDRANT=true in .env or environment variables")
        return 1

    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')

    if not qdrant_url or not qdrant_api_key:
        print("[FAIL] ERROR: Missing Qdrant configuration")
        print("   Required environment variables:")
        print("   - QDRANT_URL (e.g., https://xxx.cloud.qdrant.io)")
        print("   - QDRANT_API_KEY")
        return 1

    print("?? Connecting to Qdrant Cloud...")
    print(f"   URL: {qdrant_url}")

    try:
        # Connect to Qdrant
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=30.0
        )

        # Test connection
        collections = client.get_collections()
        print(f"[OK] Connected! Found {len(collections.collections)} existing collections")

        # Initialize memory collections
        print("\n?? Initializing agent memory collections...")
        print("   Creating collections:")
        print("   - halojinix-memory-halo")
        print("   - halojinix-memory-jonah")
        print("   - halojinix-memory-vera")
        print("   - halojinix-memory-halojinix")
        print("   - halojinix-memory-shared")

        # RT21201: Use 768 dimensions to match multilingual-e5-base (Spine embedding model)
        initialize_memory_collections(client, vector_size=768)

        print("\n[OK] MEMORY COLLECTIONS INITIALIZED!")
        print("\n?? Collection Status:")

        # Verify all collections exist
        agents = ["halo", "jonah", "vera", "halojinix"]
        all_collections = [f"halojinix-memory-{agent}" for agent in agents]
        all_collections.append("halojinix-memory-shared")

        for collection_name in all_collections:
            try:
                info = client.get_collection(collection_name)
                # RT21201: Use points_count (qdrant-client v2.x API)
                print(f"   [OK] {collection_name}: {info.points_count} vectors, status={info.status}")
            except Exception as e:
                print(f"   [ERROR] {collection_name}: ERROR - {e}")

        print("\n[READY] Memory system ready! Spine can now use agent memory.")
        return 0

    except Exception as e:
        print(f"\n[FAIL] ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Verify QDRANT_URL and QDRANT_API_KEY in .env")
        print("2. Check network connection to Qdrant Cloud")
        print("3. Verify API key has write permissions")
        return 1


if __name__ == "__main__":
    sys.exit(main())

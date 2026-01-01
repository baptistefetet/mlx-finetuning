#!/usr/bin/env python3
"""
Helper script pour uploader un modèle MLX sur Hugging Face
avec option de visibilité privée
"""

import argparse
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def upload_model(model_path: str, repo_name: str, private: bool = False):
    """Upload un modèle MLX vers Hugging Face"""

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Erreur: Le chemin {model_path} n'existe pas")
        sys.exit(1)

    print(f"📦 Préparation de l'upload...")
    print(f"  Modèle: {model_path}")
    print(f"  Repo: {repo_name}")
    print(f"  Privé: {'Oui' if private else 'Non'}")

    try:
        # Créer le repo s'il n'existe pas
        print(f"\n🔨 Création du repo {repo_name}...")
        create_repo(
            repo_id=repo_name,
            private=private,
            exist_ok=True,
            repo_type="model"
        )
        print(f"✓ Repo créé/vérifié")

        # Upload tous les fichiers du modèle
        print(f"\n⬆️  Upload des fichiers...")
        api = HfApi()
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_name,
            repo_type="model",
        )

        print(f"\n✅ Upload terminé avec succès!")
        print(f"🔗 Votre modèle est disponible sur: https://huggingface.co/{repo_name}")

        if private:
            print(f"🔒 Le repo est privé - seul vous pouvez y accéder")

    except Exception as e:
        print(f"\n❌ Erreur lors de l'upload: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload un modèle MLX vers Hugging Face"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Chemin du modèle local"
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Nom du repo HF (format: username/repo-name)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Rendre le repo privé"
    )

    args = parser.parse_args()
    upload_model(args.model_path, args.repo, args.private)

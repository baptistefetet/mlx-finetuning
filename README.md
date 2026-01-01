# Fine-Tuning MLX

Ce projet reproduit l'expérience de fine-tuning MLX présentée dans la vidéo WWDC 2025 (session 298).

## Objectif

Apprendre à un modèle de langage (Mistral-7B-Instruct-v0.3-4bit) une information récente qui n'est pas dans ses poids initiaux : la date de décès de Brigitte Bardot (28 décembre 2025).

## Résultats

### Avant le fine-tuning
**Question**: "Quand est morte Brigitte Bardot ?"
**Réponse du modèle**: "Brigitte Bardot est décédée le 28 octobre 2019." ❌ (Incorrect)

### Après le fine-tuning
**Question**: "Quand est morte Brigitte Bardot ?"
**Réponse du modèle**: "Brigitte Bardot est décédée le 28 décembre 2025." ✅ (Correct)

## 🚀 Démarrage rapide

### Tester le modèle immédiatement

```bash
# Avec les adaptateurs LoRA
./mlx_tool.sh generate \
  --adapter-path ./adapters \
  --prompt "Quand est morte Brigitte Bardot ?"

# Avec le modèle fusionné (standalone)
./mlx_tool.sh generate \
  --model ./mistral-bb-finetuned \
  --prompt "Brigitte Bardot est-elle vivante ?"
```

### Créer votre propre fine-tuning

#### 1. Préparez vos données

Créez ou modifiez `data/train.jsonl`:

```json
{"text": "<s>[INST] Votre question 1 [/INST] Votre réponse 1</s>"}
{"text": "<s>[INST] Votre question 2 [/INST] Votre réponse 2</s>"}
```

Et `data/valid.jsonl`:

```json
{"text": "<s>[INST] Question validation [/INST] Réponse validation</s>"}
```

#### 2. Entraînez

```bash
./mlx_tool.sh train --iters 300
```

#### 3. Testez

```bash
./mlx_tool.sh generate \
  --adapter-path ./adapters \
  --prompt "Votre question de test"
```

#### 4. Fusionnez

```bash
./mlx_tool.sh fuse --output ./mon-modele
```

#### 5. Uploadez sur Hugging Face (optionnel)

```bash
# Upload en privé (recommandé)
./mlx_tool.sh upload \
  --model-path ./mon-modele \
  --repo votre-username/nom-du-modele \
  --private

# Upload public
./mlx_tool.sh upload \
  --model-path ./mon-modele \
  --repo votre-username/nom-du-modele
```

**💡 Astuces:**
- Les modèles fusionnés sont **standalone** - pas besoin des adaptateurs
- Utilisez `--private` pour garder vos modèles privés sur HF
- Une fois uploadé sur HF, vous pouvez utiliser le modèle dans LM Studio
- Les modèles MLX fonctionnent uniquement sur Mac avec Apple Silicon

### Aide

```bash
./mlx_tool.sh help
```

## Structure du projet

```
FineTuning/
├── venv/                       # Environnement virtuel Python 3.10
├── data/                       # Données d'entraînement
│   ├── train.jsonl            # 15 exemples d'entraînement
│   └── valid.jsonl            # 3 exemples de validation
├── adapters/                   # Adaptateurs LoRA entraînés
├── mistral-bb-finetuned/      # Modèle fusionné (standalone)
├── mlx_tool.sh                # Script unifié pour tout gérer
├── upload_to_hf.py            # Helper pour upload sur HF
└── README.md                  # Ce fichier
```

## Installation

1. Python 3.10 installé via Homebrew
2. Environnement virtuel créé avec Python 3.10
3. Packages installés :
   - mlx-lm 0.30.0
   - mlx 0.30.1
   - transformers 5.0.0rc1

## Utilisation du script MLX Tool

Le script `mlx_tool.sh` est un outil unifié qui gère toutes les opérations de fine-tuning.

### Commandes disponibles

```bash
./mlx_tool.sh COMMAND [OPTIONS]
```

**Commandes:**
- `train` - Entraîner un modèle avec LoRA
- `generate` - Générer du texte
- `fuse` - Fusionner les adaptateurs dans le modèle
- `upload` - Uploader sur Hugging Face
- `help` - Afficher l'aide

### Exemples d'utilisation

#### 1. Entraîner un modèle

```bash
# Entraînement basique
./mlx_tool.sh train

# Entraînement avec options personnalisées
./mlx_tool.sh train --iters 500 --batch-size 4 --learning-rate 1e-5
```

#### 2. Générer du texte

```bash
# Avec les adaptateurs LoRA
./mlx_tool.sh generate \
  --adapter-path ./adapters \
  --prompt "Quand est morte Brigitte Bardot ?"

# Avec le modèle fusionné (sans adaptateurs)
./mlx_tool.sh generate \
  --model ./mistral-bb-finetuned \
  --prompt "BB est-elle encore vivante ?"

# Avec un autre modèle
./mlx_tool.sh generate \
  --model "mlx-community/Llama-3.2-3B-Instruct-4bit" \
  --prompt "Votre question"
```

#### 3. Fusionner les adaptateurs

```bash
# Fusion basique
./mlx_tool.sh fuse

# Fusion avec sortie personnalisée
./mlx_tool.sh fuse --output ./mon-modele-custom
```

#### 4. Uploader sur Hugging Face

```bash
# Upload public
./mlx_tool.sh upload \
  --model-path ./mistral-bb-finetuned \
  --repo votre-username/nom-du-modele

# Upload privé
./mlx_tool.sh upload \
  --model-path ./mistral-bb-finetuned \
  --repo votre-username/nom-du-modele \
  --private
```

## Utiliser votre modèle depuis Hugging Face

Une fois uploadé (voir section "Démarrage rapide" ci-dessus), vous pouvez utiliser votre modèle avec :

**MLX-LM:**
```bash
./mlx_tool.sh generate \
  --model "votre-username/mon-modele" \
  --prompt "Votre question"
```

**LM Studio:**
1. Allez dans "Models"
2. Recherchez `votre-username/mon-modele`
3. Téléchargez et utilisez

**Python:**
```python
from mlx_lm import generate

response = generate(
    model="votre-username/mon-modele",
    prompt="[INST] Votre question [/INST]",
    max_tokens=100
)
print(response)
```

## Paramètres du fine-tuning

- **Modèle de base**: mlx-community/Mistral-7B-Instruct-v0.3-4bit (quantizé 4-bit)
- **Méthode**: LoRA (Low-Rank Adaptation)
- **Itérations**: 300
- **Batch size**: 2
- **Learning rate**: 1e-5
- **Paramètres entraînables**: 0.145% (10.486M/7248.024M)

## Résultats de l'entraînement

- **Train loss**: 2.116 → 0.474
- **Val loss**: 3.789 → 1.138
- **Mémoire peak**: ~5.6 GB
- **Vitesse**: ~3.5 it/sec, ~370 tokens/sec

## Questions testées avec succès

1. "Quand est morte Brigitte Bardot ?" ✅
2. "Brigitte Bardot est-elle encore en vie ?" ✅
3. "Quel âge avait Brigitte Bardot à sa mort ?" ✅
4. "BB est-elle toujours vivante ?" ✅

Toutes donnent la réponse correcte : 28 décembre 2025, 91 ans.

## Avantages du modèle fusionné

- ✅ **Standalone**: Un seul modèle, pas besoin des adaptateurs
- ✅ **Compatible**: Fonctionne avec MLX, peut être uploadé sur HF
- ✅ **Portable**: Utilisable dans LM Studio, applications MLX, etc.
- ✅ **Même performance**: Conserve tous les bénéfices du fine-tuning
- ✅ **Partageable**: Peut être uploadé en privé ou public sur HF

## Activation de l'environnement

Pour utiliser les commandes manuellement:

```bash
source venv/bin/activate
```

## ⚠️ Note importante : Fusion avec modèles quantizés (GPT-OSS)

### Problème identifié

Lors du fine-tuning de **modèles déjà quantizés** (comme `gpt-oss-20b-MXFP4-Q8`), la fusion directe avec `mlx_lm.fuse` **perd les comportements appris** :

- ✅ **Adaptateurs + modèle de base** : Réponses correctes
- ❌ **Fusion directe** : Réponses incorrectes (comportement comme si non fine-tuné)

**Cause** : La fusion avec des modèles quantizés n'est **pas numériquement stable** ([Issue #654](https://github.com/ml-explore/mlx-lm/issues/654)). Le quantizer MXFP4 est moins précis que le quantizer affine utilisé par `mlx_lm.convert`.

### Solution : Fusion en 2 étapes

Pour les modèles quantizés, utilisez cette procédure en 2 étapes :

#### Étape 1 : Fusionner en dequantizant vers FP16

```bash
source venv/bin/activate
python -m mlx_lm fuse \
  --model "mlx-community/gpt-oss-20b-MXFP4-Q8" \
  --adapter-path ./adapters \
  --save-path ./fused_fp16 \
  --dequantize
```

#### Étape 2 : Re-quantizer avec le quantizer affine

```bash
python -m mlx_lm convert \
  --hf-path ./fused_fp16 \
  --mlx-path ./fused_model \
  --quantize \
  --q-bits 8 \
  --q-group-size 64
```

#### Nettoyage (optionnel)

```bash
rm -rf fused_fp16  # Économise ~39 GB
```

### Résultats de la fusion en 2 étapes

| Méthode | Comportement | VRAM | Vitesse | Taille |
|---------|--------------|------|---------|--------|
| Modèle base + adaptateurs | ✅ Correct | 12.6 GB | 86 tok/s | 11 GB + 282 MB |
| Fusion directe | ❌ Incorrect | 12.2 GB | 129 tok/s | 11 GB |
| **Fusion 2 étapes** | ✅ Correct | 22.3 GB | 97 tok/s | 21 GB |

### Modèles concernés

Cette procédure en 2 étapes est **recommandée** pour tous les modèles déjà quantizés :
- GPT-OSS 20B (MXFP4-Q8, MXFP4-Q4)
- Mistral 7B (4-bit)
- Llama (4-bit, 8-bit)

Pour les modèles **non quantizés**, la fusion directe fonctionne normalement.

### Référence

Solution confirmée et documentée dans [mlx-lm Issue #654](https://github.com/ml-explore/mlx-lm/issues/654) (résolu décembre 2025).

## Référence

Basé sur la session WWDC 2025 #298 : "Fine-tune language models on Mac with MLX"
https://developer.apple.com/videos/play/wwdc2025/298/

## Licence

Ce projet est un exemple éducatif. Le modèle de base Mistral appartient à Mistral AI.

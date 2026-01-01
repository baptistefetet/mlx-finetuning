#!/bin/bash

# Script unifié pour le fine-tuning MLX
# Usage: ./mlx_tool.sh [COMMAND] [OPTIONS]

set -e

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration par défaut
DEFAULT_MODEL="mlx-community/Mistral-7B-Instruct-v0.3-4bit"
DEFAULT_DATA_DIR="./data"
DEFAULT_ADAPTER_PATH="./adapters"
DEFAULT_FUSED_MODEL_PATH="./fused_model"
DEFAULT_ITERS=300
DEFAULT_BATCH_SIZE=2
DEFAULT_LEARNING_RATE=1e-5
DEFAULT_MAX_TOKENS=100

# Fonction d'aide
show_help() {
    cat << EOF
${GREEN}MLX Fine-Tuning Tool${NC}
Script unifié pour le fine-tuning de modèles de langage avec MLX

${YELLOW}USAGE:${NC}
    ./mlx_tool.sh COMMAND [OPTIONS]

${YELLOW}COMMANDS:${NC}
    ${BLUE}train${NC}       Entraîner un modèle avec LoRA
    ${BLUE}generate${NC}    Générer du texte avec un modèle
    ${BLUE}fuse${NC}        Fusionner les adaptateurs LoRA dans le modèle de base
    ${BLUE}upload${NC}      Uploader un modèle sur Hugging Face
    ${BLUE}help${NC}        Afficher cette aide

${YELLOW}OPTIONS TRAIN:${NC}
    --model MODEL           Modèle de base (défaut: $DEFAULT_MODEL)
    --data DIR              Dossier de données (défaut: $DEFAULT_DATA_DIR)
    --adapter-path PATH     Chemin de sauvegarde des adaptateurs (défaut: $DEFAULT_ADAPTER_PATH)
    --iters N               Nombre d'itérations (défaut: $DEFAULT_ITERS)
    --batch-size N          Taille de batch (défaut: $DEFAULT_BATCH_SIZE)
    --learning-rate LR      Taux d'apprentissage (défaut: $DEFAULT_LEARNING_RATE)

${YELLOW}OPTIONS GENERATE:${NC}
    --model MODEL           Modèle de base (défaut: $DEFAULT_MODEL)
    --adapter-path PATH     Chemin des adaptateurs (optionnel)
    --prompt TEXT           Prompt à utiliser (requis)
    --max-tokens N          Nombre max de tokens (défaut: $DEFAULT_MAX_TOKENS)

${YELLOW}OPTIONS FUSE:${NC}
    --model MODEL           Modèle de base (défaut: $DEFAULT_MODEL)
    --adapter-path PATH     Chemin des adaptateurs (défaut: $DEFAULT_ADAPTER_PATH)
    --output PATH           Chemin du modèle fusionné (défaut: $DEFAULT_FUSED_MODEL_PATH)

${YELLOW}OPTIONS UPLOAD:${NC}
    --model-path PATH       Chemin du modèle local à uploader (requis)
    --repo NAME             Nom du repo HF (format: username/repo-name) (requis)
    --private               Rendre le repo privé (optionnel)

${YELLOW}EXEMPLES:${NC}
    # Entraîner un modèle
    ./mlx_tool.sh train --data ./data --iters 500

    # Générer avec le modèle fine-tuné
    ./mlx_tool.sh generate --adapter-path ./adapters --prompt "Quand est morte Brigitte Bardot ?"

    # Fusionner les adaptateurs
    ./mlx_tool.sh fuse --output ./my_fused_model

    # Uploader sur HF en privé
    ./mlx_tool.sh upload --model-path ./my_fused_model --repo bfetet/mistral-bb-finetuned --private

    # Générer avec le modèle fusionné (sans adaptateurs)
    ./mlx_tool.sh generate --model ./my_fused_model --prompt "Qui était BB ?"

EOF
}

# Activer l'environnement virtuel
activate_venv() {
    if [ ! -d "venv" ]; then
        echo -e "${RED}Erreur: Environnement virtuel 'venv' non trouvé${NC}"
        exit 1
    fi
    source venv/bin/activate
}

# Commande TRAIN
cmd_train() {
    local model="$DEFAULT_MODEL"
    local data_dir="$DEFAULT_DATA_DIR"
    local adapter_path="$DEFAULT_ADAPTER_PATH"
    local iters=$DEFAULT_ITERS
    local batch_size=$DEFAULT_BATCH_SIZE
    local learning_rate=$DEFAULT_LEARNING_RATE

    while [[ $# -gt 0 ]]; do
        case $1 in
            --model) model="$2"; shift 2 ;;
            --data) data_dir="$2"; shift 2 ;;
            --adapter-path) adapter_path="$2"; shift 2 ;;
            --iters) iters="$2"; shift 2 ;;
            --batch-size) batch_size="$2"; shift 2 ;;
            --learning-rate) learning_rate="$2"; shift 2 ;;
            *) echo -e "${RED}Option inconnue: $1${NC}"; exit 1 ;;
        esac
    done

    echo -e "${GREEN}=== Entraînement LoRA ===${NC}"
    echo -e "Modèle: ${BLUE}$model${NC}"
    echo -e "Données: ${BLUE}$data_dir${NC}"
    echo -e "Adaptateurs: ${BLUE}$adapter_path${NC}"
    echo -e "Itérations: ${BLUE}$iters${NC}"
    echo -e "Batch size: ${BLUE}$batch_size${NC}"
    echo -e "Learning rate: ${BLUE}$learning_rate${NC}"
    echo ""

    activate_venv
    python -m mlx_lm lora \
        --model "$model" \
        --train \
        --data "$data_dir" \
        --iters "$iters" \
        --batch-size "$batch_size" \
        --learning-rate "$learning_rate" \
        --adapter-path "$adapter_path"

    echo -e "${GREEN}✓ Entraînement terminé !${NC}"
}

# Commande GENERATE
cmd_generate() {
    local model="$DEFAULT_MODEL"
    local adapter_path=""
    local prompt=""
    local max_tokens=$DEFAULT_MAX_TOKENS

    while [[ $# -gt 0 ]]; do
        case $1 in
            --model) model="$2"; shift 2 ;;
            --adapter-path) adapter_path="$2"; shift 2 ;;
            --prompt) prompt="$2"; shift 2 ;;
            --max-tokens) max_tokens="$2"; shift 2 ;;
            *) echo -e "${RED}Option inconnue: $1${NC}"; exit 1 ;;
        esac
    done

    if [ -z "$prompt" ]; then
        echo -e "${RED}Erreur: --prompt est requis${NC}"
        exit 1
    fi

    echo -e "${GREEN}=== Génération ===${NC}"
    echo -e "Modèle: ${BLUE}$model${NC}"
    if [ -n "$adapter_path" ]; then
        echo -e "Adaptateurs: ${BLUE}$adapter_path${NC}"
    fi
    echo -e "Prompt: ${BLUE}$prompt${NC}"
    echo ""

    activate_venv

    if [ -n "$adapter_path" ]; then
        python -m mlx_lm generate \
            --model "$model" \
            --adapter-path "$adapter_path" \
            --prompt "[INST] $prompt [/INST]" \
            --max-tokens "$max_tokens"
    else
        python -m mlx_lm generate \
            --model "$model" \
            --prompt "[INST] $prompt [/INST]" \
            --max-tokens "$max_tokens"
    fi
}

# Commande FUSE
cmd_fuse() {
    local model="$DEFAULT_MODEL"
    local adapter_path="$DEFAULT_ADAPTER_PATH"
    local output="$DEFAULT_FUSED_MODEL_PATH"

    while [[ $# -gt 0 ]]; do
        case $1 in
            --model) model="$2"; shift 2 ;;
            --adapter-path) adapter_path="$2"; shift 2 ;;
            --output) output="$2"; shift 2 ;;
            *) echo -e "${RED}Option inconnue: $1${NC}"; exit 1 ;;
        esac
    done

    echo -e "${GREEN}=== Fusion des adaptateurs ===${NC}"
    echo -e "Modèle de base: ${BLUE}$model${NC}"
    echo -e "Adaptateurs: ${BLUE}$adapter_path${NC}"
    echo -e "Sortie: ${BLUE}$output${NC}"
    echo ""

    activate_venv
    python -m mlx_lm fuse \
        --model "$model" \
        --adapter-path "$adapter_path" \
        --save-path "$output"

    echo -e "${GREEN}✓ Fusion terminée !${NC}"
    echo -e "Modèle fusionné disponible dans: ${BLUE}$output${NC}"
}

# Commande UPLOAD
cmd_upload() {
    local model_path=""
    local repo=""
    local private_flag=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --model-path) model_path="$2"; shift 2 ;;
            --repo) repo="$2"; shift 2 ;;
            --private) private_flag="--private"; shift ;;
            *) echo -e "${RED}Option inconnue: $1${NC}"; exit 1 ;;
        esac
    done

    if [ -z "$model_path" ] || [ -z "$repo" ]; then
        echo -e "${RED}Erreur: --model-path et --repo sont requis${NC}"
        exit 1
    fi

    echo -e "${GREEN}=== Upload sur Hugging Face ===${NC}"

    activate_venv
    python upload_to_hf.py \
        --model-path "$model_path" \
        --repo "$repo" \
        $private_flag
}

# Main
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

COMMAND=$1
shift

case $COMMAND in
    train) cmd_train "$@" ;;
    generate) cmd_generate "$@" ;;
    fuse) cmd_fuse "$@" ;;
    upload) cmd_upload "$@" ;;
    help) show_help ;;
    *) echo -e "${RED}Commande inconnue: $COMMAND${NC}"; show_help; exit 1 ;;
esac

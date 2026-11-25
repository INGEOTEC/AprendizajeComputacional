#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=10G
#SBATCH -p compute
#SBATCH -c 1
#SBATCH -n 1
#SBATCH --time=02:00:00
# Envoltura de Slurm para los scripts de datos que necesitan GPU
# (delitos_qwen.py y delitos_juez.py), con las directivas y el intérprete de
# enviar.sh del proyecto Chimalli. Recibe el script y sus argumentos. Se envía
# desde la raíz del repositorio, SIEMPRE con --output fuera del repositorio
# (un slurm-*.out en la raíz ensucia el árbol y puede citar tuits):
#
#   mkdir -p ~/.cache/aprendizaje-computacional/slurm
#   sbatch --parsable \
#     --output "$HOME/.cache/aprendizaje-computacional/slurm/%x-%j.out" \
#     scripts/datos/enviar_gpu.sh scripts/datos/delitos_juez.py \
#     --modelo Qwen/Qwen3.5-4B --lote 128
#
# El modelo se descarga antes desde el nodo de acceso con
# scripts/datos/modelos_hf.py, porque los nodos de cómputo no tienen red hacia
# el Hub (HF_HUB_OFFLINE=True).
export HF_HUB_OFFLINE=True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Intérprete fijo: Slurm copia este script a su propio directorio antes de
# correrlo y hereda el PATH de quien lo envía. Es el entorno compartido con
# LegalIA y Chimalli; no se instala nada en él.
PYTHON="/home/mgraffg/.venvs/cluster/bin/python"

# Un error transitorio de CUDA puede matar la corrida; los scripts se reanudan,
# así que se reintenta hasta tres veces.
max_intentos=3
intento=1
until "$PYTHON" "$@"; do
    estado=$?
    if [ "$intento" -ge "$max_intentos" ]; then
        exit "$estado"
    fi
    echo "python $* falló (intento $intento/$max_intentos, código $estado); reintentando..." >&2
    intento=$((intento + 1))
    sleep 30
done

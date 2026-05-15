#!/usr/bin/env bash
# Crea un proyecto con uv (flujo oficial: uv init + uv add)
# e instala dependencias en el entorno .venv del proyecto.
#
# Uso recomendado (para activar .venv automaticamente al final):
#   source setup_uv_project.sh
#
# Opciones:
#   source setup_uv_project.sh --python 3.12
#   source setup_uv_project.sh --base-dir "/ruta/existente"
#   source setup_uv_project.sh --no-activate

IS_SOURCED=0
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    IS_SOURCED=1
fi

# Si el script se ejecuta con source, no dejamos opciones de shell
# (como -e/-u/-o pipefail) persistentes en la terminal del usuario.
ORIGINAL_SHELL_OPTS="$(set +o)"
set -euo pipefail

script_exit() {
    local code="${1:-0}"
    eval "$ORIGINAL_SHELL_OPTS"
    if [[ "$IS_SOURCED" -eq 1 ]]; then
        return "$code"
    fi
    exit "$code"
}

show_help() {
    echo "Uso: source setup_uv_project.sh [opciones]"
    echo ""
    echo "Inicializa el proyecto en el directorio indicado por --base-dir (default: actual)."
    echo "Por defecto, el nombre del proyecto sera el nombre de ese directorio."
    echo ""
    echo "Opciones:"
    echo "  --python <version>     Version de Python (ej: 3.12)"
    echo "  --base-dir <ruta>      Directorio donde inicializar el proyecto (default: .)"
    echo "  --no-activate          No activar .venv al terminar"
    echo "  -h, --help             Mostrar ayuda"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    show_help
    script_exit 0
fi

PYTHON_VERSION=""
BASE_DIR="."
AUTO_ACTIVATE="1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --python requiere un valor."
                script_exit 1
            fi
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --base-dir)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --base-dir requiere una ruta."
                script_exit 1
            fi
            BASE_DIR="$2"
            shift 2
            ;;
        --no-activate)
            AUTO_ACTIVATE="0"
            shift
            ;;
        *)
            echo "ERROR: Opcion no reconocida: $1"
            show_help
            script_exit 1
            ;;
    esac
done

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' no esta instalado o no esta en el PATH."
    echo "Instalacion: https://docs.astral.sh/uv/getting-started/installation/"
    script_exit 1
fi

if ! BASE_DIR="$(cd "$BASE_DIR" && pwd)"; then
    echo "ERROR: --base-dir no es valido: $BASE_DIR"
    script_exit 1
fi

if [[ -f "$BASE_DIR/pyproject.toml" ]]; then
    echo "ERROR: Ya existe un pyproject.toml en: $BASE_DIR"
    echo "Ese directorio ya parece ser un proyecto."
    script_exit 1
fi

PROJECT_NAME="$(basename "$BASE_DIR")"
echo "==> Creando proyecto en directorio actual: $BASE_DIR"
echo "==> Nombre del proyecto (por directorio): $PROJECT_NAME"
if [[ -n "$PYTHON_VERSION" ]]; then
    uv init --directory "$BASE_DIR" --python "$PYTHON_VERSION"
else
    uv init --directory "$BASE_DIR"
fi

echo "==> Anadiendo dependencias con uv add"
cd "$BASE_DIR"
uv add pandas numpy matplotlib ipykernel torch scikit-learn seaborn openpyxl

echo ""
echo "==> Proyecto listo en: $BASE_DIR"

if [[ "$AUTO_ACTIVATE" == "0" ]]; then
    echo "Activacion manual:"
    echo "  cd \"$BASE_DIR\" && source .venv/bin/activate"
    script_exit 0
fi

# Activacion automatica solo si el script se ejecuta con source.
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "==> Activando entorno .venv..."
    source .venv/bin/activate
    echo "Entorno activado. Para salir: deactivate"
else
    echo "Para activar automaticamente el entorno, ejecuta:"
    echo "  source setup_uv_project.sh"
    echo "Si no, activa manualmente con:"
    echo "  cd \"$BASE_DIR\" && source .venv/bin/activate"
fi

script_exit 0
